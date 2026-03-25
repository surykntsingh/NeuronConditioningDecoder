import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import BioGptTokenizer, BioGptForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from accelerate import Accelerator
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

BASE_PATH = '/mnt/surya/dataset/rsna_pneumonia/rsna_pnumonia_challenge/data/kaggle'
train_data_dir = f'{BASE_PATH}/stage_2_train_images'


class CbmActivationDataset(Dataset):
    def __init__(self, data_path, tokenizer, repeat_factor=100, max_length=40):
        """
        data_path should contain:
            {
                "text_embeddings": tensor [K, D],
                "concept_texts": list[str]
            }
        """
        self.tokenizer = tokenizer
        data = torch.load(data_path, map_location="cpu")

        activations = data["activations"]  # .float()  # CLIP similarity
        images = data["image_embeddings"]  # .float()
        concepts = data["concepts"]
        image_ids = data['image_ids']
        image_embeddings = []
        concept_embeddings = []
        concept_texts = []

        for image_id in image_ids:
            image_embeddings.append(images[image_id])

        for concept in concepts:
            concept_embeddings.append(concept['embeddings'])
            concept_texts.append(concept['texts'])

        image_embeddings = torch.stack(image_embeddings, dim=0).squeeze(1)
        concept_embeddings = torch.stack(concept_embeddings, dim=0).squeeze(1)

        reconstructed_text_embeddings = self.reconstruct_text_embeddings(image_embeddings, activations)
        reconstructed_text_embeddings = F.normalize(reconstructed_text_embeddings, dim=-1)

        print(f'reconstructed_text_embeddings: {reconstructed_text_embeddings.shape}')

        reconstructed_concepts = [
            {
                'text': concept_texts[i],
                'reconstructed_embedding': reconstructed_text_embeddings[i],
                'original_embedding': concept_embeddings[i]
            }

            for i, concept_text in enumerate(concept_texts)
        ]

        self.concepts = []
        for i in range(repeat_factor):
            self.concepts.extend(reconstructed_concepts)

        self.max_length = max_length

    def __len__(self):
        return len(self.concepts)

    def reconstruct_text_embeddings(self, F, Z):
        """
        F: [N, D] image embeddings (normalized)
        Z: [N, K] similarity matrix

        Returns:
            T_hat: [K, D] reconstructed text embeddings
        """

        # Compute pseudo-inverse solution
        # (F^T F)^-1 F^T Z

        Ft = F.T  # [D, N]
        FtF = Ft @ F  # [D, D]

        # Regularization for stability
        lambda_reg = 1e-4
        FtF_reg = FtF + lambda_reg * torch.eye(FtF.size(0), device=F.device)

        FtF_inv = torch.linalg.inv(FtF_reg)  # [D, D]

        T_hat = FtF_inv @ Ft @ Z  # [D, K]
        T_hat = T_hat.T  # [K, D]

        return T_hat

    def __getitem__(self, idx):

        item = self.concepts[idx]
        text = item['text']
        embedding = item['reconstructed_embedding'].squeeze(0)
        original_embedding = item['original_embedding'].squeeze(0)

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        return {
            "embedding": embedding,
            "original_embedding": original_embedding,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label": text
        }


class BootstrappedConceptDataset(Dataset):
    def __init__(
            self,
            data_path,
            tokenizer,
            num_bootstrap=100,
            subset_size=500,
            max_length=40,
            add_noise_std=0.0,
            normalize_embedding=True,
            device='cpu'
    ):
        """
        data_path should contain:
            - image_embeddings: [N, D]
            - activations: [N, K]
            - concept_texts: list of K strings
            - concept_embeddings: [K, D]  (IMPORTANT)
        """

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        data = torch.load(data_path, map_location=device)
        Z_s = data["activations"]  # .float()  # CLIP similarity
        images = data["image_embeddings"]  # .float()
        concepts = data["concepts"]
        image_ids = data['image_ids']
        image_embeddings = []
        concept_embeddings = []
        concept_texts = []

        z = len(Z_s)

        for image_id in image_ids:
            image_embeddings.append(images[image_id])

        for concept in concepts:
            concept_embeddings.append(concept['embeddings'])
            concept_texts.append(concept['texts'])

        F_all = torch.stack(image_embeddings, dim=0).squeeze(1)
        T_all = torch.stack(concept_embeddings, dim=0).squeeze(1)

        # F_all = data["image_embeddings"]      # [N, D]
        # Z_all = activations # data["activations"]           # [N, K]
        # concept_texts = data["concept_texts"]
        # T_all = data["concept_embeddings"]    # [K, D]

        # Normalize image embeddings (IMPORTANT)
        F_all = F.normalize(F_all, dim=-1)

        N, D = F_all.shape
        K = Z_s[0].shape[1]

        print(f"N={N}, D={D}, K={K}")
        print(f"Bootstrapping {num_bootstrap} samples per concept...")

        loop = tqdm(range(z), leave=False, total=z)
        for i in loop:
            for k in range(K):
                Z_all = Z_s[i]

                z_k = Z_all[:, k]  # [N]
                text = concept_texts[k]
                true_embedding = T_all[k]  # [D]

                for b in range(num_bootstrap):

                    # ---- Sample subset ----
                    indices = torch.randint(0, N, (subset_size,))

                    F_b = F_all[indices]  # [subset, D]
                    z_b = z_k[indices]  # [subset]

                    # ---- Reconstruction: v = Fᵀ z ----
                    Ft = F_b.T  # [D, subset]
                    FtF = Ft @ F_b  # [D, D]

                    lambda_reg = 1e-2
                    FtF_reg = FtF + lambda_reg * torch.eye(D, device=device)

                    FtF_inv = torch.linalg.inv(FtF_reg, device=device)

                    v = FtF_inv @ Ft @ z_b  # [D]
                    # ---- Optional noise ----
                    if add_noise_std > 0:
                        v = v + add_noise_std * torch.randn_like(v, device=device)

                    # ---- Normalize (recommended) ----
                    if normalize_embedding:
                        v = F.normalize(v, dim=-1)

                    self.samples.append({
                        "reconstructed_embedding": v,
                        "text": text,
                        "original_embedding": true_embedding
                    })

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]
        text = item['text']
        embedding = item['reconstructed_embedding'].squeeze(0)
        original_embedding = item['original_embedding'].squeeze(0)

        tokenized = self.tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return {
            "embedding": embedding,
            "original_embedding": original_embedding,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label": text
        }


class ContrastiveEmbeddingConditionedDecoder(nn.Module):
    def __init__(self, embed_dim, lora_r=16, temperature=0.07, con_lambda=4, dropout=0.2,
                 lm_name="microsoft/biogpt"):
        super().__init__()
        self.con_lambda = con_lambda
        self.tokenizer = BioGptTokenizer.from_pretrained(lm_name)
        base_model = BioGptForCausalLM.from_pretrained(lm_name)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.lm = get_peft_model(base_model, lora_config)

        for name, param in self.lm.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        self.hidden_dim = self.lm.config.hidden_size
        self.temperature = temperature
        self.scale = nn.Parameter(torch.tensor(10.0))

        # Projection WITHOUT tanh
        self.proj = nn.Linear(embed_dim, self.hidden_dim)

        hidden_dim = 2 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embed_dim)
        )

    def compute_condition_token(self, embedding):
        raw = self.net(embedding)
        raw = self.proj(raw)
        cond_vec = F.normalize(raw, dim=-1)
        cond_vec = cond_vec * self.scale
        return cond_vec

    def forward(self, embedding, input_ids, attention_mask):

        B = embedding.size(0)

        # ---------------------------
        # 1️⃣ Project embedding
        # ---------------------------
        cond_vec = self.compute_condition_token(embedding)

        cond_token = cond_vec.unsqueeze(1)

        # ---------------------------
        # 2️⃣ Standard LM Forward
        # ---------------------------
        token_embeds = self.lm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([cond_token, token_embeds], dim=1)

        cond_mask = torch.ones(B, 1, device=attention_mask.device)
        attention_mask = torch.cat([cond_mask, attention_mask], dim=1)

        cond_labels = torch.full((B, 1), -100, device=input_ids.device)
        labels = torch.cat([cond_labels, input_ids], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        lm_loss = outputs.loss

        # ---------------------------
        # 3️⃣ Extract text representations
        # ---------------------------
        # Use last hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len+1, hidden]

        # Remove conditioning token position
        text_hidden = last_hidden[:, 1:, :]  # [B, seq_len, hidden]

        # Mean pool text tokens
        text_repr = text_hidden.mean(dim=1)  # [B, hidden]
        text_repr = F.normalize(text_repr, dim=-1)

        # ---------------------------
        # 4️⃣ Contrastive InfoNCE Loss
        # ---------------------------
        logits = cond_vec @ text_repr.T  # [B, B]
        logits = logits / self.temperature

        labels_contrast = torch.arange(B, device=embedding.device)

        contrast_loss = F.cross_entropy(logits, labels_contrast)

        # ---------------------------
        # 5️⃣ Combined Loss
        # ---------------------------
        total_loss = lm_loss + self.con_lambda * contrast_loss

        return total_loss, lm_loss.detach(), contrast_loss.detach()

class LoRAConditionTokenDecoder(nn.Module):
    def __init__(self, image_dim, lora_r=8, lm_name="microsoft/biogpt"):
        super().__init__()

        self.tokenizer = BioGptTokenizer.from_pretrained(lm_name)
        base_model = BioGptForCausalLM.from_pretrained(lm_name)

        # ---- LoRA ----
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.lm = get_peft_model(base_model, lora_config)

        # Freeze base weights
        for name, param in self.lm.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        self.hidden_dim = self.lm.config.hidden_size

        # ---- FiLM modulation ----
        self.gamma_proj = nn.Linear(1, image_dim)
        self.beta_proj  = nn.Linear(1, image_dim)

        # ---- Project to LM hidden dim ----
        self.cond_proj = nn.Sequential(
            nn.Linear(image_dim, self.hidden_dim),
            nn.Tanh()
        )

    def forward(self, image_embeddings, z_k, input_ids, attention_mask):

        B = image_embeddings.size(0)

        if image_embeddings.dim() == 3:
            image_embeddings = image_embeddings.squeeze(1)

        # ---- FiLM ----
        gamma = torch.tanh(self.gamma_proj(z_k)) * 0.1
        beta  = torch.tanh(self.beta_proj(z_k))  * 0.1

        h_mod = (1 + gamma) * image_embeddings + beta

        # ---- Conditioning token embedding ----
        cond_token = self.cond_proj(h_mod)  # [B, hidden_dim]
        cond_token = cond_token.unsqueeze(1)  # [B, 1, hidden_dim]

        # ---- Text token embeddings ----
        token_embeds = self.lm.get_input_embeddings()(input_ids)

        # ---- Concatenate ----
        inputs_embeds = torch.cat([cond_token, token_embeds], dim=1)

        # ---- Adjust attention mask ----
        cond_mask = torch.ones(B, 1, device=attention_mask.device)
        attention_mask = torch.cat([cond_mask, attention_mask], dim=1)

        # ---- Labels (ignore cond token position) ----
        cond_labels = torch.full((B, 1), -100, device=input_ids.device)
        labels = torch.cat([cond_labels, input_ids], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss


def train_decoder(model, dataloader,accelerator, epochs=10, lr=1e-6):

    device = accelerator.device
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(dataloader, leave=False, total=len(dataloader))

        for batch in loop:
            image_embeddings = batch["image_embedding"]#.to(device)
            z_k = batch["z_k"]#.to(device)
            input_ids = batch["input_ids"]#.to(device)
            attention_mask = batch["attention_mask"]#.to(device)

            optimizer.zero_grad()

            loss = model(
                image_embeddings,
                z_k,
                input_ids,
                attention_mask
            )

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    return model        

def train_decoder(model, dataloader, accelerator, epochs=10, lr=1e-6, weight_decay=0.01):
    device = accelerator.device
    model.to(device)
    # model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    for epoch in range(epochs):

        # ------------------
        # Training
        # ------------------
        model.train()
        total_loss = 0
        total_lm_loss = 0
        total_contrast_loss = 0

        loop = tqdm(dataloader, total=len(dataloader))

        for batch in loop:

            optimizer.zero_grad()


            loss, lm_loss, contrast_loss = model(
                embedding=batch["embedding"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_contrast_loss += contrast_loss.item()
            total_lm_loss += lm_loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(dataloader)
        lm_loss = total_lm_loss / len(dataloader)
        contrast_loss = total_contrast_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f}| LM: {lm_loss:.4f} | Contrast: {contrast_loss:.4f}")

        return model


def generate_from_embedding(model, embedding, tokenizer, device='cuda'):
    model.eval()
    model.to(device)

    with torch.no_grad():
        embedding = embedding.unsqueeze(0).to(device)
        # raw = model.proj[0](embedding)
        # print(raw.mean(), raw.std())

        # cond_token = model.proj(embedding).unsqueeze(1)
        # cond_token = F.normalize(cond_token, dim=-1)* model.scale
        # cond_token = torch.zeros_like(cond_token)

        cond_vec = model.compute_condition_token(embedding)

        cond_token = cond_vec.unsqueeze(1)

        prompt = ""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        token_embeds = model.lm.get_input_embeddings()(inputs["input_ids"])

        print(
            f'cond_token: {cond_token.mean()} {cond_token.std()} token_embeds:{token_embeds.mean()} {token_embeds.std()}')

        inputs_embeds = torch.cat([cond_token, token_embeds], dim=1)

        attention_mask = torch.cat([
            torch.ones(1, 1, device=device),
            inputs["attention_mask"]
        ], dim=1)

        outputs = model.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

class EmbeddingEvaluator:
    def __init__(self, model_name='dmis-lab/biobert-v1.1'):
        assert model_name in [
            'dmis-lab/biobert-v1.1',
            'aaditya/Llama3-OpenBioLLM-8B',
            'openai-communitya/gpt2',
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'NeuML/pubmedbert-base-embeddings'
        ], f'Not Possible {model_name}.'
        self.model_name = model_name
        self._get_embedding_model()

    def _get_embedding_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    @torch.no_grad()
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_score(self, ref_text, hyp_text, scale=.5):
        ref_embedding = self.get_embedding(ref_text)
        hyp_embedding = self.get_embedding(hyp_text)
        score = cosine_similarity([ref_embedding], [hyp_embedding])[0][0]
        if (scale != 0) and (score > scale):
            score = (score - scale) / (1 - scale)
        return score





def get_emb_similarity_score(embedding_evaluator, results):
    scores = []
    for r in results:
        score = embedding_evaluator.get_score(r['original_concept'], r['predicted_concept'])
        # print(score)
        scores.append(score)

    return scores, sum(scores) / len(scores)

def evaluate(data_path_x, test_model):
    dsx = CbmActivationDataset(data_path_x, tokenizer, repeat_factor=1)
    test_model.eval()
    result = []
    for i in range(len(dsx)):
        item = ds[i]
        predicted = generate_from_embedding(test_model, item['embedding'], tokenizer)
        # print(f'gt: {item["label"]}, predicted: {predicted}')
        result.append({'original_concept': item["label"], 'predicted_concept': predicted})

    scores, mean_score = get_emb_similarity_score(embedding_evaluator, result)
    print(f'{data_path_x} mean_score: {mean_score}')


if __name__=="__main__":
    base_path = '/mnt/surya/projects/Wsi-rgen/notebooks'
    # data_path = base_path +'/'+ 'cbmad_outputs/clip_activations.pt'
    data_path = base_path +'/'+'cbmad_outputs/rsna_multi_neuron_activations.pt'
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    batch_size = 32
    ds = BootstrappedConceptDataset(data_path, tokenizer, num_bootstrap=50, subset_size=5000, add_noise_std=0.001, device='cpu')

    # ds = CbmActivationDataset(data_path, tokenizer)
    print(len(ds))

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model = ContrastiveEmbeddingConditionedDecoder(
        embed_dim=512,
        lora_r=32
    )



    # model.to(device)
    model = train_decoder(model, dataloader, accelerator, epochs=50, lr=5e-6)
    accelerator.wait_for_everyone()


    unwrapped_model = accelerator.unwrap_model(model)
    model_path = "models/decoder/lang_decoder_multigpu.pt"
    torch.save(unwrapped_model.state_dict(), model_path)

    print(f'evaluating...')
    embedding_evaluator = EmbeddingEvaluator()


    test_model = ContrastiveEmbeddingConditionedDecoder(
        embed_dim=512,
        lora_r=32
    )
    test_model.load_state_dict(torch.load(model_path))

    data_path_x = 'cbmad_outputs/neuron_activations_new.pt'

    evaluate(base_path +'/'+'cbmad_outputs/test_cbm_activations_3.pt', test_model)
    evaluate(base_path +'/'+'cbmad_outputs/neuron_activations_new.pt', test_model)
    evaluate(base_path +'/'+'cbmad_outputs/chexperts_neuron_activations.pt', test_model)
    evaluate(base_path +'/'+'cbmad_outputs/chexperts_neuron_activations_new_cbm.pt', test_model)







