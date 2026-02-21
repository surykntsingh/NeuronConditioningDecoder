import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import BioGptTokenizer, BioGptForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from accelerate import Accelerator

BASE_PATH = '/mnt/surya/dataset/rsna_pneumonia/rsna_pnumonia_challenge/data/kaggle'
train_data_dir = f'{BASE_PATH}/stage_2_train_images'

class ConceptDecoderDataset(Dataset):
    def __init__(self, data_path, tokenizer, top_k=0.2):
        """
        image_embeddings: [N, D]
        concept_matrix: [N, K]
        glossary_texts: list of length K
        """
        # self.__backbone_utils = backbone_utils
        # self.__image_size = image_size
        
        self.samples = []
        self.tokenizer = tokenizer


        data_dict = torch.load(data_path)

        image_ids = data_dict['image_ids']
        concept_texts = data_dict['concept_texts']
        concept_matrix = data_dict['activations']
        image_embeddings = data_dict['image_embeddings']

        N,K = len(image_ids), len(concept_texts)
        print(f'concept_matrix.shape: {concept_matrix.shape}, (N,K): {(N,K)}')
        assert(concept_matrix.shape == (N,K))
        N, K = concept_matrix.shape

        for k in range(K):
            concept_scores = concept_matrix[:, k]
            threshold = torch.quantile(concept_scores, 1 - top_k)

            indices = torch.where(concept_scores >= threshold)[0]

            for idx in indices:
                self.samples.append({
                    "image_embedding": image_embeddings[image_ids[idx]],
                    "z_k": concept_scores[idx].unsqueeze(0),
                    "text": concept_texts[k]
                })

    def get_transformation(self):
        return transforms.Compose([
                transforms.Resize(self.__image_size),
                transforms.Grayscale(num_output_channels=1)
            ])
    
    # def __get_image_embedding(self, image_id):
    #     img = self.__backbone_utils.read_image(image_id)
    #     img = self.get_transformation()(img)
    #     img_embedding = self.__backbone_utils.get_image_embedding(img)
    #     return img_embedding

        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokenized = self.tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40
        )

        return {
            "image_embedding": item["image_embedding"],
            "z_k": item["z_k"],
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            'label': item["text"]
        }

class LoRANeuronConditionedDecoder(nn.Module):
    def __init__(self, image_dim,lora_r, prefix_length=5, lm_name="microsoft/biogpt"):
        super().__init__()

        self.tokenizer = BioGptTokenizer.from_pretrained(lm_name)
        base_model = BioGptForCausalLM.from_pretrained(lm_name)
        self.prefix_scale = 1

        # ---- Apply LoRA ----
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

        self.prefix_length = prefix_length
        self.hidden_dim = self.lm.config.hidden_size

        # ---- FiLM layers ----
        self.gamma_proj = nn.Linear(1, image_dim)
        self.beta_proj = nn.Linear(1, image_dim)

        # ---- Neuron conditioning projection ----
        self.prefix_proj = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, prefix_length * self.hidden_dim)
        )
        # Precompute LM embedding std for scale matching
        with torch.no_grad():
            emb_std = self.lm.get_input_embeddings().weight.std()
            print(f'emb_std: {emb_std}')
        self.register_buffer("lm_embedding_std", emb_std)

    def forward(self, image_embeddings, z_k, input_ids, attention_mask):

        B = image_embeddings.size(0)
        image_embeddings = image_embeddings.squeeze(1)
        gamma = torch.tanh(self.gamma_proj(z_k))  # [-1, 1]
        beta  = torch.tanh(self.beta_proj(z_k))

        # Scale modulation strength (important for stability)
        gamma = 0.1 * gamma
        beta  = 0.1 * beta

        h_mod = (1 + gamma) * image_embeddings + beta

        # # ---- Concatenate neuron scalar ----
        # conditioning_input = torch.cat([image_embeddings, z_k], dim=1)
        # print(f'image_embeddings: {image_embeddings.shape}, gamma: {gamma.shape}, beta: {beta.shape}')
        prefix = self.prefix_proj(h_mod)

        # print(f'prefix: {prefix.shape}, self.hidden_dim: {self.hidden_dim}')

        prefix = prefix.view(B, self.prefix_length, self.hidden_dim)

        # ---- Stability normalization ----
        prefix = torch.tanh(prefix)  # bound values
        # prefix = F.normalize(prefix, dim=-1)  # unit norm per token
        prefix = prefix * self.prefix_scale #self.lm_embedding_std  # match LM scale

        # ---- Token embeddings ----
        token_embeds = self.lm.get_input_embeddings()(input_ids)

        # ---- Concatenate prefix tokens ----
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)


        # ---- Adjust attention mask ----
        prefix_mask = torch.ones(
            B, self.prefix_length,
            device=attention_mask.device
        )

        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Labels (ignore prefix tokens)
        prefix_labels = torch.full(
            (B, self.prefix_length),
            -100,
            device=input_ids.device
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)

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

            loss.backward()
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




if __name__=="__main__":
    
    data_path = 'cbmad_outputs/neuron_activations_new.pt'

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

    ds = ConceptDecoderDataset(data_path,tokenizer)


    batch_size = 32
    # device = 'cuda'
    dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    model = LoRANeuronConditionedDecoder(
            image_dim=512,
            prefix_length=2,
            lora_r=16
        )

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    model.to(device)
    model = train_decoder(model, dataloader, epochs=5, device=device)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), "models/decoder/neuron_decoder_lora_multigpu_1.pt")


