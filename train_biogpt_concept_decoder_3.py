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
    def __init__(self, data_path, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer

        data_dict = torch.load(data_path, map_location="cpu")

        concept_matrix = data_dict["activations"]
        image_embeddings = data_dict["image_embeddings"]
        concept_texts = data_dict["concept_texts"]

        N, K = concept_matrix.shape

        # Mild scaling (no z-score)
        concept_matrix = concept_matrix * 5.0
        concept_matrix = torch.clamp(concept_matrix, -1.5, 1.5)

        for i in range(N):
            for k in range(K):

                z = concept_matrix[i, k].item()
                concept_name = concept_texts[k]

                if z > 0.5:
                    text = f"Radiographic finding: Clear evidence of {concept_name}."
                elif z < -0.5:
                    text = f"Radiographic finding: No evidence of {concept_name}."
                else:
                    text = f"Radiographic finding: Possible {concept_name}."

                self.samples.append({
                    "image_embedding": image_embeddings[i],
                    "z_k": torch.tensor([z], dtype=torch.float32),
                    "text": text
                })

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
        }

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




if __name__=="__main__":
    base_path = '/mnt/surya/projects/Wsi-rgen/notebooks'
    data_path = base_path +'/'+ 'cbmad_outputs/clip_activations.pt'

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
    model = LoRAConditionTokenDecoder(
            image_dim=512,
            lora_r=8
        )

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # model.to(device)
    model = train_decoder(model, dataloader, accelerator, epochs=5)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), "models/decoder/neuron_decoder_lora_multigpu_4.pt")


