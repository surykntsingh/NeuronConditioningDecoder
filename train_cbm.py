import os
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
import torchmetrics

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class MedCLIPModelUtils():
    def __init__(self, base_path):
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel, PromptClassifier
        from medclip import MedCLIPProcessor
        self.__base_path = base_path
        self.__vlm_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.__vlm_model.from_pretrained()
        # self.__vlm_model.cuda()
        self.__processor = MedCLIPProcessor()

    def read_image(self, image_id):
        img_path = f'{self.__base_path}/{image_id}.dcm'
        dicom_image = pydicom.dcmread(img_path)
        pixels = dicom_image.pixel_array
        return Image.fromarray(pixels)

    def get_image_embedding(self, image):
        inputs = self.__processor(images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            image_embeddings = self.__vlm_model.encode_image(inputs["pixel_values"].cuda())

        return image_embeddings.squeeze(1)

    def get_text_embedding(self, text):
        text_inputs = self.__processor(text=text, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_embeddings = self.__vlm_model.encode_text(text_inputs["input_ids"])

        return text_embeddings

radiology_glossary_2 = {
  "pulmonary_opacities": [
    { "concept": "Consolidation", "variants": ["Pulmonary consolidation", "Dense airspace opacity", "Lobar consolidation"] },
    { "concept": "Ground-glass opacity", "variants": ["Ground-glass attenuation", "Hazy airspace opacity", "Subtle parenchymal haziness"] },
    { "concept": "Patchy airspace opacity", "variants": ["Patchy airspace disease", "Scattered airspace opacities", "Multifocal patchy opacity"] },
    { "concept": "Diffuse airspace disease", "variants": ["Diffuse bilateral opacities", "Widespread airspace disease", "Extensive parenchymal opacity"] },
    { "concept": "Focal opacity", "variants": ["Focal parenchymal opacity", "Localized lung opacity", "Isolated airspace opacity"] },
    { "concept": "Nodular opacity", "variants": ["Pulmonary nodule", "Rounded parenchymal opacity", "Discrete nodular density"] },
    { "concept": "Multiple nodules", "variants": ["Multiple pulmonary nodules", "Numerous nodular opacities", "Multifocal nodules"] },
    { "concept": "Mass-like opacity", "variants": ["Mass-like consolidation", "Large focal opacity", "Space-occupying pulmonary lesion"] },
    { "concept": "Cavitary lesion", "variants": ["Cavitary opacity", "Lucent center within opacity", "Cavity formation"] }
  ],

  "interstitial_patterns": [
    { "concept": "Reticular opacity", "variants": ["Reticular pattern", "Fine linear opacities", "Interstitial reticulation"] },
    { "concept": "Reticulonodular pattern", "variants": ["Reticulonodular opacities", "Mixed linear-nodular pattern", "Interstitial nodular markings"] },
    { "concept": "Septal thickening", "variants": ["Interlobular septal thickening", "Kerley lines", "Septal lines"] },
    { "concept": "Honeycombing", "variants": ["Honeycomb pattern", "Cystic subpleural spaces", "Advanced fibrotic pattern"] }
  ],

  "lung_volume_and_aeration": [
    { "concept": "Atelectasis", "variants": ["Subsegmental atelectasis", "Plate-like atelectasis", "Linear volume loss"] },
    { "concept": "Lobar collapse", "variants": ["Lobar atelectasis", "Complete lobar collapse", "Segmental collapse"] },
    { "concept": "Volume loss", "variants": ["Regional volume loss", "Reduced lung volume", "Parenchymal collapse"] },
    { "concept": "Hyperinflation", "variants": ["Hyperexpanded lungs", "Increased lung volumes", "Overinflated lungs"] },
    { "concept": "Air trapping", "variants": ["Regional air trapping", "Asymmetric lucency", "Persistent hyperlucent lung"] }
  ],

  "pleural_abnormalities": [
    { "concept": "Pleural effusion", "variants": ["Small pleural effusion", "Blunted costophrenic angle", "Dependent pleural fluid"] },
    { "concept": "Large pleural effusion", "variants": ["Moderate to large effusion", "Extensive pleural fluid", "Hemithorax opacification"] },
    { "concept": "Loculated effusion", "variants": ["Loculated pleural fluid", "Encapsulated effusion", "Non-layering effusion"] },
    { "concept": "Pleural thickening", "variants": ["Pleural-based thickening", "Focal pleural opacity", "Chronic pleural change"] },
    { "concept": "Pleural calcification", "variants": ["Calcified pleura", "Pleural plaques", "Pleural calcific densities"] },
    { "concept": "Pneumothorax", "variants": ["Apical pneumothorax", "Visible pleural line", "Air in pleural space"] },
    { "concept": "Hydropneumothorax", "variants": ["Air-fluid level in pleural space", "Combined air and fluid", "Hydropneumothorax"] }
  ],

  "airways": [
    { "concept": "Bronchial wall thickening", "variants": ["Peribronchial thickening", "Thickened bronchial walls", "Tram-track opacities"] },
    { "concept": "Air bronchogram", "variants": ["Visible air bronchograms", "Patent bronchi in consolidation", "Air-filled bronchi"] },
    { "concept": "Bronchiectasis", "variants": ["Dilated bronchi", "Cylindrical bronchiectasis", "Prominent bronchial markings"] },
    { "concept": "Mucus plugging", "variants": ["Endobronchial plugging", "Mucoid impaction", "Bronchial filling defects"] }
  ],

  "cardiomediastinal": [
    { "concept": "Cardiomegaly", "variants": ["Enlarged cardiac silhouette", "Increased cardiothoracic ratio", "Globular heart"] },
    { "concept": "Small cardiac silhouette", "variants": ["Small heart size", "Decreased cardiothoracic ratio", "Reduced cardiac silhouette"] },
    { "concept": "Mediastinal widening", "variants": ["Widened mediastinum", "Broad mediastinal contour", "Increased mediastinal width"] },
    { "concept": "Prominent pulmonary vasculature", "variants": ["Pulmonary vascular congestion", "Prominent hilar vessels", "Engorged pulmonary vasculature"] },
    { "concept": "Pulmonary edema pattern", "variants": ["Perihilar opacities", "Bat-wing distribution", "Vascular redistribution"] }
  ],

  "spatial_distribution": [
    { "concept": "Bilateral involvement", "variants": ["Bilateral lung opacities", "Both lungs involved", "Symmetric bilateral findings"] },
    { "concept": "Unilateral involvement", "variants": ["Unilateral opacity", "Single lung involvement", "Right or left sided abnormality"] },
    { "concept": "Upper lung zone predominance", "variants": ["Upper lobe predominance", "Apical distribution", "Upper zone involvement"] },
    { "concept": "Mid lung zone predominance", "variants": ["Mid lung involvement", "Perihilar-mid zone pattern", "Central lung predominance"] },
    { "concept": "Lower lung zone predominance", "variants": ["Basilar predominance", "Lower lobe involvement", "Dependent lung opacities"] },
    { "concept": "Peripheral distribution", "variants": ["Subpleural distribution", "Peripheral lung opacities", "Outer lung zone involvement"] }
  ],

  "osseous_and_soft_tissue": [
    { "concept": "Rib fracture", "variants": ["Acute rib fracture", "Cortical rib disruption", "Healing rib fracture"] },
    { "concept": "Clavicular fracture", "variants": ["Clavicle fracture", "Displaced clavicular injury", "Cortical clavicular irregularity"] },
    { "concept": "Vertebral compression fracture", "variants": ["Compression deformity", "Vertebral height loss", "Spinal compression fracture"] },
    { "concept": "Soft tissue emphysema", "variants": ["Subcutaneous emphysema", "Soft tissue gas", "Air in chest wall tissues"] }
  ],

  "devices_and_artifacts": [
    { "concept": "Endotracheal tube", "variants": ["Endotracheal tube in situ", "ET tube projecting over trachea", "Intubation tube present"] },
    { "concept": "Nasogastric tube", "variants": ["NG tube", "Enteric tube", "Feeding tube coursing below diaphragm"] },
    { "concept": "Central venous catheter", "variants": ["Central line", "Venous catheter tip in SVC", "Internal jugular catheter"] },
    { "concept": "Chest tube", "variants": ["Thoracostomy tube", "Pleural drainage catheter", "Chest tube in place"] },
    { "concept": "Pacemaker or ICD", "variants": ["Cardiac pacemaker", "Implantable cardiac device", "Pacemaker leads present"] }
  ]
}


class ConceptDataset(Dataset):
    def __init__(self, data_df, backbone_utils, image_size=(1024, 1024), aug_factor=0):
        self.__data_df = data_df
        self.__image_size = image_size
        self.__data_df['aug'] = False
        self.__backbone_utils = backbone_utils

        self.dataset = []

        for i in range(len(self.__data_df)):
            image_id = self.__data_df.loc[i, 'patientId']
            # img = backbone_utils.read_image(image_id)
            # self.get_transformation()(img)
            data = {
                'idx': i,
                'image_id': image_id,
                'label': self.__data_df.loc[i, 'Target'],
                # 'img_emb': backbone_utils.get_image_embedding(img),
                'aug': False
            }
            self.dataset.append(data)

        for i in range(aug_factor):
            offset = len(self.dataset)
            for j in range(len(self.__data_df)):
                image_id = self.__data_df.loc[j, 'patientId']
                # img = backbone_utils.read_image(image_id)
                # img = self.get_augmentations()(img)
                data = {
                    'idx': offset + j,
                    'image_id': image_id,
                    'label': self.__data_df.loc[j, 'Target'],
                    # 'img_emb': backbone_utils.get_image_embedding(img),
                    'aug': True
                }
                self.dataset.append(data)

        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def get_augmentations(self):
        augmentations = transforms.Compose([
            transforms.Resize(self.__image_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomEqualize(0.2),
            transforms.RandomPosterize(8, p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.GaussianBlur(kernel_size=3),
            # transforms.ToTensor(),
            transforms.RandomResizedCrop(self.__image_size, scale=(0.8, 1.0)),
            transforms.Grayscale(num_output_channels=1),
        ])

        return augmentations

    def get_transformation(self):
        return transforms.Compose([
            transforms.Resize(self.__image_size),
            transforms.Grayscale(num_output_channels=1)
        ])

    def __normalize_image(self, image):
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val + 1e-8)

        return image

    def __getitem__(self, idx):
        data = self.dataset[idx]
        assert (idx == data['idx'])

        img = self.__backbone_utils.read_image(data['image_id'])
        if data['aug']:
            img = self.get_augmentations()(img)
        else:
            img = self.get_transformation()(img)
        img_embedding = self.__backbone_utils.get_image_embedding(img)

        label = torch.tensor(data['label'], dtype=torch.int).to('cuda')

        # print(f'embedding: {img_embedding}, label: {label}')
        return img_embedding, label


class ConceptModel(pl.LightningModule):
    def __init__(self, concepts, threshold=0.5, lr=1e-5, wd=1e-4, awd=0.01, alpha=1, beta=1e-2, gamma=1,
                 temperature=0.7):
        super().__init__()
        self.__lr = lr
        self.__wd = wd
        self.__awd = awd
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.temperature = temperature
        self.concepts = concepts
        concept_embeddings = torch.stack([concept['embeddings'] for concept in self.concepts], dim=0)

        self.register_buffer(
            "concept_embeddings",
            F.normalize(concept_embeddings, dim=1)
        )

        # print(f'lr: {self.__lr}, wd: {self.__wd}, alpha: {self.__alpha}, beta: {self.__beta},  gamma: {self.__gamma}')
        print(f'concept_embeddings shape : {self.concept_embeddings.shape}')
        num_neurons, input_dim = self.concept_embeddings.shape

        # print(f'input_dim:: {input_dim}, hidden_dim_0:: {hidden_dim_0}, hidden_dim_1:: {hidden_dim_1}, num_neurons:: {num_neurons}')
        self.cb_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_neurons, bias=False)
        )
        # --- Task Head ---
        self.classifier = nn.Linear(num_neurons, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_metrics = {
            'train': torchmetrics.classification.BinaryAccuracy().to('cuda'),
            'val': torchmetrics.classification.BinaryAccuracy().to('cuda'),
            'test': torchmetrics.classification.BinaryAccuracy().to('cuda')
        }

    def hybrid_alignment_loss(self, z, projections, lambda_row=1.0, lambda_col=2.0, eps=1e-8):
        """
        z: [B, K] concept activations
        projections: [B, K] CLIP similarities
        """

        # -------------------------
        # Row-wise cosine^3
        # -------------------------
        z_row = F.normalize(z, dim=1)
        p_row = F.normalize(projections, dim=1)
        cos_row = (z_row * p_row).sum(dim=1)
        row_loss = torch.mean(1 - cos_row ** 3)

        # -------------------------
        # Column-wise cosine^3
        # -------------------------
        z_col = F.normalize(z, dim=0)
        p_col = F.normalize(projections, dim=0)
        cos_col = (z_col * p_col).sum(dim=0)
        col_loss = torch.mean(1 - cos_col ** 3)

        # -------------------------
        # Hybrid
        # -------------------------
        return lambda_row * row_loss + lambda_col * col_loss, col_loss

    def orthogonality_loss(self, z):
        z_norm = F.normalize(z, dim=0)
        gram = z_norm.T @ z_norm
        I = torch.eye(gram.size(0), device=z.device)
        return torch.norm(gram - I, p='fro') ** 2

    def forward(self, input_embeddings):
        input_embeddings = input_embeddings.squeeze(1)
        # print(f'input shape:: {x.shape}')

        cb_embedding = self.cb_projector(input_embeddings)
        logits = self.classifier(cb_embedding).squeeze(-1)

        return logits, cb_embedding

    def step(self, batch, stage):
        image_embeddings, labels = batch
        image_embeddings = image_embeddings.squeeze(1)
        image_embeddings = F.normalize(image_embeddings, dim=1)

        logits, cb_activations = self(image_embeddings)

        # ----- Task loss -----
        task_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # ----- Text similarity projections -----
        # image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(self.concept_embeddings, dim=1)

        projections = (image_embeddings @ text_embeddings.T) / self.temperature

        sim_loss, mean_sim = self.hybrid_alignment_loss(cb_activations, projections)

        # ----- Sparsity -----
        sparsity_loss = torch.mean(torch.abs(cb_activations))

        # ----- Orthogonality -----
        ortho_loss = self.orthogonality_loss(cb_activations)

        total_loss = (
                task_loss
                + self.__gamma * sim_loss
                + self.__alpha * sparsity_loss
                + self.__beta * ortho_loss
        )
        metrics = {
            'sim': mean_sim,
            'acc': self.acc_metrics[stage](logits, labels)
        }

        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_task_loss", task_loss)
        self.log(f"{stage}_sim_loss", sim_loss)
        self.log(f"{stage}_sparsity", sparsity_loss)
        self.log(f"{stage}_ortho", ortho_loss)

        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        stage = 'train'
        loss, metrics = self.step(batch, stage)

        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # print(f'metrics: {metrics}')
        for metric, value in metrics.items():
            self.log(f"{stage}_{metric}", value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        stage = 'val'
        loss, metrics = self.step(batch, stage)

        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for metric, value in metrics.items():
            self.log(f"{stage}_{metric}", value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        stage = 'test'
        loss, metrics = self.step(batch, stage)

        x, y = batch
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for metric, value in metrics.items():
            self.log(f"{stage}_{metric}", value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        input_embeddings = input_embeddings.squeeze(1)
        _, cb_activations = self(input_embeddings)

        return cb_activations

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.__lr, weight_decay=self.__wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]


class ConceptTrainer():

    def __init__(self, data_df, backbone_utils, concept_features, checkpoints_dir,
                 max_epoch=20, lr=1e-5, test_frac=0.2, batch_size=64, aug_factor=0, balance_class=True,
                 alpha=0.6, beta=0.5, gamma=1e-3):
        self.__max_epochs = max_epoch
        self.__lr = lr
        self.__beta = beta

        # self.model.apply(self.__init_weights)
        self.__test_frac = test_frac
        self.__batch_size = batch_size
        self.__checkpoints_dir = checkpoints_dir
        self.backbone_utils = backbone_utils
        self.concepts = [
            {
                'name': concept['name'],
                'texts': concept['texts'],
                'embeddings': torch.mean(backbone_utils.get_text_embedding(concept['texts']).cuda(), dim=0)
            } for concept in concept_features
        ]

        # self.concept_embeddings = backbone_utils.get_text_embedding([concept['texts'][0] for concept in self.concepts])
        self.concept_embeddings = torch.stack([concept['embeddings'] for concept in self.concepts], dim=0)
        # self.concept_embeddings = backbone_utils.get_text_embedding(concept_features)

        self.model = ConceptModel(self.concepts, alpha=alpha, beta=beta, gamma=gamma)
        # self.model.load_state_dict(torch.load(model_path))

        # if balance_class: #oversampling positive class
        #     positive = data_df[data_df['Target']==1]
        #     data_df = pd.concat([data_df, positive])
        #     data_df = data_df.reset_index(drop=True)

        train_df, test_df = train_test_split(data_df, test_size=test_frac, stratify=data_df['Target'])
        # train_df, val_df = train_test_split(train_df, test_size=test_frac, stratify=train_df['Target'])
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        # self.val_df = val_df.reset_index(drop=True)

        print('train distribution', self.train_df.groupby('Target').count())
        # print('val distribution',self.val_df.groupby('Target').count())
        print('test distribution', self.test_df.groupby('Target').count())
        self.train_ds = ConceptDataset(self.train_df, backbone_utils, aug_factor=aug_factor)
        # self.val_ds = CBEDataset(self.val_df, backbone_utils, aug_factor=aug_factor)
        self.test_ds = ConceptDataset(self.test_df, backbone_utils)

        print(f'num samples: train {len(self.train_ds)} | test {len(self.test_ds)}')

        self.train_dataloader = DataLoader(self.train_ds, batch_size=self.__batch_size, num_workers=0)
        self.test_dataloader = DataLoader(self.test_ds, batch_size=self.__batch_size, shuffle=False)
        # self.val_dataloader = DataLoader(self.val_ds, batch_size=self.__batch_size, num_workers=0)

        pl.seed_everything(42)

    def train(self, devices = (1,2,3,4,5,6,7), fast_dev_run=False):

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.__checkpoints_dir}/checkpoints/1",  # Directory to save checkpoints
            filename="model-{epoch:02d}-{val_loss:.2f}",  # Naming convention
            monitor="val_loss",  # Metric to monitor for saving best checkpoints
            mode="min",  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )

        early_stopping = EarlyStopping(monitor="val_loss",
                                       mode="min",
                                       patience=3,
                                       verbose=True,
                                       min_delta=0.001
                                       )

        trainer = pl.Trainer(accelerator='gpu',
                             devices=devices,
                             max_epochs=self.__max_epochs,
                             strategy='ddp',
                             callbacks=[
                                 checkpoint_callback,
                                 early_stopping
                             ],
                             enable_progress_bar=True,
                             log_every_n_steps=2,
                             fast_dev_run=fast_dev_run
                             )

        self.model.hparams.lr = self.__lr

        print(f"\n\n\n{'== ' * 20} Model training begins {'== ' * 20}")
        # batch = next(iter(self.train_dataloader))
        # inputs, labels = batch

        # print(f'train inputs:: {inputs}, labels::{labels}')
        #
        # batch = next(iter(self.val_dataloader))
        # inputs, labels = batch
        #
        # print(f'val inputs:: {inputs}, labels::{labels}')

        trainer.fit(self.model, self.train_dataloader, self.test_dataloader)
        print(f"\n\n\n{'== ' * 20} Model training finished {'== ' * 20}")
        trainer.strategy.barrier()
        train_metrics = trainer.logged_metrics
        print(f'train_metrics:: {train_metrics}')
        return train_metrics

    def test(self, devices = (1,2,3,4,5,6,7),fast_dev_run=False):

        trainer = pl.Trainer(accelerator='gpu',
                             devices=devices,
                             max_epochs=self.__max_epochs,
                             strategy='ddp',
                             enable_progress_bar=True,
                             log_every_n_steps=2,
                             num_nodes=1,
                             fast_dev_run=fast_dev_run
                             )
        print(f"\n\n\n{'== ' * 20} Model testing begins {'== ' * 20}")

        # batch = next(iter(self.test_dataloader))
        # inputs, labels = batch
        #
        # print(f'test inputs:: {inputs}, labels::{labels}')rad

        trainer.test(self.model, self.test_dataloader)
        print(f"\n\n\n{'== ' * 20} Model testing finished {'== ' * 20}")
        trainer.strategy.barrier()
        test_metrics = trainer.callback_metrics

        print(f'test_metrics:: {test_metrics}')

        return test_metrics

    def predict(self, df, devices = (1,2,3,4,5,6,7), fast_dev_run=False):
        ds = ConceptDataset(df, self.backbone_utils)
        dataloader = DataLoader(ds, batch_size=self.__batch_size, shuffle=False)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=[0],
                             max_epochs=self.__max_epochs,
                             strategy=devices,
                             enable_progress_bar=True,
                             log_every_n_steps=2,
                             num_nodes=1,
                             fast_dev_run=fast_dev_run
                             )
        print(f"\n\n\n{'== ' * 20} predicting {'== ' * 20}")
        output = trainer.predict(self.model, dataloader)
        return torch.cat(output, dim=0)



    def save_model(self, model_path):
        print(f'Saving the model at path:: {model_path}')
        torch.save(self.model.state_dict(), model_path)

def save_activations(data_df, output_activations, save_path):
    image_ids = data_df['patientId'].tolist()
    image_embeddings = {}
    imgs = []
    transformations = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.Grayscale(num_output_channels=1)
    ])

    loop = tqdm(image_ids, leave=False, total=len(image_ids))

    for image_id in loop:
        img = mlu.read_image(image_id)

        img = transformations(img)
        img_embedding = mlu.get_image_embedding(img)
        # print(img_embedding.device)
        imgs.append(img_embedding)
        image_embeddings[image_id] = img_embedding

    concepts = [
        {
            'name': concept['name'],
            'texts': concept['texts'],
            'embeddings': mlu.get_text_embedding(concept['texts'])
        } for concept in CONCEPTS
    ]

    imgs = torch.stack(imgs, dim=0).squeeze(1)
    concept_embeddings = torch.stack([concept['embeddings'] for concept in concepts], dim=0).squeeze(1)
    # projections = imgs @ concept_embeddings.T

    concept_texts = [c['texts'] for c in concepts]

    print(
        f'imgs: {imgs.shape}, concept_embeddings: {concept_embeddings.shape}, output_activations: {output_activations.shape}, concept_texts: {len(concept_texts)}')
    # op_path = 'cbmad_outputs/clip_activations_new.pt'
    torch.save({
        'image_ids': image_ids,
        'concept_texts': concept_texts,
        'activations': output_activations,
        'image_embeddings': image_embeddings,
        'concepts': concepts
    }, save_path)


if __name__=="__main__":
    BASE_PATH = '/mnt/surya/dataset/rsna_pneumonia/rsna_pnumonia_challenge/data/kaggle'
    train_csv_path = f'{BASE_PATH}/stage_2_train_labels.csv'
    train_data_dir = f'{BASE_PATH}/stage_2_train_images'

    checkpoints_dir = 'models/cbm_multigpu/1'
    model_path = 'models/cbm_models/concept_model_1.pt'
    save_path = 'cbmad_outputs/cbm_activations_multigpu.pt'

    CONCEPTS = []
    for x, a in radiology_glossary_2.items():
        for t in a:
            for v in t['variants']:
                concept = {
                    'name': t['concept'],
                    'texts': v
                }
                CONCEPTS.append(concept)


    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    mlu = MedCLIPModelUtils(train_data_dir)
    data_df = pd.read_csv(train_csv_path)

    final_trainer = ConceptTrainer(data_df, mlu, CONCEPTS, checkpoints_dir, lr=1e-6, alpha=0, beta=0, gamma=2)

    fast_dev_run = False
    devices = (1, 2, 3, 4, 5, 6, 7)
    final_trainer.train(devices, fast_dev_run)
    print(f'Saving model at {model_path}')
    final_trainer.save_model(model_path)

    # prediction
    output = final_trainer.predict(data_df, devices, fast_dev_run)


    save_activations(data_df, output, save_path)