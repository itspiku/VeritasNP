import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data_processing.text_preprocessing import TextPreprocessor
from src.encoders.text_encoder import NepaliTextEncoder
from src.encoders.image_encoder import VisualEncoder
from src.encoders.forensics_encoder import ELAEncoder
from src.fusion.modality_alignment import ModalityAlignment
from src.fusion.divergence_score import DivergenceScore
from src.classification.final_classifier import FinalClassifier

class VeritasTrainer:
    def __init__(self, device='cpu', lr=2e-5):
        self.device = torch.device(device)
        print(f"Initializing VeritasTrainer on {self.device}...")
        
        # 1. Preprocessors
        self.text_preprocessor = TextPreprocessor()
        
        # 2. Encoders
        self.text_enc = NepaliTextEncoder(freeze_base=True).to(self.device)
        self.img_enc = VisualEncoder().to(self.device)
        self.ela_enc = ELAEncoder().to(self.device)
        
        # 3. Fusion & Classification
        self.align = ModalityAlignment().to(self.device)
        self.div = DivergenceScore().to(self.device)
        self.classifier = FinalClassifier().to(self.device)
        
        # 4. Optimizer & Loss
        all_params = (
            list(self.text_enc.parameters()) + 
            list(self.img_enc.parameters()) + 
            list(self.ela_enc.parameters()) +
            list(self.align.parameters()) + 
            list(self.div.parameters()) + 
            list(self.classifier.parameters())
        )
        
        self.optimizer = optim.AdamW(all_params, lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_epoch(self, dataloader):
        self.text_enc.train()
        self.img_enc.train()
        self.ela_enc.train()
        self.align.train()
        self.div.train()
        self.classifier.train()
        
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch in tqdm(dataloader, desc="Training Epoch"):
            texts = batch['text']
            labels = batch['label'].float().to(self.device)
            batch_size = len(texts)
            
            # --- Modality 1: Text ---
            tokens = self.text_preprocessor.tokenize(texts)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            text_features = self.text_enc(input_ids, attention_mask)
            
            # --- Modality 2 & 3: Images & Forensics (DUMMY placeholders) ---
            dummy_images = torch.zeros((batch_size, 3, 224, 224), device=self.device)
            dummy_ela = torch.zeros((batch_size, 3, 224, 224), device=self.device)
            
            image_features = self.img_enc(dummy_images)
            ela_features = self.ela_enc(dummy_ela)
            
            # --- Modality 4: Graph (DUMMY placeholder) ---
            graph_features = torch.zeros((batch_size, 512), device=self.device)
            
            # --- FUSION ---
            aligned_features = self.align(text_features, image_features)
            divergence_features = self.div(ela_features, graph_features)
            
            # --- CLASSIFICATION ---
            logits = self.classifier(aligned_features, divergence_features).squeeze()
            
            # --- LOSS & BACKWARD ---
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Compute accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += batch_size
            
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct_preds / total_preds
        return avg_loss, accuracy

if __name__ == "__main__":
    print("Trainer script loaded successfully! Ready for training loop.")
