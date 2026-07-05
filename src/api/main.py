import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_processing.text_preprocessing import TextPreprocessor
from src.encoders.text_encoder import NepaliTextEncoder
from src.encoders.image_encoder import VisualEncoder
from src.encoders.forensics_encoder import ELAEncoder
from src.fusion.modality_alignment import ModalityAlignment
from src.fusion.divergence_score import DivergenceScore
from src.classification.final_classifier import FinalClassifier

app = FastAPI(title="VeritasNP - Fake News Detection API", version="1.0")

class NewsArticle(BaseModel):
    text: str
    source_type: str = "unknown"
    category: str = "unknown"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# Global model state
models = {}
preprocessor = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.on_event("startup")
async def startup_event():
    global models, preprocessor
    print(f"Loading VeritasNP models on {device}...")
    
    preprocessor = TextPreprocessor()
    
    # Initialize architecture
    models['text_enc'] = NepaliTextEncoder(freeze_base=True).to(device)
    models['img_enc'] = VisualEncoder().to(device)
    models['ela_enc'] = ELAEncoder().to(device)
    models['align'] = ModalityAlignment().to(device)
    models['div'] = DivergenceScore().to(device)
    models['classifier'] = FinalClassifier().to(device)
    
    # Put all models in evaluation mode
    for model in models.values():
        model.eval()
        
    # Attempt to load trained weights if they exist
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints')
    # Use the most recent epoch (we assume epoch 3 is the best for now, or fallback)
    for epoch in [3, 2, 1]:
        ckpt_path = os.path.join(checkpoint_dir, f'veritas_model_epoch_{epoch}.pth')
        if os.path.exists(ckpt_path):
            print(f"Found trained weights: {ckpt_path}. Loading...")
            try:
                # Need weights_only=False temporarily for MuRIL compatibility based on the CVE rules for older versions, or just load them if valid. 
                # Since we fixed the CVE by updating PyTorch to >= 2.6, weights_only=True is safe.
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                models['text_enc'].load_state_dict(ckpt['text_encoder'])
                models['img_enc'].load_state_dict(ckpt['image_encoder'])
                models['ela_enc'].load_state_dict(ckpt['forensics_encoder'])
                models['align'].load_state_dict(ckpt['modality_alignment'])
                models['div'].load_state_dict(ckpt['divergence_score'])
                models['classifier'].load_state_dict(ckpt['final_classifier'])
                print("Weights loaded successfully!")
                break
            except Exception as e:
                print(f"Could not load weights, using untrained models. Error: {e}")
                break
    else:
        print("No trained weights found in checkpoints/. Using untrained architecture.")

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(article: NewsArticle):
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="News text cannot be empty.")
        
    try:
        with torch.no_grad():
            # 1. Process Text
            tokens = preprocessor.tokenize([article.text])
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            text_features = models['text_enc'](input_ids, attention_mask)
            
            # 2. Dummy Image & Graph features
            dummy_images = torch.zeros((1, 3, 224, 224), device=device)
            dummy_ela = torch.zeros((1, 3, 224, 224), device=device)
            graph_features = torch.zeros((1, 512), device=device)
            
            image_features = models['img_enc'](dummy_images)
            ela_features = models['ela_enc'](dummy_ela)
            
            # 3. Fusion
            aligned_features = models['align'](text_features, image_features)
            divergence_features = models['div'](ela_features, graph_features)
            
            # 4. Classification
            logits = models['classifier'](aligned_features, divergence_features).squeeze()
            
            # Apply sigmoid to get probability between 0 and 1
            # 1 = Fake, 0 = Real
            probability = torch.sigmoid(logits).item()
            
            # If the probability is a single float, we can process it directly
            prediction_label = "FAKE" if probability > 0.5 else "REAL"
            
            # Confidence is how far it is from 0.5
            confidence = (probability if prediction_label == "FAKE" else 1.0 - probability) * 100.0
            
            return PredictionResponse(
                prediction=prediction_label,
                confidence=round(confidence, 2)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
