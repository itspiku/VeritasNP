import os
import torch
from src.data_processing.dataset import create_dataloader
from src.training.trainer import VeritasTrainer

def main():
    print("VeritasNP: Multimodal Fake News Detection")
    print("=========================================")
    
    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 2. Setup Data
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data set')
    
    print(f"Loading data from: {data_dir}")
    # Use a smaller batch size to avoid Out-Of-Memory on GPU due to heavy models
    batch_size = 16 
    dataloader = create_dataloader(data_dir, batch_size=batch_size, shuffle=True)
    
    print(f"Total batches per epoch: {len(dataloader)}")
    
    # 3. Setup Trainer
    trainer = VeritasTrainer(device=device, lr=2e-5)
    
    # 4. Training Loop
    num_epochs = 3
    print(f"\nStarting training for {num_epochs} epochs...")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        avg_loss, accuracy = trainer.train_epoch(dataloader)
        
        print(f"Epoch {epoch} Complete! Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/veritas_model_epoch_{epoch}.pth"
        torch.save({
            'text_encoder': trainer.text_enc.state_dict(),
            'image_encoder': trainer.img_enc.state_dict(),
            'forensics_encoder': trainer.ela_enc.state_dict(),
            'modality_alignment': trainer.align.state_dict(),
            'divergence_score': trainer.div.state_dict(),
            'final_classifier': trainer.classifier.state_dict(),
        }, checkpoint_path)
        
        print(f"Model weights saved to {checkpoint_path}")
        
    print("\nTraining Complete! 🚀")

if __name__ == "__main__":
    main()
