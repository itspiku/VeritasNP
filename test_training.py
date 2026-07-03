import os
import torch
from src.training.trainer import VeritasTrainer

def test_trainer():
    print("Testing Trainer Initialization...")
    trainer = VeritasTrainer(device='cpu')
    
    print("\nCreating Dummy Dataloader...")
    # Mocking what the DataLoader would return
    dummy_batch = {
        'text': [
            "यो एउटा नक्कली समाचार हो जसले मानिसहरूलाई भ्रमित पार्न सक्छ।", 
            "नेपाल राष्ट्र बैंकले नयाँ नीति जारी गरेको छ।"
        ],
        'label': torch.tensor([1, 0])
    }
    
    # We create a dummy dataloader (just a list with one batch)
    dummy_dataloader = [dummy_batch]
    
    print("\nRunning a Single Training Step...")
    loss, accuracy = trainer.train_epoch(dummy_dataloader)
    
    print(f"\nTraining step complete! Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")
    print("Backpropagation successful!")

if __name__ == "__main__":
    test_trainer()
