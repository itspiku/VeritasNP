import torch
import torch.nn as nn

class FinalClassifier(nn.Module):
    def __init__(self, aligned_dim=512, divergence_dim=128, num_classes=1):
        super(FinalClassifier, self).__init__()
        # Binary Classification (Real = 0, Fake = 1)
        in_dim = aligned_dim + divergence_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, aligned_features, divergence_features):
        """
        aligned_features: (Batch, 512)
        divergence_features: (Batch, 128)
        Returns: Logits (Batch, 1)
        """
        final_rep = torch.cat([aligned_features, divergence_features], dim=-1)
        return self.classifier(final_rep)
