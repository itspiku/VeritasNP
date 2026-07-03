import torch
import torch.nn as nn

class ModalityAlignment(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, aligned_dim=512):
        super(ModalityAlignment, self).__init__()
        # Project both to a shared dimension
        self.text_proj = nn.Linear(text_dim, aligned_dim)
        self.image_proj = nn.Linear(image_dim, aligned_dim)
        
        # Fuse them
        self.fusion = nn.Sequential(
            nn.Linear(aligned_dim * 2, aligned_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, text_features, image_features):
        """
        text_features: (Batch, text_dim)
        image_features: (Batch, image_dim)
        """
        t_proj = torch.relu(self.text_proj(text_features))
        i_proj = torch.relu(self.image_proj(image_features))
        
        # Concatenate and fuse
        fused = torch.cat([t_proj, i_proj], dim=-1)
        return self.fusion(fused)
