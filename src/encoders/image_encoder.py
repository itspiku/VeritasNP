import torch
import torch.nn as nn
from torchvision import models

class VisualEncoder(nn.Module):
    def __init__(self, output_size=768):
        super(VisualEncoder, self).__init__()
        # Use a lightweight EfficientNet (B0 has 1280 output channels)
        effnet = models.efficientnet_b0(pretrained=True)
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(effnet.children())[:-1])
        
        self.projection = nn.Linear(1280, output_size)
        
    def forward(self, images):
        """
        images: (Batch, Channels, Height, Width)
        Returns: (Batch, output_size) aligned with text dimension
        """
        x = self.feature_extractor(images)
        x = x.view(x.size(0), -1) # Flatten (B, 1280, 1, 1) -> (B, 1280)
        return self.projection(x)
