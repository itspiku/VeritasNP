import torch
import torch.nn as nn
from torchvision import models

class ELAEncoder(nn.Module):
    def __init__(self, output_size=512):
        super(ELAEncoder, self).__init__()
        # Use ResNet-50 for forensics (ELA Map) analysis
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50 has 2048 out channels
        self.projection = nn.Linear(2048, output_size)
        
    def forward(self, ela_images):
        """
        ela_images: (Batch, 3, 224, 224) - Error Level Analysis maps
        """
        x = self.feature_extractor(ela_images)
        x = x.view(x.size(0), -1)
        return self.projection(x)
