import torch
import torch.nn as nn

class DivergenceScore(nn.Module):
    def __init__(self, forensics_dim=512, graph_dim=512, out_dim=128):
        super(DivergenceScore, self).__init__()
        # Learns a non-linear interaction/divergence metric between forensics and metadata
        self.interaction = nn.Sequential(
            nn.Linear(forensics_dim + graph_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, forensics_features, graph_features):
        """
        forensics_features: (Batch, 512)
        graph_features: (Batch, 512)
        """
        concat_features = torch.cat([forensics_features, graph_features], dim=-1)
        return self.interaction(concat_features)
