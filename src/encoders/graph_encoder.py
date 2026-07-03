import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    print("WARNING: torch_geometric not installed. GraphSage will not work until you install it.")
    SAGEConv = None

class MetadataGraphSage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MetadataGraphSage, self).__init__()
        if SAGEConv is None:
            raise ImportError("Please install PyTorch Geometric (torch_geometric) to use MetadataGraphSage.")
            
        # 2-layer GraphSAGE
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        x: Node feature matrix (NumNodes, in_channels)
        edge_index: Graph connectivity (2, NumEdges)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
