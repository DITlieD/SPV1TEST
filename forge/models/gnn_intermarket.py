import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT_Intermarket(nn.Module):
    """
    A Graph Attention Network (GAT) to learn relationships between assets.
    It takes the graph of assets and outputs enriched feature embeddings for each asset.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT_Intermarket, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # On the last layer, we average the outputs of the attention heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Dropout is applied within the GATConv layers
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        
        return x