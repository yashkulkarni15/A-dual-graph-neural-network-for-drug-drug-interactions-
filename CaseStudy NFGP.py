import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DualGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(DualGNN, self).__init__()
        self.conv_molecular = GCNConv(num_features, hidden_dim)
        self.conv_interaction = GCNConv(num_features, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_molecular, edge_index_molecular, x_interaction, edge_index_interaction):
        # Process molecular structure graph
        x_molecular = F.relu(self.conv_molecular(x_molecular, edge_index_molecular))
        x_molecular = F.dropout(x_molecular, training=self.training)
        x_molecular = torch.mean(x_molecular, dim=0)  # Aggregate node features

        # Process drug-drug interaction graph
        x_interaction = F.relu(self.conv_interaction(x_interaction, edge_index_interaction))
        x_interaction = F.dropout(x_interaction, training=self.training)
        x_interaction = torch.mean(x_interaction, dim=0)  # Aggregate node features

        # Concatenate features from both graphs
        x_combined = torch.cat([x_molecular, x_interaction], dim=-1)

        # Fully connected layers
        x_combined = F.relu(self.fc1(x_combined))
        x_combined = F.dropout(x_combined, training=self.training)
        out = self.fc2(x_combined)
        return torch.sigmoid(out)

# Example usage
num_nodes = 10  # Number of nodes in each graph
num_features = 64  # Number of features in the input data
hidden_dim = 32  # Dimensionality of hidden layers
num_classes = 1  # Binary classification (interaction or no interaction)

# Create model instance
model = DualGNN(num_features, hidden_dim, num_classes)

# Example data (molecular structure and interaction graphs)
x_molecular = torch.randn(num_nodes, num_features)
edge_index_molecular = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x_interaction = torch.randn(num_nodes, num_features)
edge_index_interaction = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

# Forward pass
output = model(x_molecular, edge_index_molecular, x_interaction, edge_index_interaction)
print("Output probability:", output.item())
