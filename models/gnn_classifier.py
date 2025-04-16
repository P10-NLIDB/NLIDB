# models/gnn_classifier.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNClassifier(torch.nn.Module):
    """
    A simple 2-layer Graph Convolutional Network (GCN) for binary graph classification.

    Args:
        in_dim (int): Input feature dimension for each node. For now this is 3 from one-hot: [question, table, column]
        hidden_dim (int): Dimension of hidden layers. Default is 64.
    """
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Aggregate node embeddings to graph-level
        out = self.classifier(x)
        return torch.sigmoid(out).squeeze()


def train(model, loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(loader):.4f}")


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            preds = (model(batch) > 0.5).long()
            labels = batch.y.long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Accuracy: {acc:.4f}")
