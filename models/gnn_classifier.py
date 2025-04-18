# models/gnn_classifier.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNClassifier(torch.nn.Module):
    """
    A GNN (GCN) for binary graph classification.

    Args:
        in_dim (int): Input feature dimension for each node. For now this is 3 from one-hot: [question, table, column]
        hidden_dim (int): Dimension of hidden layers. Default is 64.
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return torch.sigmoid(out).squeeze()


def train(model, loader, epochs=10, lr=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
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
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = (model(batch) > 0.5).long()
            labels = batch.y.long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Accuracy: {acc:.4f}")
    return acc
