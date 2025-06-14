# models/gnn_classifier.py

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.nn import GATConv, global_mean_pool


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(dropout)

        self.heads = heads
        self.concat = False

        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=heads,
                    concat=self.concat, dropout=dropout)
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim, heads=heads,
                        concat=self.concat, dropout=dropout)
            )

        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        out = self.classifier(x)
        return out.view(-1)  # Return raw logits


def train(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Increase weight for positive class to improve recall
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_agg = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

        acc, prec, rec, f1 = evaluate(model, val_loader, threshold=0.3)
        agg = (acc + prec + rec + f1)/4

        if agg > best_agg:
            best_agg = agg
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Best model was from epoch {best_epoch} with aggregate score {best_agg:.4f}")
    model.load_state_dict(best_model_state)
    return model



def evaluate(model, loader, threshold=0.3):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long().cpu()
            labels = batch.y.long().cpu()

            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1)
    recall = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)

    print(f"Evaluation | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return acc, precision, recall, f1


def get_prediction_results(model, loader, original_entries):
    """
    Runs the model on the loader and returns the entries it predicted as True or False.

    Args:
        model: Trained GNN model.
        loader: DataLoader with batched graphs
        original_entries: List of original dataset entries aligned with the graphs.

        
    Returns:
        Tuple[List[dict], List[dict]]: (predicted_true_entries, predicted_false_entries)
    """
    model.eval()
    device = next(model.parameters()).device
    true_entries = []
    false_entries = []

    index = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = (model(batch) > 0.5).long().cpu().toList()
            batch_size = len(preds)

            for i in range(batch_size):
                entry = original_entries[index]
                if preds[i] == 1:
                    true_entries.append(entry)
                else:
                    false_entries.append(entry)
                index += 1

    return true_entries, false_entries
