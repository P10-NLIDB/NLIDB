from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from collections import Counter

class BERTClassifierFromEmbeddings(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.pooling = global_mean_pool
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.pooling(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()  # raw logits, no sigmoid



def train_BERTclassifier_early_stop(model, train_loader, val_loader, num_epochs=30, lr=1e-3, weight_decay=1e-2, device=None, early_stop_patience=5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # --- Compute class imbalance for pos_weight
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch.y.view(-1).tolist())
    label_counts = Counter(all_labels)
    pos = label_counts.get(1.0, 1)
    neg = label_counts.get(0.0, 1)
    pos_weight_val = neg / pos
    print(f"Label dist: {label_counts} | pos_weight: {pos_weight_val:.2f}")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_model_state = None
    patience = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds = model(batch)  # [batch_size]
            labels = batch.y.float().to(device)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f}")

        acc, precision, recall, f1 = evaluate(model, val_loader, device)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{early_stop_patience}")
            if patience >= early_stop_patience:
                print("Early stopping.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"Best F1 on val set: {best_val_f1:.4f}")

    return model


def train_BERTclassifier(model, train_loader, val_loader, num_epochs=30, lr=1e-3, weight_decay=1e-2, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Compute pos_weight for class imbalance
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch.y.view(-1).tolist())
    label_counts = Counter(all_labels)
    pos = label_counts.get(1.0, 1)
    neg = label_counts.get(0.0, 1)
    pos_weight_val = neg / pos
    print(f"Label dist: {label_counts} | pos_weight: {pos_weight_val:.2f}")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)  # raw logits
            labels = batch.y.float().to(device)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f}")

        # Optional: track validation metrics
        acc, precision, recall, f1 = evaluate(model, val_loader, device)

    return model

@torch.no_grad()
def evaluate(model, loader, device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().cpu()
        labels = batch.y.long().cpu()

        all_preds.extend(preds.view(-1).tolist())
        all_labels.extend(labels.view(-1).tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Eval | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
    return acc, precision, recall, f1