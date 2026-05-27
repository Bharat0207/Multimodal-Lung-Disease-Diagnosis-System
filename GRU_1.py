import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# DATA PATH
# =========================

DATA_DIR = r"D:\mimic_project\dataset"

# =========================
# LOAD DATA
# =========================

X_vitals = np.load(DATA_DIR + r"\vitals_timeseries.npy")
X_labs = np.load(DATA_DIR + r"\labs_timeseries.npy")

X = np.concatenate([X_vitals, X_labs], axis=2)

print("Input shape:", X.shape)

SUBJECT_IDS = np.load(DATA_DIR + r"\subject_ids.npy")

labels_df = pd.read_csv(DATA_DIR + r"\patients.csv")

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

label_map = labels_df.set_index("subject_id")

Y = []
for sid in SUBJECT_IDS:
    Y.append(label_map.loc[sid, TARGET_LABELS].values)

Y = np.array(Y, dtype=np.float32)

print("Labels:", Y.shape)

# =========================
# TRAIN / VAL SPLIT
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=(Y.sum(axis=1) > 0)
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# =========================
# DATASET
# =========================

class GRUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# CLASS WEIGHTS
# =========================

pos_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - pos_counts

pos_weight = torch.tensor(
    neg_counts / (pos_counts + 1e-6),
    dtype=torch.float32
).to(device)

print("Class weights:", pos_weight)

# =========================
# MODEL
# =========================

class GRUNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        hidden = 128

        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.norm = nn.LayerNorm(hidden * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 4)
        )

    def forward(self, x):

        out, _ = self.gru(x)

        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        x = torch.cat([avg_pool, max_pool], dim=1)

        return self.fc(x)

    # =========================
    # FEATURE EXTRACTOR FOR FUSION
    # =========================
    def extract_features(self, x):

        out, _ = self.gru(x)

        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        x = torch.cat([avg_pool, max_pool], dim=1)

        return x   # 512-dim feature


# =========================
# DATALOADER
# =========================

train_loader = DataLoader(
    GRUDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    GRUDataset(X_val, y_val),
    batch_size=64
)

# =========================
# INIT MODEL
# =========================

model = GRUNet(X.shape[2]).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=5,
    factor=0.5
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# =========================
# TRAIN
# =========================

best_auc = 0
patience = 10
counter = 0

EPOCHS = 80

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    # =========================
    # VALIDATION
    # =========================

    model.eval()
    preds = []

    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb))
            preds.append(probs.cpu().numpy())

    preds = np.vstack(preds)

    macro_auc = np.mean([
        roc_auc_score(y_val[:, i], preds[:, i])
        for i in range(4)
    ])

    scheduler.step(macro_auc)

    print(f"Epoch {epoch+1} Loss {total_loss/len(train_loader):.4f} AUROC {macro_auc:.4f}")

    if macro_auc > best_auc:
        best_auc = macro_auc
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break

# =========================
# LOAD BEST MODEL
# =========================

model.load_state_dict(best_model)
model.eval()

# =========================
# EVALUATION
# =========================

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

all_probs = []
all_labels = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(yb.numpy())

y_pred = np.vstack(all_probs)
y_true = np.vstack(all_labels)

# Binary predictions
y_pred_bin = (y_pred >= 0.5).astype(int)

print("\n====== FINAL RESULTS ======")

label_aurocs = []
label_auprcs = []

for i, label in enumerate(TARGET_LABELS):

    y_t = y_true[:, i]
    y_p = y_pred[:, i]
    y_b = y_pred_bin[:, i]

    acc = accuracy_score(y_t, y_b)
    prec = precision_score(y_t, y_b, zero_division=0)
    rec = recall_score(y_t, y_b, zero_division=0)
    f1 = f1_score(y_t, y_b, zero_division=0)

    auc = roc_auc_score(y_t, y_p)
    auprc = average_precision_score(y_t, y_p)

    cm = confusion_matrix(y_t, y_b)

    label_aurocs.append(auc)
    label_auprcs.append(auprc)

    print(f"\n===== {label} =====")
    print("Accuracy:", round(acc,4))
    print("Precision:", round(prec,4))
    print("Recall:", round(rec,4))
    print("F1:", round(f1,4))
    print("AUROC:", round(auc,4))
    print("AUPRC:", round(auprc,4))
    print("Confusion Matrix:")
    print(cm)

# =========================
# MACRO METRICS
# =========================

print("\nMacro AUROC:", round(np.mean(label_aurocs), 4))
print("Macro AUPRC:", round(np.mean(label_auprcs), 4))

# =========================
# SAVE FULL DATASET PREDICTIONS (IMPORTANT FOR FUSION)
# =========================

print("\nSaving full dataset predictions for fusion...")

full_dataset = GRUDataset(X, Y)

full_loader = DataLoader(
    full_dataset,
    batch_size=64,
    shuffle=False   # VERY IMPORTANT
)

all_preds = []

with torch.no_grad():
    for xb, _ in full_loader:
        xb = xb.to(device)
        probs = torch.sigmoid(model(xb))
        all_preds.append(probs.cpu().numpy())

all_preds = np.vstack(all_preds)

print("Saved GRU preds shape:", all_preds.shape)

np.save("gru_preds.npy", all_preds)