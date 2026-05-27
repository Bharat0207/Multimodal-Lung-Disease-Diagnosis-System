import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# MODEL
# =========================

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# DATASET
# =========================

class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# MAIN
# =========================

def main():

    # =========================
    # LOAD STATIC DATA
    # =========================

    df = pd.read_csv("C:/mimic_project/dataset/patients.csv")

    # STATIC FEATURES
    FEATURE_COLS = [
        "anchor_age",
        "gender",
        "diabetes",
        "hypertension",
        "copd",
        "heart_disease",
    ]

    TARGET_LABELS = [
        "Pneumonia",
        "Pleural Effusion",
        "Edema",
        "Consolidation"
    ]

    # =========================
    # PREPROCESSING
    # =========================

    # Convert gender to numeric if needed
    if df["gender"].dtype == "object":
        df["gender"] = df["gender"].map({"M": 0, "F": 1})

    X = df[FEATURE_COLS].values.astype(np.float32)
    Y = df[TARGET_LABELS].values.astype(np.float32)

    print("Feature shape:", X.shape)

    # =========================
    # TRAIN / VAL SPLIT
    # =========================

    idx = np.arange(len(X))

    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=Y[:,0]
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]

    # =========================
    # NORMALIZATION
    # =========================

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    train_loader = DataLoader(DS(X_train,y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(DS(X_val,y_val), batch_size=32)

    # =========================
    # MODEL
    # =========================

    model = MLP(X.shape[1]).to(device)

    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts

    pos_weight = torch.tensor(
        neg_counts/(pos_counts+1e-6),
        dtype=torch.float32
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = 0
    best_preds = None

    # =========================
    # TRAINING
    # =========================

    for epoch in range(30):

        model.train()
        total_loss = 0

        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # =========================
        # VALIDATION
        # =========================

        model.eval()
        preds = []

        with torch.no_grad():
            for xb,_ in val_loader:
                xb = xb.to(device)
                preds.append(torch.sigmoid(model(xb)).cpu().numpy())

        preds = np.vstack(preds)

        aucs = [roc_auc_score(y_val[:,i], preds[:,i]) for i in range(4)]
        macro_auc = np.mean(aucs)

        print(f"\nEpoch {epoch+1} Loss {total_loss/len(train_loader):.4f}")
        print("Macro AUROC:", round(macro_auc,3))

        if macro_auc > best_auc:
            best_auc = macro_auc
            best_preds = preds.copy()
            torch.save(model.state_dict(),"mlp_static.pth")

    print("\nBEST AUROC:", best_auc)

    # =========================
    # SAVE PREDICTIONS (FOR FUSION)
    # =========================

    model.eval()

    X_all = scaler.transform(X)

    X_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)

    with torch.no_grad():
        full_preds = torch.sigmoid(model(X_tensor)).cpu().numpy()

    np.save("mlp_static_preds.npy", full_preds)

    print("Saved static predictions!")

if __name__ == "__main__":
    main()