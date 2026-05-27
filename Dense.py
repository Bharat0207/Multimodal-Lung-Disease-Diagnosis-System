import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# PATHS
# =========================

DATA_PATH = r"D:\mimic_project\dataset\clean_images.csv"
IMG_ROOT = r"D:\mimic_project\images"

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

# =========================
# DATASET
# =========================

class CXRDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        try:
            img = Image.open(row["full_path"]).convert("RGB")
        except:
            idx = np.random.randint(0, len(self.df))
            return self.__getitem__(idx)

        img = self.transform(img)

        label = torch.tensor(
            row[TARGET_LABELS].values.astype(np.float32)
        )

        return img, label

# =========================
# MAIN
# =========================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(DATA_PATH)

    df["full_path"] = df["image_path"].apply(
        lambda x: os.path.join(IMG_ROOT, os.path.basename(x))
    )

    df = df[df["full_path"].apply(os.path.exists)]

    print("Images found:", len(df))

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    print("Train:", len(train_df), "Val:", len(val_df))

    # =========================
    # TRANSFORMS
    # =========================

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    train_loader = DataLoader(
        CXRDataset(train_df, train_transform),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        CXRDataset(val_df, val_transform),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================

    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    model = model.to(device)

    torch.backends.cudnn.benchmark = True

    # freeze
    for p in model.features.parameters():
        p.requires_grad = False

    # =========================
    # LOSS
    # =========================

    pos_counts = train_df[TARGET_LABELS].sum().values
    neg_counts = len(train_df) - pos_counts

    pos_weight = torch.tensor(
        neg_counts/(pos_counts+1e-6),
        dtype=torch.float32
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_auc = 0

    # =========================
    # TRAIN
    # =========================

    for epoch in range(20):

        if epoch == 5:
            print("Unfreezing backbone...")
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        model.train()
        total_loss = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1} Loss {total_loss/len(train_loader):.4f}")

        # =========================
        # VALIDATION
        # =========================

        model.eval()
        preds, labels_true = [], []

        with torch.no_grad():
            for imgs, targets in val_loader:

                imgs = imgs.to(device)

                with torch.cuda.amp.autocast():
                    probs = torch.sigmoid(model(imgs))

                preds.append(probs.cpu().numpy())
                labels_true.append(targets.numpy())

        preds = np.vstack(preds)
        labels_true = np.vstack(labels_true)

        aucs = [
            roc_auc_score(labels_true[:,i], preds[:,i])
            for i in range(4)
        ]

        macro_auc = np.mean(aucs)

        print("Validation AUROC:", [round(a,3) for a in aucs])
        print("Macro AUROC:", round(macro_auc,3))

        if macro_auc > best_auc:
            best_auc = macro_auc
            torch.save(model.state_dict(), "best_densenet.pth")
            print("Saved best model!")

    # =========================
    # FULL DATA INFERENCE (ONCE ONLY)
    # =========================

    print("\nGenerating DenseNet predictions...")

    full_loader = DataLoader(
        CXRDataset(df, val_transform),
        batch_size=64, 
        shuffle=False,
        num_workers=2
    )

    all_preds = []

    model.eval()

    with torch.no_grad():
        for imgs, _ in tqdm(full_loader):

            imgs = imgs.to(device)

            with torch.cuda.amp.autocast():
                probs = torch.sigmoid(model(imgs))

            all_preds.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)

    print("Image preds:", all_preds.shape)

    # =========================
    # PATIENT LEVEL
    # =========================

    df_preds = df.copy().reset_index(drop=True)

    df_preds["pred_0"] = all_preds[:,0]
    df_preds["pred_1"] = all_preds[:,1]
    df_preds["pred_2"] = all_preds[:,2]
    df_preds["pred_3"] = all_preds[:,3]

    if "subject_id" not in df_preds.columns:
        raise ValueError("❌ subject_id missing in CSV")

    patient_preds = df_preds.groupby("subject_id")[
        ["pred_0","pred_1","pred_2","pred_3"]
    ].mean()

    subject_ids = np.load(r"D:\mimic_project\dataset\subject_ids.npy")

    patient_preds = patient_preds.loc[subject_ids]

    dense_preds = patient_preds.values

    print("Final Dense preds:", dense_preds.shape)

    np.save("dense_patient_preds.npy", dense_preds)

    print("DONE — ready for fusion")


if __name__ == "__main__":
    main()