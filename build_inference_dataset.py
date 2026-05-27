import numpy as np
import pandas as pd
import os

DATA_DIR = r"D:\mimic_project\dataset"
IMG_ROOT = r"D:\mimic_project\images"

# =========================
# LOAD DATA
# =========================

subject_ids = np.load(DATA_DIR + r"\subject_ids.npy")

vitals = np.load(DATA_DIR + r"\vitals_timeseries.npy")   # (N,24,5)
labs   = np.load(DATA_DIR + r"\labs_timeseries.npy")     # (N,24,3)

df = pd.read_csv(DATA_DIR + r"\patients.csv")
df = df.set_index("subject_id")
df = df.loc[subject_ids]

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

# =========================
# COMBINE TIMESERIES
# =========================

timeseries = np.concatenate([vitals, labs], axis=2)  # (N,24,8)

# =========================
# CREATE TABULAR FEATURES
# =========================

def extract_features(ts):
    return np.concatenate([
        ts.mean(axis=0),
        ts.std(axis=0),
        ts.min(axis=0),
        ts.max(axis=0)
    ])  # 8*4 = 32

tabular = np.array([extract_features(ts) for ts in timeseries])

print("Tabular shape:", tabular.shape)

# =========================
# IMAGE PATH (IMPORTANT)
# =========================

# You must have mapping already from clean_images.csv
img_df = pd.read_csv(DATA_DIR + r"\clean_images.csv")

img_df["subject_id"] = img_df["subject_id"].astype(int)
img_df = img_df.groupby("subject_id").first()

image_paths = []

for sid in subject_ids:
    if sid in img_df.index:
        image_paths.append(img_df.loc[sid]["image_path"])
    else:
        image_paths.append("")

image_paths = np.array(image_paths)

# =========================
# LABELS
# =========================

labels = df[TARGET_LABELS].values

# =========================
# SAVE FINAL DATASET
# =========================

np.save("final_timeseries.npy", timeseries)
np.save("final_tabular.npy", tabular)
np.save("final_labels.npy", labels)
np.save("final_image_paths.npy", image_paths)

print("\nFINAL DATASET CREATED")
print("Timeseries:", timeseries.shape)
print("Tabular:", tabular.shape)
print("Labels:", labels.shape)
print("Images:", image_paths.shape)