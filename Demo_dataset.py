import numpy as np
import pandas as pd
import shutil
import os

# =========================
# LOAD FULL DATA
# =========================

timeseries = np.load("final_timeseries.npy")
tabular = np.load("final_tabular.npy")
image_paths = np.load("final_image_paths.npy", allow_pickle=True)
labels = np.load("final_labels.npy")

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

# =========================
# SELECT 100 PATIENTS
# =========================

N = 100
indices = np.random.choice(len(timeseries), N, replace=False)

# =========================
# CREATE FOLDER
# =========================

OUT_DIR = "demo_dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

# =========================
# PREPARE CSV DATA
# =========================

rows = []

for i, idx in enumerate(indices):

    src_img = image_paths[idx]

    if not os.path.exists(src_img):
        continue

    dst_img = os.path.join(IMG_DIR, f"{i}.jpg")
    shutil.copy(src_img, dst_img)

    row = {
        "patient_id": i,
        "original_idx": int(idx), 
        "image_path": dst_img
    }

    # add labels
    for j, label in enumerate(TARGET_LABELS):
        row[label] = labels[idx][j]

    # add tabular features
    for k in range(tabular.shape[1]):
        row[f"feat_{k}"] = tabular[idx][k]

    rows.append(row)

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "demo.csv"), index=False)

# =========================
# SAVE TIMESERIES
# =========================

np.save(os.path.join(OUT_DIR, "demo_timeseries.npy"), timeseries[indices])

print("\nDemo dataset created!")
print("CSV:", df.shape)
print("Timeseries:", timeseries[indices].shape)