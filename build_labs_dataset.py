import pandas as pd
import numpy as np

# =========================
# PATHS
# =========================

LAB_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\hosp\labevents.csv.gz"
ICU_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\icu\icustays.csv.gz"

SUBJECT_PATH = r"D:\mimic_project\dataset\subject_ids.npy"
OUT_PATH = r"D:\mimic_project\dataset\labs_timeseries.npy"

# =========================
# LOAD SUBJECT IDS
# =========================

subjects = np.load(SUBJECT_PATH)
print("Total subjects:", len(subjects))

subject_index = {sid: i for i, sid in enumerate(subjects)}

# =========================
# LAB ITEM IDS
# =========================

LAB_ITEMIDS = {
    51300:0,  # WBC
    51256:1,  # Neutrophils
    50889:2   # CRP
}

# =========================
# LOAD ICU STAYS
# =========================

print("Loading ICU stays...")

icu = pd.read_csv(
    ICU_PATH,
    usecols=["subject_id","hadm_id","intime"]
)

icu["intime"] = pd.to_datetime(icu["intime"])

icu_map = {
    (row.subject_id, row.hadm_id): row.intime
    for _, row in icu.iterrows()
}

# =========================
# INIT TENSOR
# =========================

tensor = np.zeros((len(subjects), 24, 3), dtype=np.float32)

# =========================
# STREAM LABEVENTS
# =========================

print("Streaming lab events...")

chunksize = 200_000
chunk_id = 0

for chunk in pd.read_csv(
    LAB_PATH,
    usecols=["subject_id","hadm_id","itemid","charttime","valuenum"],
    chunksize=chunksize
):

    chunk_id += 1
    print(f"Processing chunk {chunk_id}...")

    chunk = chunk[chunk["itemid"].isin(LAB_ITEMIDS)]
    if len(chunk) == 0:
        continue

    chunk = chunk.loc[chunk["subject_id"].map(subject_index).notna()]
    if len(chunk) == 0:
        continue

    chunk = chunk.dropna(subset=["valuenum"])

    chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

    chunk["intime"] = [
        icu_map.get((sid, hadm), None)
        for sid, hadm in zip(chunk["subject_id"], chunk["hadm_id"])
    ]

    chunk["intime"] = pd.to_datetime(chunk["intime"], errors="coerce")

    chunk = chunk.dropna(subset=["intime"])

    chunk["hour"] = (
        (chunk["charttime"] - chunk["intime"])
        .dt.total_seconds() // 3600
    )

    chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] < 24)]
    if len(chunk) == 0:
        continue

    for sid, hr, item, val in zip(
        chunk["subject_id"].values,
        chunk["hour"].astype(int).values,
        chunk["itemid"].values,
        chunk["valuenum"].values
    ):
        i = subject_index[sid]
        j = LAB_ITEMIDS[item]
        tensor[i, hr, j] = val

# =========================
# NORMALIZE
# =========================

print("Normalizing labs...")

tensor = np.nan_to_num(tensor)
mean = tensor.mean()
std = tensor.std() + 1e-6
tensor = (tensor - mean) / std

np.save(OUT_PATH, tensor)

print("Labs saved:", tensor.shape)