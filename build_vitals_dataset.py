import pandas as pd
import numpy as np

# =========================
# PATHS (KEEP YOUR DRIVE PATHS)
# =========================

ICU_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\icu\icustays.csv.gz"
CHARTEVENTS_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\icu\chartevents.csv.gz"

SUBJECT_PATH = r"D:\mimic_project\dataset\subject_ids.npy"
OUT_PATH = r"D:\mimic_project\dataset\vitals_timeseries.npy"

# =========================
# LOAD SUBJECT IDS
# =========================

subjects = np.load(SUBJECT_PATH)
print("Total subjects:", len(subjects))

subject_index = {sid: i for i, sid in enumerate(subjects)}

# =========================
# VITAL ITEM IDS
# =========================

VITAL_ITEMIDS = {
    211:0,        # HR
    615:1,        # RR
    678:2,        # TEMP
    220050:3,     # SBP
    220051:4      # DBP
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

# FAST MAP instead of merge
icu_map = {
    (row.subject_id, row.hadm_id): row.intime
    for _, row in icu.iterrows()
}

# =========================
# INIT TENSOR
# =========================

tensor = np.zeros((len(subjects), 24, 5), dtype=np.float32)

# =========================
# STREAM CHARTEVENTS
# =========================

print("Streaming chartevents...")

chunksize = 200_000
chunk_id = 0

for chunk in pd.read_csv(
    CHARTEVENTS_PATH,
    usecols=["subject_id","hadm_id","itemid","charttime","valuenum"],
    chunksize=chunksize
):

    chunk_id += 1
    print(f"Processing chunk {chunk_id}...")

    # keep only required vitals
    chunk = chunk[chunk["itemid"].isin(VITAL_ITEMIDS)]
    if len(chunk) == 0:
        continue

    # keep only our subjects
    chunk = chunk.loc[chunk["subject_id"].map(subject_index).notna()]
    if len(chunk) == 0:
        continue

    chunk = chunk.dropna(subset=["valuenum"])

    chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

    # FAST intime mapping
    chunk["intime"] = [
        icu_map.get((sid, hadm), None)
        for sid, hadm in zip(chunk["subject_id"], chunk["hadm_id"])
    ]

    chunk = chunk.dropna(subset=["intime"])

    # hour calculation
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
        j = VITAL_ITEMIDS[item]
        tensor[i, hr, j] = val

# =========================
# NORMALIZE
# =========================

print("Normalizing vitals...")

tensor = np.nan_to_num(tensor)
mean = tensor.mean()
std = tensor.std() + 1e-6
tensor = (tensor - mean) / std

np.save(OUT_PATH, tensor)

print("Vitals saved:", tensor.shape)