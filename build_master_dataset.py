#build_master_dataset
import os
import pandas as pd
import numpy as np

# =====================================
# PATHS
# =====================================

IMG_ROOT = r"G:\My Drive\MIMIC-CXR-JPG\selected_images"

META_PATH = r"G:\My Drive\MIMIC-CXR-JPG\mimic-cxr-2.0.0-metadata.csv.gz"

CHEXPERT_PATH = r"G:\My Drive\MIMIC-CXR-JPG\mimic-cxr-2.0.0-chexpert.csv.gz"

ICU_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\icu\icustays.csv.gz"

PATIENTS_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\hosp\patients.csv.gz"

ADMISSIONS_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\hosp\admissions.csv.gz"

DIAGNOSES_PATH = r"G:\My Drive\MIMIC-IV\physionet.org\files\mimiciv\3.1\hosp\diagnoses_icd.csv.gz"

OUT_DIR = r"D:\mimic_project\dataset"

os.makedirs(OUT_DIR, exist_ok=True)

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

# =====================================
# STEP 1 — SCAN IMAGES
# =====================================

print("\nScanning images...")

image_paths = []

for root, _, files in os.walk(IMG_ROOT):
    for f in files:
        if f.endswith(".jpg"):
            image_paths.append(os.path.join(root, f))

print("Images found:", len(image_paths))

# =====================================
# STEP 2 — LOAD METADATA
# =====================================

print("\nLoading metadata...")

meta = pd.read_csv(META_PATH)

meta = meta[["dicom_id","subject_id","study_id"]]

# =====================================
# STEP 3 — BUILD IMAGE DATASET
# =====================================

image_df = pd.DataFrame({
    "image_path": image_paths,
    "dicom_id":[os.path.basename(p).replace(".jpg","") for p in image_paths]
})

image_df = image_df.merge(meta,on="dicom_id",how="inner")

# Load patient demographics
print("Loading patient demographics and admission features...")
patients = pd.read_csv(PATIENTS_PATH)
patients = patients[["subject_id","gender","anchor_age"]]

# Encode gender (M=1, F=0)
patients["gender_encoded"] = (patients["gender"] == "M").astype(int)
patients = patients[["subject_id","gender_encoded","anchor_age"]]
patients.columns = ["subject_id","gender","age"]

# Load admissions data (only for mortality flag)
admissions = pd.read_csv(ADMISSIONS_PATH)

# Select relevant admission features
admissions_features = admissions[[
    "subject_id", "hadm_id",
    "hospital_expire_flag"
]].copy()

admissions_features.columns = [
    "subject_id", "hadm_id",
    "mortality_flag"
]

# Load diagnoses and create condition flags
print("Loading diagnoses data...")
diagnoses = pd.read_csv(DIAGNOSES_PATH)
icd_map = pd.read_csv('G:\\My Drive\\MIMIC-IV\\physionet.org\\files\\mimiciv\\3.1\\hosp\\d_icd_diagnoses.csv.gz')

# Create a mapping of ICD codes to conditions
def get_condition_codes(icd_map, pattern):
    """Extract ICD codes matching a condition pattern"""
    matches = icd_map[icd_map['long_title'].str.contains(pattern, case=False, regex=True)]
    return set(matches['icd_code'].astype(str).values)

# Get codes for each condition
diabetes_codes = get_condition_codes(icd_map, r'diabetes|diabetes mellitus')
hypertension_codes = get_condition_codes(icd_map, r'hypertension|hypertensive')
copd_codes = get_condition_codes(icd_map, r'chronic obstructive pulmonary|COPD|emphysema')
heart_disease_codes = get_condition_codes(icd_map, r'heart failure|cardiac|coronary|myocardial infarction|angina|ischemic')
kidney_disease_codes = get_condition_codes(icd_map, r'chronic kidney|renal|end stage renal|ESRD')

print(f"Found {len(diabetes_codes)} diabetes codes")
print(f"Found {len(hypertension_codes)} hypertension codes")
print(f"Found {len(copd_codes)} COPD codes")
print(f"Found {len(heart_disease_codes)} heart disease codes")
print(f"Found {len(kidney_disease_codes)} kidney disease codes")

# Convert codes to strings in diagnoses
diagnoses['icd_code_str'] = diagnoses['icd_code'].astype(str)

# Create condition flags for each admission
condition_flags = diagnoses.groupby(['subject_id', 'hadm_id']).apply(
    lambda x: pd.Series({
        'diabetes': int(any(x['icd_code_str'].isin(diabetes_codes))),
        'hypertension': int(any(x['icd_code_str'].isin(hypertension_codes))),
        'copd': int(any(x['icd_code_str'].isin(copd_codes))),
        'heart_disease': int(any(x['icd_code_str'].isin(heart_disease_codes))),
        'kidney_disease': int(any(x['icd_code_str'].isin(kidney_disease_codes)))
    })
).reset_index()

# Merge with admission features
admissions_features = admissions_features.merge(condition_flags, on=['subject_id', 'hadm_id'], how='left')

# Fill NaN with 0 (no diagnosis found)
admissions_features['diabetes'] = admissions_features['diabetes'].fillna(0)
admissions_features['hypertension'] = admissions_features['hypertension'].fillna(0)
admissions_features['copd'] = admissions_features['copd'].fillna(0)
admissions_features['heart_disease'] = admissions_features['heart_disease'].fillna(0)
admissions_features['kidney_disease'] = admissions_features['kidney_disease'].fillna(0)

# Merge with demographics
image_df = image_df.merge(patients, on="subject_id", how="left")
image_df = image_df.merge(admissions_features, on="subject_id", how="left")

print("Images with metadata:",len(image_df))
print("Patients:",image_df.subject_id.nunique())

# =====================================
# STEP 4 — LOAD LABELS
# =====================================

print("\nLoading CheXpert labels...")

chexpert = pd.read_csv(CHEXPERT_PATH)

chexpert = chexpert[["subject_id","study_id"]+TARGET_LABELS]

chexpert[TARGET_LABELS] = (
    chexpert[TARGET_LABELS]
    .replace(-1,0)
    .fillna(0)
    .astype(int)
)

# =====================================
# STEP 5 — MERGE LABELS
# =====================================

image_df = image_df.merge(
    chexpert,
    on=["subject_id","study_id"],
    how="left"
)

image_df[TARGET_LABELS] = image_df[TARGET_LABELS].fillna(0)

print("Labeled images:",len(image_df))

# =====================================
# STEP 6 — FILTER ICU PATIENTS
# =====================================

print("\nFiltering ICU patients...")

icu = pd.read_csv(ICU_PATH,usecols=["subject_id"])

icu_patients = set(icu.subject_id.unique())

image_df = image_df[
    image_df.subject_id.isin(icu_patients)
]

print("Patients after ICU filter:",image_df.subject_id.nunique())

# =====================================
# STEP 7 — PATIENT LEVEL DATASET
# =====================================

patient_labels = (
    image_df
    .groupby("subject_id")[TARGET_LABELS]
    .max()
    .reset_index()
)

patient_images = (
    image_df
    .groupby("subject_id")["image_path"]
    .apply(list)
    .reset_index()
)

# Aggregate static features to patient level
patient_static = (
    image_df
    .groupby("subject_id")[[
        "gender", "age",
        "diabetes", "hypertension", "copd", "heart_disease", "kidney_disease"
    ]]
    .first()
    .reset_index()
)

patient_df = patient_labels.merge(
    patient_images,
    on="subject_id"
)

patient_df = patient_df.merge(
    patient_static,
    on="subject_id",
    how="left"
)

print("\nPatients before balancing:",len(patient_df))

# =====================================
# STEP 8 — BALANCE DATASET
# =====================================

print("\nBalancing dataset...")

positive = patient_df[
    patient_df[TARGET_LABELS].sum(axis=1) > 0
]

negative = patient_df[
    patient_df[TARGET_LABELS].sum(axis=1) == 0
]

negative_sample = negative.sample(
    n=min(len(negative), len(positive)),
    random_state=42
)

balanced_df = pd.concat([
    positive,
    negative_sample
])

balanced_df = balanced_df.sample(frac=1,random_state=42)

print("Final balanced patients:",len(balanced_df))

# =====================================
# STEP 9 — SAVE DATASET
# =====================================

balanced_df.to_csv(
    os.path.join(OUT_DIR,"patients.csv"),
    index=False
)

image_df.to_csv(
    os.path.join(OUT_DIR,"images.csv"),
    index=False
)

np.save(
    os.path.join(OUT_DIR,"subject_ids.npy"),
    balanced_df.subject_id.values
)

print("\nDataset saved")
print("Patients:",len(balanced_df))