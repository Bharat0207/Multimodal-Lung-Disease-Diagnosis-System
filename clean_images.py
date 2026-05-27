#clean_images.py
import pandas as pd
import numpy as np

# =========================
# PATHS
# =========================

PATIENT_PATH = r"D:\mimic_project\dataset\patients.csv"
IMAGE_PATH = r"D:\mimic_project\dataset\images.csv"

OUT_PATH = r"D:\mimic_project\dataset\clean_images.csv"

MAX_IMAGES_PER_PATIENT = 5


# =========================
# LOAD DATA
# =========================

print("Loading patients...")
patients = pd.read_csv(PATIENT_PATH)

print("Patients:", len(patients))

print("Loading images...")
images = pd.read_csv(IMAGE_PATH)

print("Images:", len(images))


# =========================
# FILTER PATIENTS
# =========================

patient_ids = set(patients["subject_id"])

images = images[images["subject_id"].isin(patient_ids)]

print("Images after patient filter:", len(images))


# =========================
# REMOVE DUPLICATE STUDIES
# =========================

print("Removing duplicate studies...")

images = images.sort_values("study_id")

images = images.drop_duplicates(
    subset=["subject_id", "study_id"]
)

print("Images after removing duplicates:", len(images))


# =========================
# LIMIT IMAGES PER PATIENT
# =========================

print("Limiting images per patient...")

images = (
    images
    .groupby("subject_id")
    .head(MAX_IMAGES_PER_PATIENT)
    .reset_index(drop=True)
)

print("Images after limit:", len(images))


# =========================
# SAVE
# =========================

images.to_csv(OUT_PATH, index=False)

print("\nClean image dataset saved")

print("Final images:", len(images))
print("Patients:", images["subject_id"].nunique())