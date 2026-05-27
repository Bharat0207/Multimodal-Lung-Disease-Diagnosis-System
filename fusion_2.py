import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# =========================
# LOAD PREDICTIONS
# =========================

gru   = np.load("gru_preds.npy")
mlp   = np.load("mlp_preds.npy")
dense = np.load("dense_patient_preds.npy")

print("Shapes:", gru.shape, mlp.shape, dense.shape)

# Weight DenseNet
dense = dense * 2.5

# =========================
# FEATURE STACK
# =========================

X = np.concatenate([gru, mlp, dense], axis=1)
print("Fusion input:", X.shape)

# =========================
# LOAD LABELS
# =========================

DATA_DIR = r"D:\mimic_project\dataset"

subject_ids = np.load(DATA_DIR + r"\subject_ids.npy")

df = pd.read_csv(DATA_DIR + r"\patients.csv")
df = df.set_index("subject_id")

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

df = df.loc[subject_ids]
Y = df[TARGET_LABELS].values.astype(np.float32)

# =========================
# SPLIT (TRAIN + VAL + TEST)
# =========================

# First split: train (70%) + temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=(Y.sum(axis=1) > 0)
)

# Second split: val (15%) + test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=(y_temp.sum(axis=1) > 0)
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# =========================
# TRAIN XGBOOST
# =========================

models = []
val_preds = []
test_preds = []

for i, label in enumerate(TARGET_LABELS):

    print(f"\nTraining XGBoost for {label}...")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist"
    )

    model.fit(X_train, y_train[:, i])

    # Validation predictions
    val_p = model.predict_proba(X_val)[:, 1]
    test_p = model.predict_proba(X_test)[:, 1]

    models.append(model)
    val_preds.append(val_p)
    test_preds.append(test_p)

val_preds = np.vstack(val_preds).T
test_preds = np.vstack(test_preds).T

# =========================
# VALIDATION METRICS
# =========================

print("\n====== VALIDATION ======")

val_aucs = []

for i, label in enumerate(TARGET_LABELS):
    auc = roc_auc_score(y_val[:, i], val_preds[:, i])
    val_aucs.append(auc)
    print(label, "AUROC:", round(auc, 4))

print("Macro AUROC:", round(np.mean(val_aucs), 4))

# =========================
# THRESHOLD TUNING (ON VAL)
# =========================

thresholds = []

for i in range(4):
    best_t, best_f1 = 0.5, 0

    for t in np.linspace(0.3, 0.8, 100):
        pred_bin = (val_preds[:, i] >= t).astype(int)
        f1 = f1_score(y_val[:, i], pred_bin, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    thresholds.append(best_t)

print("Best thresholds:", thresholds)

# =========================
# FINAL TEST RESULTS
# =========================

print("\n====== FINAL TEST RESULTS ======")

test_aucs, test_auprcs = [], []

for i, label in enumerate(TARGET_LABELS):

    probs = test_preds[:, i]
    preds_bin = (probs >= thresholds[i]).astype(int)

    acc = accuracy_score(y_test[:, i], preds_bin)
    prec = precision_score(y_test[:, i], preds_bin, zero_division=0)
    rec = recall_score(y_test[:, i], preds_bin, zero_division=0)
    f1 = f1_score(y_test[:, i], preds_bin, zero_division=0)

    auc = roc_auc_score(y_test[:, i], probs)
    auprc = average_precision_score(y_test[:, i], probs)

    test_aucs.append(auc)
    test_auprcs.append(auprc)

    print(f"\n===== {label} =====")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1:", round(f1, 4))
    print("AUROC:", round(auc, 4))
    print("AUPRC:", round(auprc, 4))

print("\nMacro AUROC:", round(np.mean(test_aucs), 4))
print("Macro AUPRC:", round(np.mean(test_auprcs), 4))

# =========================
# SAVE MODELS + THRESHOLDS
# =========================

print("\nSaving fusion models (v2)...")

for i, label in enumerate(TARGET_LABELS):
    joblib.dump(models[i], f"fusion_{label}_1.pkl")

np.save("fusion_thresholds_1.npy", np.array(thresholds))

print("✅ Models and thresholds saved!")