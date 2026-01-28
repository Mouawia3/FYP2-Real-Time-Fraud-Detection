import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

print("🧪 FYP2 MODEL VALIDATION - PROVE F1 ≈ 99.6% IS REAL")
print("=" * 70)

# ---------------------------------------------------
# 1. Load model (DistilBERT + LoRA PEFT adapter)
# ---------------------------------------------------
print("🔄 Loading LoRA adapter model from 'models/qlora_fraud_real_final'...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)
model = PeftModel.from_pretrained(base_model, "models/qlora_fraud_real_final")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model.eval()


# ---------------------------------------------------
# 2. Load validation data
# ---------------------------------------------------
print("\n📊 Loading validation data...")
test_df = pd.read_csv("data/real/val.csv")
print(f"Validation set: {len(test_df):,} samples ({test_df['isFraud'].sum():,} fraud)")

texts = test_df["description"]
true_labels = test_df["isFraud"].values.astype(int)


# ---------------------------------------------------
# 3. Prediction helper
# ---------------------------------------------------
def predict_batch(texts_series: pd.Series, batch_size: int = 256):
    """
    Predict labels and fraud probabilities for texts in batches to avoid OOM.
    Returns:
      preds: np.ndarray[int]  (0 = legit, 1 = fraud)
      probs: np.ndarray[float] (fraud probability)
    """
    all_preds = []
    all_probs = []

    texts_list = texts_series.tolist()
    n = len(texts_list)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts_list[start:end]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)

    return np.concatenate(all_preds), np.concatenate(all_probs)


# ---------------------------------------------------
# TEST 1: Full validation set
# ---------------------------------------------------
def evaluate_dataset(csv_path: str, label: str):
    df_eval = pd.read_csv(csv_path)
    texts_eval = df_eval["description"]
    y_true = df_eval["isFraud"].values.astype(int)

    preds_eval, probs_eval = predict_batch(texts_eval)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds_eval, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, preds_eval)

    cm = confusion_matrix(y_true, preds_eval, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fn_rate = fn / (fn + tp) * 100.0 if (fn + tp) > 0 else 0.0
    tp_rate = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0

    cls_report = classification_report(
        y_true,
        preds_eval,
        target_names=["Legit", "Fraud"],
        digits=4,
        zero_division=0,
    )

    print(f"\n✅ [{label}] size={len(df_eval):,} fraud={int(df_eval['isFraud'].sum()):,}")
    print(f"   F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    print(f"   CM: tn={tn} fp={fp} fn={fn} tp={tp} | FN%={fn_rate:.2f} | Recall%={tp_rate:.2f}")

    return {
        "label": label,
        "dataset": {
            "path": csv_path,
            "size": int(len(df_eval)),
            "n_fraud": int(df_eval["isFraud"].sum()),
            "n_legit": int(len(df_eval) - df_eval["isFraud"].sum()),
            "fraud_rate": float(df_eval["isFraud"].mean()),
        },
        "metrics": {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "false_negative_rate_percent": float(fn_rate),
            "fraud_detection_rate_percent": float(tp_rate),
        },
        "classification_report_text": cls_report,
    }


# ---------------------------------------------------
# EVALUATE: Balanced val, Imbalanced test, OOD test
# ---------------------------------------------------
print("\n🔍 EVALUATION A: BALANCED VALIDATION SET (data/real/val.csv)")
res_val = evaluate_dataset("data/real/val.csv", "BALANCED_VAL")

print("\n🔍 EVALUATION B: IMBALANCED REAL-WORLD TEST (data/real/test_imbalanced.csv)")
res_imb = evaluate_dataset("data/real/test_imbalanced.csv", "IMBALANCED_TEST")

print("\n🔍 EVALUATION C: OOD SHIFT TEST (data/real/test_ood.csv)")
res_ood = evaluate_dataset("data/real/test_ood.csv", "OOD_TEST")


# ---------------------------------------------------
# Manual stress tests
# ---------------------------------------------------
print("\n🔍 STRESS TESTS (manual)")

stress_tests = {
    "REAL FRAUD": "CASH_OUT 38427.47 from C1399554611 to C988696172 step:207 oldOrg:38427 newOrg:0",
    "REAL LEGIT": "PAYMENT 9839.64 from C1231006815 to M1979787155 step:1 oldOrg:170136 newOrg:160296",
    "HIGH AMOUNT": "TRANSFER 1000000.00 from C9999999999 to C8888888888 step:500 oldOrg:0 newOrg:0",
    "SUSPICIOUS": "CASH_OUT 999999.99 from C1234567890 to C9876543210 step:999 oldOrg:999999 newOrg:0",
}

stress_results = {}
for name, text in stress_tests.items():
    single_text_series = pd.Series([text])
    s_pred, s_prob = predict_batch(single_text_series)
    label = "FRAUD" if int(s_pred[0]) == 1 else "LEGIT"
    prob_val = float(s_prob[0])
    stress_results[name] = {"label": label, "fraud_prob": prob_val}
    print(f"   {name:12}: {label:5} (prob: {prob_val:.3f})")

print("\n🏆 ALL EVALUATIONS COMPLETE")

# ---------------------------------------------------
# 5. Export metrics for report (JSON)
# ---------------------------------------------------
Path("results").mkdir(parents=True, exist_ok=True)

metrics_payload = {
    "evaluations": {
        "balanced_val": res_val,
        "imbalanced_test": res_imb,
        "ood_test": res_ood,
    },
    "stress_tests": stress_results,
}


with open("results/eval_all.json", "w") as f:
    json.dump(metrics_payload, f, indent=2)

print("\n📝 All evaluation metrics saved to results/eval_all.json")