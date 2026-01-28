import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType


print("🚀 FYP2 LoRA TRAINING - REAL PAYSIM DATA (BALANCED)")
print("=" * 70)

# -------------------------------------------------------------------
# 1. Load balanced real-world dataset
# -------------------------------------------------------------------
train_df = pd.read_csv("data/real/train.csv")
val_df = pd.read_csv("data/real/val.csv")

print(f"✅ Train: {len(train_df):,} rows ({train_df['isFraud'].sum():,} fraud)")
print(f"   Val:   {len(val_df):,} rows ({val_df['isFraud'].sum():,} fraud)")

# HuggingFace Trainer expects "labels" as the target column
train_df = train_df.rename(columns={"isFraud": "labels", "description": "text"})
val_df = val_df.rename(columns={"isFraud": "labels", "description": "text"})


train_df = train_df[["text", "labels"]]
val_df = val_df[["text", "labels"]]

# -------------------------------------------------------------------
# 2. Tokenizer & datasets
# -------------------------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DistilBERT has no EOS token; safely set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

print("🔄 Tokenizing datasets...")
train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
print("✅ Tokenization complete")

# -------------------------------------------------------------------
# 3. Base model + LoRA PEFT adapter
# -------------------------------------------------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# If we added a PAD token, resize embeddings
if tokenizer.pad_token_id >= base_model.config.vocab_size:
    base_model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# -------------------------------------------------------------------
# 4. Metrics
# -------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# -------------------------------------------------------------------
# 5. Training configuration
# -------------------------------------------------------------------
os.makedirs("models/real_lora", exist_ok=True)

training_args = TrainingArguments(
    output_dir="models/real_lora",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=False,
    save_total_limit=2,
    report_to=None,                     # no WandB / HF logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# -------------------------------------------------------------------
# 6. Train
# -------------------------------------------------------------------
print("🎯 TRAINING STARTING NOW...")
train_result = trainer.train()

# Save PEFT adapter for inference (used by dashboard & 5-agent system)
adapter_output_dir = "models/qlora_fraud_real_final"  # kept name for compatibility
trainer.save_model(adapter_output_dir)
print(f"✅ MODEL ADAPTER SAVED at: {adapter_output_dir}")

# -------------------------------------------------------------------
# 7. Final evaluation + export metrics
# -------------------------------------------------------------------
eval_results = trainer.evaluate()
print("\n📊 FINAL VALIDATION RESULTS (balanced val set):")
for k, v in eval_results.items():
    print(f"   {k}: {v}")

print(f"\n🎉 FYP2 TARGET F1 (val): {eval_results.get('eval_f1', 0):.4f}")

# Export metrics to JSON for report usage
Path("results").mkdir(parents=True, exist_ok=True)

summary = {
    "train_size": len(train_df),
    "val_size": len(val_df),
    "model_name": model_name,
    "peft_type": "LoRA",
    "training_args": {
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "warmup_steps": training_args.warmup_steps,
    },
    "eval_metrics": {
        "accuracy": eval_results.get("eval_accuracy"),
        "f1": eval_results.get("eval_f1"),
        "precision": eval_results.get("eval_precision"),
        "recall": eval_results.get("eval_recall"),
    },
}

with open("results/train_val_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n📝 Metrics saved to results/train_val_metrics.json")
print("✅ TRAINING PIPELINE COMPLETE.")
