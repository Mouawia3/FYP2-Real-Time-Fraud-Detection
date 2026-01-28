import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("🚀 FYP2 - PREPARE REAL 6.3M PAYSIM DATASET")
print("=" * 60)

# Load FULL dataset (6.3M rows)
print("📥 Loading 6.3M transactions...")
df = pd.read_csv('data/transactions.csv')

print(f"✅ LOADED: {len(df):,} rows")
print(f"   Fraud: {df['isFraud'].value_counts()}")
print(f"   Fraud rate: {df['isFraud'].mean():.4%}")

# Sample 20K rows (10K train + 10K val) with stratification
print("\n⚖️  STRATIFIED SAMPLING (20K total)...")
df_sample = df.groupby('isFraud', group_keys=False).apply(lambda x: x.sample(min(len(x), 10000), random_state=42))
print(f"   Sampled: {len(df_sample):,} rows")
print(f"   Fraud: {df_sample['isFraud'].value_counts()}")

# Create text description
df_sample['description'] = df_sample.apply(lambda row:
    f"{row['type']} {row['amount']:.2f} from {row['nameOrig']} to {row['nameDest']} "
    f"step:{row['step']} oldOrg:{row['oldbalanceOrg']:.0f} newOrg:{row['newbalanceOrig']:.0f}", axis=1)

# Split
train_df, val_df = train_test_split(df_sample, test_size=0.5, stratify=df_sample['isFraud'], random_state=42)

print(f"\n✅ FINAL DATASETS:")
print(f"   TRAIN: {len(train_df):,} ({train_df['isFraud'].sum():,} fraud)")
print(f"   VAL:   {len(val_df):,} ({val_df['isFraud'].sum():,} fraud)")

# Save
os.makedirs('data/real', exist_ok=True)
train_df[['description', 'isFraud']].to_csv('data/real/train.csv', index=False)
val_df[['description', 'isFraud']].to_csv('data/real/val.csv', index=False)

print(f"\n💾 SAVED: data/real/train.csv & data/real/val.csv")
print("✅ READY FOR QLoRA TRAINING!")
print("📝 Sample fraud text:")
print(train_df[train_df['isFraud']==1]['description'].head(2).values)
