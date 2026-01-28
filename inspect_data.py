import pandas as pd
import os

print("🔍 FYP2 FULL DATA INSPECTION (ROBUST)")
print("=" * 60)

# 1. ORIGINAL DATA - COLUMNS FIRST
print("\n📁 1. ORIGINAL DATA (data/transactions.csv)")
if os.path.exists('data/transactions.csv'):
    orig_df = pd.read_csv('data/transactions.csv', nrows=5)  # Just 5 rows to check structure
    print(f"   Shape: {len(pd.read_csv('data/transactions.csv')):,} rows x {len(orig_df.columns)} cols")
    print(f"   ALL COLUMNS: {list(orig_df.columns)}")
    print(f"   Sample rows (first 3):")
    print(orig_df.to_string(index=False))

    # Check for fraud-related columns
    fraud_cols = [col for col in orig_df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
    print(f"   Fraud-related cols: {fraud_cols}")
else:
    print("   ❌ File not found!")

# 2. Check all data folders
print("\n📁 2. PROCESSED DATA FOLDERS")
folders = ['data/clean', 'data/balanced', 'data/augmented']
for folder in folders:
    if os.path.exists(folder):
        print(f"\n   📂 {folder}:")
        for file in os.listdir(folder)[:5]:  # First 5 files
            filepath = os.path.join(folder, file)
            if os.path.isfile(filepath) and filepath.endswith('.csv'):
                try:
                    df_sample = pd.read_csv(filepath, nrows=3)
                    print(f"      {file}: {len(pd.read_csv(filepath)):,} rows")
                    print(f"         Columns: {list(df_sample.columns)}")
                    print(
                        f"         Fraud count: {df_sample.get('labels', pd.Series()).sum() if 'labels' in df_sample.columns else 'No labels'}")
                except:
                    print(f"      {file}: ERROR reading")

print("\n📁 3. FOLDER SUMMARY")
for root, dirs, files in os.walk('data'):
    if files:
        print(f"   {root}: {len(files)} files ({sum(os.path.getsize(os.path.join(root, f)) for f in files):,} bytes)")

print("\n✅ STRUCTURE ANALYSIS COMPLETE")
