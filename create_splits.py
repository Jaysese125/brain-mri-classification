from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# --- Step 1: Gather every image into a table ---
records = []
for split in ["Training", "Testing"]:
    for cls in CLASSES:
        cls_dir = DATA_DIR / split / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({"path": str(img_path), "label": cls})

df = pd.DataFrame(records)
print("Total images found:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# --- Step 2: Stratified 70/15/15 split ---
# First split off 30% as "temp" (which will become val + test)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)
# Split temp in half -> 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# --- Step 3: Save CSVs ---
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
print(f"\nSaved CSVs to {SPLITS_DIR}/")

# --- Step 4: Verify stratification ---
print("\nClass distribution per split (should be similar proportions):")
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    pct = (split_df["label"].value_counts(normalize=True) * 100).round(1)
    print(f"  {name}: {dict(pct)}")