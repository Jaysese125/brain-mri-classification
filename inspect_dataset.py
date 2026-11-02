from pathlib import Path

DATA_DIR = Path("data/raw")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

total = 0
for split in ["Training", "Testing"]:
    split_path = DATA_DIR / split
    print(f"\n=== {split} ===")
    split_total = 0
    for cls in CLASSES:
        cls_dir = split_path / cls
        if not cls_dir.exists():
            print(f"  [MISSING] {cls}")
            continue
        n = sum(1 for p in cls_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
        print(f"  {cls:12s}: {n} images")
        split_total += n
    print(f"  {'TOTAL':12s}: {split_total} images")
    total += split_total

print(f"\nGrand total: {total} images")