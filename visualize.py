import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataset import BrainMRIDataset, IDX_TO_CLASS

# Load one sample per class
csv_path = "data/splits/train.csv"
ds_no_clahe = BrainMRIDataset(csv_path, use_clahe=False)
ds_clahe = BrainMRIDataset(csv_path, use_clahe=True)

# Find one index per class
seen = {}
for i in range(len(ds_no_clahe)):
    _, label = ds_no_clahe[i]
    if label not in seen:
        seen[label] = i
    if len(seen) == 4:
        break

# Helper: undo normalization for display
def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
    img = img * std + mean
    return np.clip(img, 0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, label in enumerate(sorted(seen.keys())):
    idx = seen[label]
    img_no, _ = ds_no_clahe[idx]
    img_cl, _ = ds_clahe[idx]
    
    axes[0, col].imshow(unnormalize(img_no))
    axes[0, col].set_title(f"Original — {IDX_TO_CLASS[label]}")
    axes[0, col].axis("off")
    
    axes[1, col].imshow(unnormalize(img_cl))
    axes[1, col].set_title(f"CLAHE — {IDX_TO_CLASS[label]}")
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig("preprocessing_comparison.png", dpi=100)
plt.show()
print("Saved preprocessing_comparison.png")