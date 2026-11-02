import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Class name -> integer label (consistent ordering everywhere)
CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class BrainMRIDataset(Dataset):
    """
    Loads a brain MRI dataset from a CSV produced by create_splits.py.
    Applies CLAHE (optional) + resize + normalize.
    Augmentation only when augment=True (use for training split only).
    """
    def __init__(self, csv_path, use_clahe=True, augment=False):
        self.df = pd.read_csv(csv_path)
        self.use_clahe = use_clahe
        
        # CLAHE object: clipLimit=2.0, tileGridSize=8x8 (standard values)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Build the torchvision transform pipeline
        steps = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
        ]
        if augment:
            steps += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        steps += [
            transforms.ToTensor(),  # uint8 -> float32 in [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],   # ImageNet stds
            ),
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"].replace("\\", "/")
        label = CLASS_TO_IDX[row["label"]]

        # --- Load ---
        img = cv2.imread(path)              # BGR
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # MRI is grayscale

        # --- CLAHE ---
        if self.use_clahe:
            img = self.clahe.apply(img)

        # --- Back to 3 channels (pretrained models expect RGB) ---
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # --- Resize + normalize ---
        img = self.transform(img)

        return img, label

# ---------- Quick test ----------
if __name__ == "__main__":
    ds_train = BrainMRIDataset("data/splits/train.csv", use_clahe=True, augment=True)
    ds_val = BrainMRIDataset("data/splits/val.csv", use_clahe=True, augment=False)
    
    print(f"Train size: {len(ds_train)}")
    print(f"Val size:   {len(ds_val)}")
    
    # Grab one batch
    loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    
    print(f"\nBatch shape:   {images.shape}")        # Should be (8, 3, 224, 224)
    print(f"Batch dtype:   {images.dtype}")          # Should be torch.float32
    print(f"Pixel range:   {images.min().item():.2f} to {images.max().item():.2f}")
    print(f"Labels:        {labels.tolist()}")
    print(f"Class names:   {[IDX_TO_CLASS[int(l)] for l in labels]}")