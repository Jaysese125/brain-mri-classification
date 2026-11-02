import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pandas as pd

from dataset import BrainMRIDataset, IDX_TO_CLASS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Load your best model (EfficientNetB0)
print("Loading EfficientNetB0...")
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("efficientnet_b0_brain_mri.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 2. Define the Target Layer for Grad-CAM
# For EfficientNet, the last convolutional layer is in the final block of the 'features' module
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# 3. Load some test images
test_df = pd.read_csv("data/splits/test.csv").sample(4, random_state=42)
dataset = BrainMRIDataset("data/splits/test.csv", use_clahe=True, augment=False)

# Find the indices of the sampled images in the dataset
indices = [dataset.df[dataset.df['path'] == path].index[0] for path in test_df['path']]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for i, idx in enumerate(indices):
    image_tensor, label = dataset[idx]
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Get prediction
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Generate Grad-CAM
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare image for display (unnormalize)
    img_display = image_tensor.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    # Overlay CAM on image
    visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
    
    # Plot Original
    axes[0, i].imshow(img_display)
    axes[0, i].set_title(f"Original\nTrue: {IDX_TO_CLASS[label]}")
    axes[0, i].axis('off')
    
    # Plot CAM
    axes[1, i].imshow(visualization)
    axes[1, i].set_title(f"Grad-CAM\nPred: {IDX_TO_CLASS[pred_class]}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("gradcam_results.png", dpi=150)
plt.show()
print("\nSaved gradcam_results.png")