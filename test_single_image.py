import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from dataset import IDX_TO_CLASS

# 1. Load Model
DEVICE = torch.device("cpu")
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("efficientnet_b0_brain_mri.pth", map_location=DEVICE))
model.eval()

# 2. Point to ANY image on your PC (change this path!)
IMAGE_PATH = "google_test.jpg" 

# 3. Preprocess (CLAHE + Resize + Normalize)
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_resized = cv2.resize(img, (224, 224)) / 255.0
tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
tensor = (tensor - mean) / std
input_tensor = tensor.unsqueeze(0)

# 4. Predict
output = model(input_tensor)
probabilities = torch.softmax(output, dim=1)[0]
predicted_idx = torch.argmax(probabilities).item()

print(f"\n--- RESULT ---")
print(f"Predicted Class: {IDX_TO_CLASS[predicted_idx].upper()}")
print(f"Confidence: {probabilities[predicted_idx].item() * 100:.2f}%")

# 5. Generate Grad-CAM
cam = GradCAM(model=model, target_layers=[model.features[-2]])
targets = [ClassifierOutputTarget(predicted_idx)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img_resized); plt.title("Original (CLAHE)")
plt.subplot(1, 2, 2); plt.imshow(visualization); plt.title("AI Attention (Grad-CAM)")
plt.show()