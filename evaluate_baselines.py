import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset import BrainMRIDataset, IDX_TO_CLASS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

print(f"Using device: {DEVICE}")

# Load the unseen Test dataset
test_dataset = BrainMRIDataset("data/splits/test.csv", use_clahe=True, augment=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = [IDX_TO_CLASS[i] for i in range(4)]

def calculate_specificity(y_true, y_pred):
    """Calculate specificity for each class and macro average."""
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn_sum = mcm[:, 0, 0]
    fp_sum = mcm[:, 0, 1]
    
    # Specificity = TN / (TN + FP)
    specificity = tn_sum / (tn_sum + fp_sum + 1e-10)
    return specificity.mean()

def evaluate_model(model_name, model, model_path):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*50}")
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 1. Print Classification Report (Accuracy, Precision, Recall, F1)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 2. Calculate and print Specificity
    specificity = calculate_specificity(all_labels, all_preds)
    print(f"Macro-Averaged Specificity: {specificity:.4f}")
    
    # 3. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    filename = f'cm_{model_name}.png'
    plt.savefig(filename, dpi=150)
    print(f"--> Saved confusion matrix to {filename}")
    plt.close()

# --- 1. Evaluate ResNet50 (Config A) ---
print("Loading ResNet50...")
resnet = models.resnet50()
resnet.fc = nn.Linear(resnet.fc.in_features, 4)
evaluate_model("ResNet50", resnet, "resnet50_brain_mri.pth")

# --- 2. Evaluate EfficientNetB0 (Config B) ---
print("Loading EfficientNetB0...")
efficientnet = models.efficientnet_b0()
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 4)
evaluate_model("EfficientNetB0", efficientnet, "efficientnet_b0_brain_mri.pth")

print("\nAll Baseline Evaluations Complete!")