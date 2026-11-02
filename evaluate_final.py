import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset import BrainMRIDataset, IDX_TO_CLASS
from train_ensemble import EnsembleModel # Import the ensemble architecture we just built

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
test_dataset = BrainMRIDataset("data/splits/test.csv", use_clahe=True, augment=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = [IDX_TO_CLASS[i] for i in range(4)]

# Helper function to get predictions and probabilities
def get_predictions(model, is_ensemble=False):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_probs)

# 1. Setup Models
print("Loading all models...")
resnet = models.resnet50(); resnet.fc = nn.Linear(resnet.fc.in_features, 4)
resnet.load_state_dict(torch.load("resnet50_brain_mri.pth", map_location=DEVICE)); resnet.to(DEVICE)

effnet = models.efficientnet_b0(); effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 4)
effnet.load_state_dict(torch.load("efficientnet_b0_brain_mri.pth", map_location=DEVICE)); effnet.to(DEVICE)

ens_no_attn = EnsembleModel("resnet50_brain_mri.pth", "efficientnet_b0_brain_mri.pth", use_attention=False)
ens_no_attn.load_state_dict(torch.load("ensemble_no_attention.pth", map_location=DEVICE)); ens_no_attn.to(DEVICE)

ens_attn = EnsembleModel("resnet50_brain_mri.pth", "efficientnet_b0_brain_mri.pth", use_attention=True)
ens_attn.load_state_dict(torch.load("ensemble_with_attention.pth", map_location=DEVICE)); ens_attn.to(DEVICE)

models = [
    ("ResNet50 (Config A)", resnet),
    ("EfficientNetB0 (Config B)", effnet),
    ("Ensemble No-Attention (Config C)", ens_no_attn),
    ("Ensemble Attention (Config D)", ens_attn)
]

# 2. Evaluate and Plot ROC
plt.figure(figsize=(10, 8))

for name, model in models:
    print(f"\nEvaluating {name}...")
    y_true, y_probs = get_predictions(model)
    y_pred = np.argmax(y_probs, axis=1)
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Plot ROC Curve (Macro-average)
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 0], pos_label=0) # We just need one class to plot a line for the macro average
    # Calculate macro-average ROC properly
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for i in range(4):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= 4
    macro_auc = auc(all_fpr, mean_tpr)
    
    plt.plot(all_fpr, mean_tpr, label=f'{name} (AUC = {macro_auc:.4f})')

# Finalize ROC Plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve (Macro-Average)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("final_roc_curves.png", dpi=150)
plt.show()
print("\nSaved final_roc_curves.png")