import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import BrainMRIDataset

# --- 1. Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 4
MODEL_SAVE_PATH = "efficientnet_b0_brain_mri.pth" # New save path

print(f"Using device: {DEVICE}")

# Load datasets
train_dataset = BrainMRIDataset("data/splits/train.csv", use_clahe=True, augment=True)
val_dataset = BrainMRIDataset("data/splits/val.csv", use_clahe=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Load Pretrained EfficientNetB0 ---
print("Loading Pretrained EfficientNetB0...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Replace the final classifier layer for our 4 classes
# EfficientNet uses model.classifier, where [1] is the final Linear layer
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# --- 3. Training & Validation Loops ---
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False, desc="Training")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def run_stage(stage_name, epochs, lr):
    print(f"\n{'='*40}")
    print(f"Starting {stage_name} | Epochs: {epochs} | LR: {lr}")
    print(f"{'='*40}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved best model (Val Acc: {best_val_acc:.2f}%)")

# --- 4. Multi-Stage Transfer Learning (MSTL) ---

# STAGE 0: Train Classifier Only
for param in model.parameters(): param.requires_grad = False
for param in model.classifier.parameters(): param.requires_grad = True
run_stage("Stage 0: Classifier Only", epochs=20, lr=0.001) 

# STAGE 1: Unfreeze Last Conv Block (features[-1])
for param in model.features[-1].parameters(): param.requires_grad = True
run_stage("Stage 1: Partial Fine-Tuning", epochs=20, lr=0.0005) 

# STAGE 2: Full Fine-Tuning
for param in model.parameters(): param.requires_grad = True
run_stage("Stage 2: Full Fine-Tuning", epochs=20, lr=0.0001) 

print("\nTraining Complete! Best EfficientNet model saved.")