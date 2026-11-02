import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import BrainMRIDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 4

# --- Attention Mechanism ---
class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention_weights(x)
        return x * weights

# --- Ensemble Architecture ---
class EnsembleModel(nn.Module):
    def __init__(self, resnet_path, efficientnet_path, use_attention=True):
        super(EnsembleModel, self).__init__()
        
        self.resnet = models.resnet50()
        self.resnet.fc = nn.Identity() 
        self.resnet.load_state_dict(torch.load(resnet_path, map_location=DEVICE), strict=False)
        
        self.efficientnet = models.efficientnet_b0()
        self.efficientnet.classifier[1] = nn.Identity() 
        self.efficientnet.load_state_dict(torch.load(efficientnet_path, map_location=DEVICE), strict=False)
        
        for param in self.resnet.parameters(): param.requires_grad = False
        for param in self.efficientnet.parameters(): param.requires_grad = False
        
        fusion_dim = 2048 + 1280
        self.use_attention = use_attention
        
        if self.use_attention:
            self.attention = AttentionBlock(fusion_dim)
            
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        resnet_feat = self.resnet(x)
        efficientnet_feat = self.efficientnet(x)
        
        fused_features = torch.cat((resnet_feat, efficientnet_feat), dim=1)
        
        if self.use_attention:
            fused_features = self.attention(fused_features)
            
        out = self.classifier(fused_features)
        return out

def train_and_evaluate(model, model_name, save_path, epochs=15):
    print(f"\n{'='*40}\nTraining {model_name}\n{'='*40}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    train_dataset = BrainMRIDataset("data/splits/train.csv", use_clahe=True, augment=True)
    val_dataset = BrainMRIDataset("data/splits/val.csv", use_clahe=True, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
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
            
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model (Val Acc: {best_val_acc:.2f}%)")

# ==========================================
# SAFETY BLOCK: This ensures the training 
# only runs if you execute this file directly.
# ==========================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    model_no_attn = EnsembleModel("resnet50_brain_mri.pth", "efficientnet_b0_brain_mri.pth", use_attention=False).to(DEVICE)
    train_and_evaluate(model_no_attn, "Feature-Level Ensemble (No Attention)", "ensemble_no_attention.pth")

    model_attn = EnsembleModel("resnet50_brain_mri.pth", "efficientnet_b0_brain_mri.pth", use_attention=True).to(DEVICE)
    train_and_evaluate(model_attn, "Feature-Level Ensemble (With Attention)", "ensemble_with_attention.pth")

    print("\nEnsemble Training Complete!")