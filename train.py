import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import time

# Thiết bị GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("✅ Sử dụng GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Đang sử dụng CPU.")

# Các lớp hoa
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset train/test
train_dataset = datasets.ImageFolder('data/flowers_split/train', transform=transform)
test_dataset = datasets.ImageFolder('data/flowers_split/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

def train_model(model, optimizer, criterion, n_epochs=5):
    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        start_time = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        end_time = time.time()
        print(f"✅ Epoch {epoch+1} hoàn thành - Loss: {running_loss/len(train_loader):.4f} - "
              f"Accuracy: {100*correct/total:.2f}% - Thời gian: {end_time - start_time:.2f} giây")

def evaluate_model(model):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100 * correct / total
    print(f"📊 Độ chính xác trên tập test: {acc:.2f}%")
    return acc

# ------------------- Huấn luyện ResNet-50 -------------------
print("🚀 Huấn luyện ResNet-50...")
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
optimizer_res = torch.optim.Adam(resnet.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

train_model(resnet, optimizer_res, criterion, n_epochs=5)
evaluate_model(resnet)
torch.save(resnet.state_dict(), 'resnet_flowers.pth')
print("💾 Lưu xong: resnet_flowers.pth")

# ------------------- Huấn luyện EfficientNet-B0 -------------------
print("🚀 Huấn luyện EfficientNet-B0...")
effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, len(class_names))
optimizer_eff = torch.optim.Adam(effnet.parameters(), lr=1e-4)

train_model(effnet, optimizer_eff, criterion, n_epochs=5)
evaluate_model(effnet)
torch.save(effnet.state_dict(), 'effnet_flowers.pth')
print("💾 Lưu xong: effnet_flowers.pth")
