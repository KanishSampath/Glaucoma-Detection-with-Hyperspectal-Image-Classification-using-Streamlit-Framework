import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
base_dir = "E:/project extension/GLAUCOMA-DETECTION/eyepac-light-v2-512-jpg"
metadata_file = "new_metadata.csv"

# Load metadata
metadata = pd.read_csv(metadata_file)

# Custom Dataset Class
class GlaucomaDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = os.path.join(base_dir, row['file_path'].lstrip('/'))
        
        if not os.path.exists(image_path):
            print(f"Warning: Missing file {image_path}, using a blank image instead.")
            image = Image.new("RGB", (128, 128), (0, 0, 0))  # Black image as a placeholder
        else:
            image = Image.open(image_path).convert("RGB")
        
        label = row['label_binary']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split dataset
train_metadata, val_metadata = train_test_split(metadata, test_size=0.2, random_state=42)
train_dataset = GlaucomaDataset(train_metadata, transform=transform)
val_dataset = GlaucomaDataset(val_metadata, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define CNN Model
class GlaucomaCNN(nn.Module):
    def __init__(self):
        super(GlaucomaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device):
    model.to(device)
    start_time = time.time()
    train_acc_list, val_acc_list = [], []
    
    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        val_acc_list.append(val_acc)
        print(f"Validation Accuracy: {val_acc:.2f}%")
    
    end_time = time.time()
    print(f"Training completed in {((end_time - start_time)/60):.2f} minutes.")
    return model, train_acc_list, val_acc_list

# Train ResNet Model
resnet_model = models.resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model = resnet_model.to(device)
resnet_criterion = nn.CrossEntropyLoss()
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

print("Training ResNet model...")
resnet_model, resnet_train_acc, resnet_val_acc = train_model(resnet_model, train_loader, val_loader, resnet_criterion, resnet_optimizer, num_epochs=5, device=device)
torch.save(resnet_model.state_dict(), 'resnet_glaucoma_gpu.pth')
print("ResNet Model saved as 'resnet_glaucoma_gpu.pth'.")

# Train CNN Model
cnn_model = GlaucomaCNN().to(device)
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

print("Training CNN model...")
cnn_model, cnn_train_acc, cnn_val_acc = train_model(cnn_model, train_loader, val_loader, cnn_criterion, cnn_optimizer, num_epochs=5, device=device)
torch.save(cnn_model.state_dict(), 'cnn_glaucoma_gpu.pth')
print("CNN Model saved as 'cnn_glaucoma_gpu.pth'.")

# Plot Comparison Chart
plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), cnn_train_acc, label='CNN Train Acc', marker='o')
plt.plot(range(1, 6), cnn_val_acc, label='CNN Val Acc', marker='o')
plt.plot(range(1, 6), resnet_train_acc, label='ResNet Train Acc', marker='s')
plt.plot(range(1, 6), resnet_val_acc, label='ResNet Val Acc', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of CNN and ResNet18 for Glaucoma Detection')
plt.legend()
plt.grid()
plt.show()
