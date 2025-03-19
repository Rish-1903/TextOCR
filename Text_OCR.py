import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pdf2image import convert_from_path
from torchvision import transforms

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert PDFs to images
def pdfs_to_images(pdfs_folder, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    for pdf_name in os.listdir(pdfs_folder):
        # Skip the specific PDF file
        if pdf_name == "PORCONES.228.35 – 1636.pdf":
            continue
        
        if pdf_name.endswith(".pdf"):
            pdf_path = os.path.join(pdfs_folder, pdf_name)
            folder_name = pdf_name.replace(".pdf", "")
            output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                image.save(f"{output_folder}/page_{i+1}.jpg", "JPEG")

# Dataset class
transform = transforms.Compose([
    transforms.Resize((64, 256)),  # Increased resolution
    transforms.Grayscale(),
    transforms.RandomRotation(5),  # Small rotations
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset class
class OCRDataset(Dataset):
    def __init__(self, image_root, label_encoder):
        self.image_root = image_root
        self.image_paths = []
        self.labels = []
        self.label_encoder = label_encoder
        
        for folder_name in sorted(os.listdir(image_root)):  
            folder_path = os.path.join(image_root, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(".jpg"):
                        self.image_paths.append(os.path.join(folder_path, image_name))
                        self.labels.append(folder_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for ResNet
        image = transforms.ToPILImage()(image)
        image = transform(image)
        
        # Encode label
        label_encoded = self.label_encoder.transform([label])[0]
        
        return image, torch.tensor(label_encoded, dtype=torch.long)

# Improved CRNN model with ResNet
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3])  # Remove last few layers
        
        self.rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=3, bidirectional=True)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  # Feature extraction
        print(f"Shape after feature extractor: {x.shape}")  # Debugging
    
        x = x.permute(0, 2, 3, 1)  # Change shape to (batch, height, width, channels)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # Flatten spatial dimensions (batch, seq_len, features)
        x = x.permute(1, 0, 2)  # Convert to (seq_len, batch, feature_dim) for RNN
    
        print(f"Shape before RNN: {x.shape}")  # Debugging
    
        x, _ = self.rnn(x)  # Pass through RNN
        x = self.fc(x.mean(dim=0))  # Fully connected layer (batch, num_classes)
    
        return x



# Training function
def train(model, dataloader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy: {100 * correct / total}%")

# Convert PDFs to images
pdfs_to_images("Pdfs", "images")

# Collect labels and encode
all_labels = []
for folder_name in os.listdir("images"):
    folder_path = os.path.join("images", folder_name)
    if os.path.isdir(folder_path):
        all_labels.append(folder_name)

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Create dataset and dataloader
dataset = OCRDataset("images", label_encoder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = CRNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
train(model, dataloader, criterion, optimizer, num_epochs=30)

# Evaluate the model
evaluate(model, dataloader)