## Import necessary libraries and set random seed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

torch.manual_seed(2022)

## Set parameters and data preprocessing
data_dir = "/Users/peixingrui/Desktop/天气识别/weather_photos"
batch_size = 30
img_height = 180
img_width = 180

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## Load datasets

dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(dataset.classes)
print(f"Number of classes in the dataset: {num_classes}")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 45 * 45, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 45 * 45)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Net()
print(model)
...


## Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Train the model
epochs = 32
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    train_loss = 0.0
    train_correct = 0
    val_loss = 0.0
    val_correct = 0

    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))
    train_accs.append(train_correct.double() / len(train_dataset))
    val_accs.append(val_correct.double() / len(val_dataset))
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {train_loss/len(train_loader):.3f}.. "
          f"Train accuracy: {train_correct.double() / len(train_dataset):.3f}.. "
          f"Validation loss: {val_loss/len(val_loader):.3f}.. "
          f"Validation accuracy: {val_correct.double() / len(val_dataset):.3f}")

## Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Load and predict using an image
from PIL import Image

image_path = "/Users/peixingrui/Desktop/WechatIMG5834.png"

try:
    image = Image.open(image_path)
    image.show()  # To ensure the image is correctly loaded
    print("Image loaded successfully.")

    # Use the previously defined transform for preprocessing
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(f"Input tensor shape: {input_tensor.shape}")  # To ensure the input tensor shape is correct

    # Ensure the model is in evaluation mode
    model.eval()
    print("Model set to evaluation mode.")

    # Use the model for prediction
    with torch.no_grad():
        print("Predicting...")
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        print("Prediction done.")

    # Print the predicted result
    weather_classes = dataset.classes
    print(f"The predicted weather is: {weather_classes[predicted.item()]}")

except Exception as e:
    print(f"An error occurred: {e}")
