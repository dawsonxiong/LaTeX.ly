import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as f

# PyTorch settings
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

# Task 1: Image Preprocessing
data = torchvision.datasets.ImageFolder(
    root='data/train',
    transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

# Load and display a sample image
image_path = 'data/train/-/65_0.png'
image = Image.open(image_path)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
preprocessed_image = preprocess(image)

# Adding batch dimension
input_tensor = preprocessed_image.unsqueeze(0)

# Task 2: Data Loading and Preparation

def load_split(dataset, batch_size, test_split=0.3, random_seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split = int(np.floor(test_split * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]

    testset_size = len(test_indices)
    indices = list(range(testset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split = int(np.floor(0.5 * testset_size))
    val_indices, test_indices = indices[split:], indices[:split]

    # Creating data samplers:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Creating data loaders:
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=test_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=val_sampler)

    return train_loader, test_loader, val_loader

# Randomly splitting training set into train, test, validation sets
batch_size = 32
train_loader, test_loader, val_loader = load_split(data, batch_size, test_split=0.3)

# Task 3: Building a CNN and One-hot-encoding

class OCRNet(nn.Module):
    def __init__(self, num_features):
        super(OCRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(num_features, 86)  # Output has 86 classes instead of 10

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

num_features = 32 * 64 * 64 # 64px x 64px x 32 channels
model = OCRNet(num_features)

# Task 4: Set the Optimizer and Loss Functions

# Using Stochastic Gradient Descent (SGD) as the optimizer, and Cross Entropy Loss for loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Task 5: Training the Model

def train(model, train_loader, optimizer, criterion, num_epochs=20, print_every=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_count = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_count += 1

            # Print loss for each step
            if (i+1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Average Loss: {running_loss/running_count:.4f}")
                running_loss = 0.0
                running_count = 0

        # Print average loss after each epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"\nEnd of Epoch {epoch+1}/{num_epochs}, Average Epoch Loss: {epoch_loss:.4f}")

train(model, train_loader, optimizer, criterion, num_epochs=20)

# Task 6: Validate the Model

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

validate(model, val_loader, criterion)

# Task 7: Testing the Model

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

test(model, test_loader, criterion)

# Task 8: Saving and Loading the Model

torch.save(model.state_dict(), 'model/LaTeXly_v4.pth')

# Load model
model.load_state_dict(torch.load('model/LaTeXly_v4.pth'))

# Task 9: Making Predictions

# Corresponding symbols for each class
classes = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '\\Delta', 'a', '\\alpha', 
    'b', '\\beta', 'c', '\\cos', 'd', '\\div', 'e', '\\exists', 'f', '\\forall', '/', 'g', '\\geq', '\\gt', 'h',
    'i', '\\in', '\\infty', '\\int', 'j', 'k', '\\lambda', '\\leq', '\\lim', '\\log', '\\lt', 'm', '\\mu', 'n', '\\neq', 'p', '\\pi', 
    '\\pm', 'q', 'r', '\\rightarrow', 's', '\\sigma', '\\sin', '\\sum', 't', '\\tan', '\\theta', '\\times', 'u', 'A', 'B', 'C',
    'E', 'F', 'G', 'I', 'N', 'P', 'R', 'S', 'T', 'X', 'v', '|', 'w', 'x', 'y', 'z', '\\{', '\\}']

def predict(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB colour type
    image = transform(image).unsqueeze(0)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
        probabilities = f.softmax(output, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)
        predicted_class = classes[predicted_idx]
        confidence = predicted_prob.item() * 100  # convert to percentage

    plt.imshow(Image.open(image_path))
    plt.title(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')
    plt.axis('off')
    plt.show()
    return f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%'

image_path = 'data/train/lt/1.png'

# Call prediction function
predict(model, image_path, preprocess)
