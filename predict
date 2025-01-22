import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as f
from PIL import Image

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

# Load model
num_features = 32 * 64 * 64
model = OCRNet(num_features)
model.load_state_dict(torch.load('model/LaTeXly_v3.pth'))
model.eval()

classes = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '\\Delta', 'a', '\\alpha',
    'b', '\\beta', 'c', '\\cos', 'd', '\\div', 'e', '\\exists', 'f', '\\forall', '/', 'g', '\\geq', '\\gt', 'h',
    'i', '\\in', '\\infty', '\\int', 'j', 'k', '\\lambda', '\\leq', '\\lim', '\\log', '\\lt', 'm', '\\mu', 'n', '\\neq', 'p', '\\pi',
    '\\pm', 'q', 'r', '\\rightarrow', 's', '\\sigma', '\\sin', '\\sum', 't', '\\tan', '\\theta', '\\times', 'u', 'A', 'B', 'C',
    'E', 'F', 'G', 'I', 'N', 'P', 'R', 'S', 'T', 'X', 'v', '|', 'w', 'x', 'y', 'z', '\\{', '\\}']

def predict_out(img_path):
    image = Image.open(img_path).convert('RGB')  # Convert to RGB colour type
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = f.softmax(output, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)
        predicted_class = classes[predicted_idx]
        confidence = predicted_prob.item() * 100  # convert to percentage

    return f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%'
