import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F  # Import softmax


class DeepFakeDetector(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DeepFakeDetector, self).__init__()
        # Individual layers to allow for visualization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(0.2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(0.2)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Updated classifier with additional layers and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),  # Added batch norm
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),   # Added batch norm
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),   # Added additional layer
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),   # Added batch norm
            nn.Dropout(dropout_rate),

            nn.Linear(256, 2)  # Output layer
        )

    def forward(self, x):
        # Forward pass with intermediates stored
        intermediates = {}

        x = self.conv1(x)
        intermediates['conv1'] = x
        x = self.relu1(x)
        intermediates['relu1'] = x
        x = self.bn1(x)
        intermediates['bn1'] = x
        x = self.dropout1(x)
        x = self.pool1(x)
        intermediates['pool1'] = x

        x = self.conv2(x)
        intermediates['conv2'] = x
        x = self.relu2(x)
        intermediates['relu2'] = x
        x = self.bn2(x)
        intermediates['bn2'] = x
        x = self.dropout2(x)
        x = self.pool2(x)
        intermediates['pool2'] = x

        x = self.conv3(x)
        intermediates['conv3'] = x
        x = self.relu3(x)
        intermediates['relu3'] = x
        x = self.bn3(x)
        intermediates['bn3'] = x
        x = self.dropout3(x)
        x = self.pool3(x)
        intermediates['pool3'] = x

        x = self.conv4(x)
        intermediates['conv4'] = x
        x = self.relu4(x)
        intermediates['relu4'] = x
        x = self.bn4(x)
        intermediates['bn4'] = x
        x = self.dropout4(x)
        x = self.pool4(x)
        intermediates['pool4'] = x

        x = x.view(x.size(0), -1)

        # Pass through classifier and store intermediate outputs
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            intermediates[f'classifier_{i}'] = x

        return x, intermediates


def preprocess_image(image_path, size=(128, 128)):  # Change size to (128, 128)
    """Preprocess an image for the model."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),  # Resize to 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def visualize_classifier_outputs(model, image_path, output_path=None):
    """Visualize outputs of the classifier layers."""
    # Load and preprocess image
    image = preprocess_image(image_path)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        _, intermediates = model(image)

    # Extract classifier layers
    classifier_layers = [
        key for key in intermediates.keys() if 'classifier' in key]

    # Visualize classifier outputs
    plt.figure(figsize=(15, 5))

    for i, layer_name in enumerate(classifier_layers):
        output = intermediates[layer_name].squeeze().cpu().numpy()

        # Normalize only if range is valid
        if output.max() > output.min():
            output = (output - output.min()) / (output.max() - output.min())

        plt.subplot(1, len(classifier_layers), i + 1)
        plt.imshow(output.reshape(1, -1), cmap='viridis', aspect='auto')
        plt.title(layer_name)
        plt.axis('off')

    final_output = intermediates['classifier_12'].squeeze().cpu().numpy()
    probabilities = F.softmax(torch.tensor(
        final_output), dim=0).numpy()  # Apply softmax

    print(
        f"Probabilities: Fake={probabilities[0]:.4f}, Real={probabilities[1]:.4f}")

    if probabilities[1] > probabilities[0]:
        print("Model predicts: REAL")
    else:
        print("Model predicts: FAKE")

    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)

    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DeepFakeDetector()

    # Set path to your image
    image_path = r"C:\Users\aaron\Pictures\Screenshots\Screenshot 2024-10-09 092103.png"

    # Visualize classifier outputs
    visualize_classifier_outputs(model, image_path, "classifier_outputs.png")
