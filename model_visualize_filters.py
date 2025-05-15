import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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

        # Adding classifier (for a complete model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # Binary classification: real or fake
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, intermediates


def preprocess_image(image_path, size=(224, 224)):
    """Preprocess an image for the model."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def visualize_feature_maps(model, image_path, output_path=None):
    """Visualize feature maps at different stages of the model."""
    # Load and preprocess image
    image = preprocess_image(image_path)

    # Set model to evaluation mode
    model.eval()

    # Get the original image for display
    orig_img = Image.open(image_path).convert('RGB')
    orig_img = orig_img.resize((224, 224))

    # Forward pass
    with torch.no_grad():
        _, intermediates = model(image)

    # Create figure for visualization
    plt.figure(figsize=(20, 12))

    # Plot original image
    plt.subplot(2, 3, 1)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')

    # Plot feature maps from each convolutional stage
    layer_names = ['pool1', 'pool2', 'pool3', 'pool4']
    for i, layer_name in enumerate(layer_names):
        # Get feature maps
        feature_maps = intermediates[layer_name].squeeze().cpu().numpy()

        # Calculate number of channels
        num_channels = feature_maps.shape[0]

        # Create a composite image by averaging across channels
        composite = np.mean(feature_maps, axis=0)

        # Normalize for visualization
        composite = (composite - composite.min()) / \
            (composite.max() - composite.min() + 1e-8)

        # Plot
        plt.subplot(2, 3, i + 2)
        plt.imshow(composite, cmap='viridis')
        plt.title(f'Layer: {layer_name}\nChannels: {num_channels}')
        plt.axis('off')

    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)

    plt.show()


def visualize_individual_filters(model, image_path, layer_name='conv1', num_filters=16, output_path=None):
    """Visualize individual filter activations for a specific layer."""
    # Load and preprocess image
    image = preprocess_image(image_path)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        _, intermediates = model(image)

    # Get feature maps for the specified layer
    if layer_name in intermediates:
        feature_maps = intermediates[layer_name].squeeze().cpu().numpy()

        # Number of filters to visualize (up to max)
        max_filters = min(num_filters, feature_maps.shape[0])

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(max_filters)))

        # Create figure
        plt.figure(figsize=(15, 15))

        for i in range(max_filters):
            plt.subplot(grid_size, grid_size, i + 1)

            # Normalize the feature map
            feature_map = feature_maps[i]
            feature_map = (feature_map - feature_map.min()) / \
                (feature_map.max() - feature_map.min() + 1e-8)

            plt.imshow(feature_map, cmap='viridis')
            plt.title(f'Filter {i + 1}')
            plt.axis('off')

        plt.suptitle(f'Feature Maps for Layer: {layer_name}', fontsize=16)
        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)

        plt.show()
    else:
        print(f"Layer {layer_name} not found in intermediates.")


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DeepFakeDetector()

    # Set path to your image
    image_path = r"C:\Users\aaron\Documents\df\Detector\df-detector-vid\output_test_2_image\train\fake\fake_id3_id31_0004_480_90852.jpg"

    # Visualize feature maps at different stages
    visualize_feature_maps(model, image_path, "feature_maps_overview.png")

    # Visualize individual filters for different layers
    visualize_individual_filters(
        model, image_path, 'conv1', num_filters=16, output_path="conv1_filters.png")
    visualize_individual_filters(
        model, image_path, 'conv2', num_filters=16, output_path="conv2_filters.png")
    visualize_individual_filters(
        model, image_path, 'conv3', num_filters=16, output_path="conv3_filters.png")
    visualize_individual_filters(
        model, image_path, 'conv4', num_filters=16, output_path="conv4_filters.png")
