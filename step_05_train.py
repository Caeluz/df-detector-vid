import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import random
from tqdm import tqdm
from step_04_deepfake_data import DeepFakeDetector, Trainer, plot_training_history


class FolderBasedDeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize dataset from a folder structure.

        Args:
            root_dir (str): Path to the root directory containing 'fake' and 'real' subfolders.
            transform (callable, optional): Optional transforms to apply.
        """
        self.root_dir = root_dir
        self.samples = []

        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            # Ensure ToTensor is included in custom transforms
            if not any(isinstance(t, transforms.ToTensor) for t in transform.transforms):
                transform_list = [transforms.ToTensor()] + \
                    list(transform.transforms)
                self.transform = transforms.Compose(transform_list)
            else:
                self.transform = transform

        # Label mapping
        self.class_to_idx = {'real': 1, 'fake': 0}

        # Load all image paths and their labels
        self._load_samples()

    def _load_samples(self):
        """Load all samples from folder structure."""
        for class_folder in ['real', 'fake']:
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.exists(class_path):
                print(f"Warning: Class folder not found: {class_path}")
                continue

            label = self.class_to_idx[class_folder]

            # Get all image files
            for img_name in tqdm(os.listdir(class_path), desc=f"Loading {class_folder} samples"):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path) and self._is_image_file(img_name):
                    self.samples.append((img_path, label))

        print(
            f"Found {len(self.samples)} images: {self._count_by_class()} in {self.root_dir}")

    def _count_by_class(self):
        """Count samples by class"""
        counts = {}
        for _, label in self.samples:
            class_name = list(self.class_to_idx.keys())[
                list(self.class_to_idx.values()).index(label)]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _is_image_file(self, filename):
        """Check if file is an image based on extension"""
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        return any(filename.lower().endswith(ext) for ext in img_extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load and convert image to tensor
            image = Image.open(img_path).convert('RGB')
            # This will convert to tensor and normalize
            image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default tensor with the same normalization as our transform
            default_tensor = torch.zeros((3, 128, 128))
            default_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(default_tensor)
            return default_tensor, torch.tensor(0)


# Updated main function to work with folder-based datasets
def main(train_dir, val_dir, output_dir, resume_checkpoint=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up transforms for training data
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to match expected input size
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Set up transforms for validation data (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to match expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    print("Loading training dataset...")
    train_dataset = FolderBasedDeepFakeDataset(
        train_dir,
        transform=train_transform
    )

    print("Loading validation dataset...")
    val_dataset = FolderBasedDeepFakeDataset(
        val_dir,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeepFakeDetector()
    trainer = Trainer(model, train_loader, val_loader, device,
                      checkpoint_dir=os.path.join(output_dir, 'checkpoints'))

    # Train the model
    history = trainer.train(num_epochs=30,
                            resume_checkpoint=resume_checkpoint,
                            early_stopping_patience=5)

    # Plot and save results
    plot_training_history(
        history,
        save_path=os.path.join(output_dir, 'training_history.png')
    )


if __name__ == "__main__":
    # Example usage with folder structure:
    main(
        # Contains 'fake' and 'real' subfolders
        train_dir='output_test_2_image/train',
        # Contains 'fake' and 'real' subfolders
        val_dir='output_test_2_image/val',
        output_dir='training_output_test_2_image',
        # Optional: resume_checkpoint='training_output_folder_based/checkpoints/checkpoint_epoch_X.pth'
    )
