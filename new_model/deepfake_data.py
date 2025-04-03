import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import os


class DeepFakeDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        """
        Initialize dataset from metadata CSV file
        Args:
            metadata_path: Path to metadata CSV file (can be metadata.csv, train_metadata.csv, or val_metadata.csv)
            transform: Optional transforms to apply
        """
        self.metadata_df = pd.read_csv(metadata_path)
        self.transform = transform

        # Verify all frame files exist
        self._verify_frames()

    def _verify_frames(self):
        """Verify all frame files exist and remove missing entries"""
        valid_frames = []
        for idx, row in tqdm(self.metadata_df.iterrows(), desc="Verifying frames", total=len(self.metadata_df)):
            if os.path.exists(row['frame_path']):
                valid_frames.append(idx)
            else:
                print(f"Warning: Frame not found: {row['frame_path']}")

        self.metadata_df = self.metadata_df.iloc[valid_frames].reset_index(
            drop=True)
        print(f"Found {len(self.metadata_df)} valid frames")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        try:
            # Load face array from .npy file
            face = np.load(row['frame_path'])

            # Convert to torch tensor and normalize
            face = torch.from_numpy(face).float()
            face = face.permute(2, 0, 1)  # Change from HWC to CHW format
            face = face / 255.0  # Normalize to [0, 1]

            if self.transform:
                face = self.transform(face)

            label = torch.tensor(row['label'], dtype=torch.long)

            return face, label

        except Exception as e:
            print(f"Error loading frame {row['frame_path']}: {str(e)}")
            # Return a default item
            return torch.zeros((3, 128, 128)), torch.tensor(0)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        Initialize early stopping object
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): 'min' for monitoring decreasing values (like loss), 'max' for increasing values (like accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.stop = False
        self.best_epoch = None

    def __call__(self, current_value, epoch):
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:  # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.stop = True
            return True
        return False


class DeepFakeDetector(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DeepFakeDetector, self).__init__()

        # CNN Feature Extraction with increased regularization
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),  # Added dropout to conv layers
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2)
        )

        # Classification layers with increased regularization
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

            nn.Linear(256, 2)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, temperature=1.0):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # Apply temperature scaling to logits
        return x / temperature


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

    def save_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        path = os.path.join(self.checkpoint_dir,
                            f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        self.scheduler.step(val_loss)

        return val_loss, 100. * correct / total

    def train(self, num_epochs, resume_checkpoint=None, early_stopping_patience=5):
        start_epoch = 0
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            start_epoch = self.load_checkpoint(resume_checkpoint)
            print(f"Resuming from epoch {start_epoch}")

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        best_val_acc = 0
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, mode='min')  # monitoring val_loss

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Save metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save checkpoint
            self.save_checkpoint(epoch + 1, train_loss,
                                 val_loss, train_acc, val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint_dir, 'best_model.pth'))

            # Early stopping check
            if early_stopping(val_loss, epoch):
                print(
                    f'\nEarly stopping triggered! Best epoch was {early_stopping.best_epoch + 1 if early_stopping.best_epoch is not None else "N/A"} with validation loss: {early_stopping.best_value:.4f}')
                # Load the best model before stopping
                self.model.load_state_dict(torch.load(
                    os.path.join(self.checkpoint_dir, 'best_model.pth')))
                break

        return history


def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main(train_metadata_path, val_metadata_path, output_dir, resume_checkpoint=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    # Create datasets
    print("Loading training dataset...")
    train_dataset = DeepFakeDataset(train_metadata_path, transform=transform)
    print("Loading validation dataset...")
    val_dataset = DeepFakeDataset(val_metadata_path)

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
    # Example usage:
    main(
        train_metadata_path=r'C:\path\to\train_metadata.csv',
        val_metadata_path=r'C:\path\to\val_metadata.csv',
        output_dir='training_output_2',
        # Set to checkpoint path to resume training
        resume_checkpoint='training_output_2/checkpoints/checkpoint_epoch_9.pth'
    )
