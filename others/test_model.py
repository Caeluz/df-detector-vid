from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch
import cv2
import sys
import os
from step_04_deepfake_data import DeepFakeDataset, DeepFakeDetector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def evaluate_model(model, test_loader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and ground truth labels
    predictions = []
    ground_truth = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Evaluate the model on the test dataset
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            # Move inputs and labels to the appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    # Load the model
    model_path = 'training_output_test_1/checkpoints/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model
    model = DeepFakeDetector()

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Load the test dataset
    test_metadata_path = 'output/metadata.csv'  # Path to the metadata file
    print(f"Loading test dataset from: {test_metadata_path}")

    test_dataset = DeepFakeDataset(test_metadata_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate the model
    metrics = evaluate_model(model, test_loader)
