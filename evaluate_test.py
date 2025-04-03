import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, auc, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from step_04_deepfake_data import DeepFakeDetector  # Import your model

# Define transformation for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class DeepFakeTestDataset(Dataset):
    """
    Custom dataset for loading images from test_metadata.csv.
    """

    def __init__(self, metadata_file, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['frame_path']
        label = self.metadata.iloc[idx]['label']

        image = Image.open(img_path).convert(
            'RGB')  # Read image and convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def evaluate_model(test_metadata_file, model_checkpoint, evaluation_folder):
    """
    Evaluate the model using images in test_metadata_file.
    """
    os.makedirs(evaluation_folder, exist_ok=True)
    predictions_file = os.path.join(evaluation_folder, 'predictions.csv')

    if os.path.exists(predictions_file):
        df = pd.read_csv(predictions_file)
        y_test = df['GroundTruth'].values
        y_pred = df['Prediction'].values
    else:
        # Load trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepFakeDetector()
        model.load_state_dict(torch.load(
            model_checkpoint, map_location=device))
        model.to(device)
        model.eval()

        # Load test dataset from metadata file
        test_dataset = DeepFakeTestDataset(
            test_metadata_file, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4)

        predictions = []
        ground_truth = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())

        # Save predictions to CSV
        y_test = np.array(ground_truth)
        y_pred = np.array(predictions)
        df = pd.DataFrame({'GroundTruth': y_test, 'Prediction': y_pred})
        df.to_csv(predictions_file, index=False)

    # Convert arrays to NumPy
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)

    # Check if both classes are present in y_test before computing ROC AUC score
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_pred)
    else:
        roc_auc = None

    # Print and save metrics
    metrics_output = f"""
    Accuracy: {accuracy}
    Precision: {precision}
    Recall: {recall}
    F1 Score: {f1}
    ROC AUC Score: {roc_auc if roc_auc is not None else 'N/A'}
    Average Precision Score: {avg_precision}
    """
    print(metrics_output)
    with open(os.path.join(evaluation_folder, 'evaluation_metrics.txt'), 'w') as f:
        f.write(metrics_output)

    # Save classification report
    class_report = classification_report(y_test, y_pred)
    with open(os.path.join(evaluation_folder, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(evaluation_folder, 'confusion_matrix.png'))
    plt.show()

    if roc_auc is not None:
        # Compute and Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(evaluation_folder, 'roc_curve.png'))
        plt.show()

    # Plot Evaluation Metrics as Bar Chart
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc if roc_auc is not None else 0,
        'Average Precision Score': avg_precision
    }

    plt.figure(figsize=(12, 8))
    bars = plt.bar(list(metrics.keys()), list(
        metrics.values()), color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Evaluation Metrics')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=2.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                 f'{yval:.2f}', ha='center', va='bottom')
    plt.savefig(os.path.join(evaluation_folder, 'evaluation_metrics.png'))
    plt.show()


if __name__ == '__main__':
    test_metadata_file = 'output_test_1_image/new_test_metadata.csv'
    model_checkpoint = 'training_output_test_1_image/checkpoints/best_model.pth'
    evaluation_folder = 'output_test_1_image/evaluations_2'
    # evaluation_folder = 'training_output_test_1_image/evaluations'
    evaluate_model(test_metadata_file, model_checkpoint, evaluation_folder)
