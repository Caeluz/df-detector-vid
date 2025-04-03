import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, auc, classification_report
from torch.utils.data import DataLoader
from step_04_deepfake_data import DeepFakeDetector, DeepFakeDataset
from tqdm import tqdm  # Import tqdm for progress bar
import os


def evaluate_model(evaluation_folder, data_type='image'):
    # Create evaluations directory if it doesn't exist
    os.makedirs(evaluation_folder, exist_ok=True)

    predictions_file = os.path.join(evaluation_folder, 'predictions.csv')

    # Initialize variables

    if os.path.exists(predictions_file):
        # Load predictions and ground truth from CSV file
        df = pd.read_csv(predictions_file)
        y_test = df['GroundTruth'].values
        y_pred = df['Prediction'].values
    else:
        # Step 1: Load the model
        model_path = 'training_output_test_1_image/checkpoints/best_model.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepFakeDetector()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Step 2: Load the test dataset
        if data_type == 'image':
            test_metadata_path = 'output_test_1_image/new_test_metadata.csv'
            test_dataset = DeepFakeDataset(test_metadata_path)
        elif data_type == 'npy':
            test_metadata_path = 'output_test_1_image/new_test_metadata.csv'
            test_dataset = DeepFakeDataset(test_metadata_path, file_type='npy')
        else:
            raise ValueError("Unsupported data type. Use 'image' or 'npy'.")

        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Step 3: Make predictions
        predictions = []
        ground_truth = []

        # Step 4: Iterate over the test dataset and make predictions
        with torch.no_grad():
            # Add tqdm progress bar
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays
        y_test = np.array(ground_truth)
        y_pred = np.array(predictions)

        # Save predictions and ground truth to a CSV file
        df = pd.DataFrame({'GroundTruth': y_test, 'Prediction': y_pred})
        df.to_csv(predictions_file, index=False)

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)

    # Print and save metrics
    metrics_output = f"""
    Accuracy: {accuracy}
    Precision: {precision}
    Recall: {recall}
    F1 Score: {f1}
    ROC AUC Score: {roc_auc}
    Average Precision Score: {avg_precision}
    """
    print(metrics_output)
    with open(os.path.join(evaluation_folder, 'evaluation_metrics.txt'), 'w') as f:
        f.write(metrics_output)

    # Print and save classification report
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    with open(os.path.join(evaluation_folder, 'classification_report.txt'), 'w') as f:
        f.write(class_report)  # type: ignore

    # Step 5: Plot and save confusion matrix and ROC curve (Visualize the results)
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(evaluation_folder, 'confusion_matrix.png'))
    plt.show()

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(evaluation_folder, 'roc_curve.png'))
    plt.show()

    # Plot and save evaluation metrics as a bar chart
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
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
    evaluate_model(evaluation_folder='new_model/evaluations',
                   data_type='image')
