### **Flowchart for Training Process**

```plaintext
Start
  |
  v
Initialize Training Pipeline
  |
  v
Load Training and Validation Metadata
  |
  v
Create Training and Validation Datasets
  |
  v
Create DataLoaders for Training and Validation
  |
  v
Initialize Model and Trainer
  |
  v
For Each Epoch:
  +-------------------------------+
  |                               |
  | Train Model on Training Data  |
  |                               |
  +-------------------------------+
  |
  v
Validate Model on Validation Data
  |
  v
Save Metrics and Checkpoints
  |
  +-----------------------------------+
  |                                   |
  | Early Stopping Triggered?         |
  |   Yes -> Stop Training            |
  |   No  -> Continue to Next Epoch   |
  +-----------------------------------+
  |
  v
Save Best Model and Training History
  |
  v
Plot Training and Validation Metrics
  |
  v
End
```

---

### **Documentation for Training Process**

#### **1. Initialization**
- **Script**: step_05_train.py
- **Purpose**: Train a deep learning model to classify real and fake frames using the `DeepFakeDetector` model.
- **Key Components**:
  - **Training Metadata**: Path to the CSV file containing training data (`train_metadata.csv`).
  - **Validation Metadata**: Path to the CSV file containing validation data (`val_metadata.csv`).
  - **Output Directory**: Directory to save checkpoints, logs, and training history.

---

#### **2. Dataset Preparation**
- **Class**: `DeepFakeDataset` (from step_04_deepfake_data.py)
- **Purpose**: Load and preprocess data for training and validation.
- **Steps**:
  1. Load metadata from the CSV file.
  2. Verify that all frame files exist.
  3. Apply transformations (e.g., resizing, normalization, data augmentation).
  4. Return a tensor and label for each frame.

---

#### **3. DataLoaders**
- **Purpose**: Efficiently load batches of data for training and validation.
- **Key Parameters**:
  - `batch_size`: Number of samples per batch (default: 32).
  - `shuffle`: Shuffle training data to improve generalization.
  - `num_workers`: Number of subprocesses for data loading.

---

#### **4. Model Initialization**
- **Class**: `DeepFakeDetector` (from step_04_deepfake_data.py)
- **Purpose**: Define the CNN architecture for feature extraction and classification.
- **Key Features**:
  - Convolutional layers with batch normalization and dropout for regularization.
  - Fully connected layers for classification.
  - Temperature scaling for logits during inference.

---

#### **5. Training Process**
- **Class**: `Trainer` (from step_04_deepfake_data.py)
- **Purpose**: Manage the training and validation process.
- **Key Methods**:
  - `train_epoch`: Train the model on the training dataset for one epoch.
  - `validate`: Evaluate the model on the validation dataset.
  - `save_checkpoint`: Save model state and metrics after each epoch.
  - `load_checkpoint`: Resume training from a saved checkpoint.

---

#### **6. Early Stopping**
- **Class**: `EarlyStopping` (from step_04_deepfake_data.py)
- **Purpose**: Stop training if validation loss does not improve for a specified number of epochs.
- **Parameters**:
  - `patience`: Number of epochs to wait before stopping.
  - `min_delta`: Minimum change in validation loss to qualify as an improvement.

---

#### **7. Metrics and Visualization**
- **Metrics**:
  - **Loss**: Cross-entropy loss for classification.
  - **Accuracy**: Percentage of correctly classified samples.
- **Visualization**:
  - Loss and accuracy curves for training and validation.
  - Saved as `training_history.png` in the output directory.

---

### **Key Files and Outputs**
1. **Checkpoints**:
   - Saved in the `checkpoints` directory.
   - Includes model state, optimizer state, and metrics for each epoch.
   - Best model is saved as `best_model.pth`.

2. **Training History**:
   - Saved as `training_history.png`.
   - Includes plots of loss and accuracy over epochs.

3. **Logs**:
   - Printed to the console during training.
   - Includes metrics for each epoch and early stopping notifications.

---

### **Example Usage**

```python
if __name__ == "__main__":
    main(
        train_metadata_path='output_test_1_image/new_train_metadata.csv',
        val_metadata_path='output_test_1_image/new_val_metadata.csv',
        dataset_type='image',
        output_dir='training_output_test_1_image',
        resume_checkpoint=None  # Optional: Path to a checkpoint to resume training
    )
```

- **Parameters**:
  - `train_metadata_path`: Path to the training metadata CSV file.
  - `val_metadata_path`: Path to the validation metadata CSV file.
  - `dataset_type`: Type of dataset (`'image'` or `'npy'`).
  - `output_dir`: Directory to save training outputs.
  - `resume_checkpoint`: Path to a checkpoint file to resume training (optional).

---

This flowchart and documentation provide a high-level overview of the training process, including dataset preparation, model training, validation, and checkpointing. Let me know if you need further details!