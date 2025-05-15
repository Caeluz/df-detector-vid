Here is a flowchart and documentation for the step_01_video_data_processor.py script. This script processes video datasets by extracting frames, performing face detection, and organizing the data into train, validation, and test splits.

---

### **Flowchart**

```plaintext
Start
  |
  v
Initialize VideoDataProcessor
  |
  v
Create Output Directories
  |
  v
Load Test Video List or Randomly Select Test Videos
  |
  v
Process Dataset
  |
  +--> For Each Directory in Video Dataset
  |       |
  |       v
  |   For Each Video in Directory
  |       |
  |       v
  |   Process Video Frames
  |       |
  |       +--> Perform Face Detection (if enabled)
  |       |       |
  |       |       v
  |       |   Save Processed Frames
  |       |
  |       v
  |   Save Metadata and Checkpoints
  |
  v
Create Train/Validation Split
  |
  v
Save Dataset Summary
  |
  v
End
```

---

### **Documentation**

#### **1. Initialization**
- **Class**: `VideoDataProcessor`
- **Purpose**: Initializes the processor with configurable parameters such as input/output paths, face detection settings, and frame sampling rate.
- **Key Methods**:
  - `_setup_face_detector`: Sets up the MTCNN face detector.
  - `_create_output_directories`: Creates necessary directories for storing processed data.

#### **2. Test Video Preparation**
- **Method**: `prepare_test_videos`
- **Purpose**: Prepares a list of test videos either by loading a predefined list or randomly selecting a portion of videos.
- **Steps**:
  1. Check for a predefined test video list (`List_of_testing_videos.txt`).
  2. If not found, randomly select test videos based on the `test_size` parameter.

#### **3. Dataset Processing**
- **Method**: `process_dataset`
- **Purpose**: Processes the entire dataset, extracting frames from videos and saving them to the appropriate directories.
- **Steps**:
  1. Load or create a metadata file (`partial_metadata.csv`).
  2. Iterate through directories and videos.
  3. For each video:
     - Extract frames using `process_video`.
     - Perform face detection (if enabled) using `_process_frame_batch`.
     - Save processed frames to the appropriate directories (real/fake).
     - Update metadata and save checkpoints periodically.

#### **4. Frame Processing**
- **Method**: `process_video`
- **Purpose**: Extracts frames from a video and processes them in batches.
- **Steps**:
  1. Open the video file using OpenCV.
  2. Sample frames based on the `frame_sampling_rate`.
  3. Process frames in batches using `_process_frame_batch`.
  4. Save processed frames and update metadata.

- **Method**: `_process_frame_batch`
- **Purpose**: Processes a batch of frames, performing face detection and cropping.
- **Steps**:
  1. Detect faces in the batch using MTCNN.
  2. Crop and resize detected faces.
  3. Save processed frames.

#### **5. Train/Validation Split**
- **Method**: `create_train_val_split`
- **Purpose**: Splits the dataset into train and validation sets.
- **Steps**:
  1. Load the mapping between videos and their frames.
  2. Exclude test videos from the split.
  3. Randomly split the remaining videos into train and validation sets.
  4. Move frames to the appropriate directories (train/val).

#### **6. Saving Metadata and Checkpoints**
- **Method**: `save_checkpoint`
- **Purpose**: Saves the current processing state to a checkpoint file (`processing_checkpoint.json`).
- **Method**: `save_processed_data`
- **Purpose**: Saves processed frames and metadata to disk.

#### **7. Signal Handling**
- **Global Function**: `signal_handler`
- **Purpose**: Handles keyboard interrupts (e.g., `Ctrl+C`) to gracefully stop processing and save progress.

---

### **Key Files and Outputs**
1. **Metadata Files**:
   - `partial_metadata.csv`: Intermediate metadata saved periodically.
   - `train_metadata.csv`: Metadata for the training set.
   - `val_metadata.csv`: Metadata for the validation set.

2. **Processed Frames**:
   - Stored in directories such as `real`, `fake`, `train/real`, `train/fake`, etc.

3. **Checkpoints**:
   - `processing_checkpoint.json`: Stores the list of processed videos and the current directory.

4. **Summary File**:
   - `dataset_summary.json`: Contains statistics about the dataset (e.g., number of frames in each split).

---

### **Example Usage**

```python
if __name__ == "__main__":
    main(
        base_path='Celeb-DF-v2', 
        output_path='output_test_2_image',
        resume=True, 
        output_format='image', 
        test_size=0.1
    )
```

- **Parameters**:
  - `base_path`: Path to the video dataset.
  - `output_path`: Path to save processed data.
  - `resume`: Whether to resume processing from the last checkpoint.
  - `output_format`: Format for saving frames (`npy` or `image`).
  - `test_size`: Proportion of videos to use for the test set.

---

This flowchart and documentation provide a high-level overview of the script's functionality and workflow. Let me know if you need further details!