import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import json
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import signal
import sys
import shutil


# Global flag to track if processing should stop
stop_processing = False


def signal_handler(sig, frame):
    """Handle keyboard interrupt to gracefully stop processing"""
    global stop_processing
    print("\nStop command received. Finishing current video and saving progress...")
    stop_processing = True
    # Don't exit immediately - let the code finish gracefully
    return


class VideoDataProcessor:

    def __init__(self,
                 base_path: str,
                 output_path: str,
                 use_face_detection: bool = True,
                 frame_sampling_rate: int = 10,
                 num_workers: Optional[int] = None,
                 output_format: str = 'npy',
                 face_batch_size: int = 32,
                 test_size: float = 0.1):
        """
        Initialize video data processor with configurable parameters

        Args:
            base_path (str): Root directory of video dataset
            output_path (str): Directory to save processed data
            use_face_detection (bool): Whether to detect and crop faces
            frame_sampling_rate (int): Process every nth frame
            num_workers (int): Number of parallel processing workers
            output_format (str): Output format for frames ('npy' or 'image')
            face_batch_size (int): Batch size for face detection
            test_size (float): Proportion of data to use for testing
        """
        self.base_path = base_path
        self.face_batch_size = face_batch_size
        self.output_path = output_path
        self.use_face_detection = use_face_detection
        self.frame_sampling_rate = frame_sampling_rate
        self.num_workers = num_workers or (os.cpu_count() or 1)
        self.output_format = output_format
        self.test_size = test_size

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize face detector
        self.face_detector = self._setup_face_detector() if use_face_detection else None

        # Create output directories
        self._create_output_directories()

    def _setup_face_detector(self):
        """Setup face detection with GPU acceleration if available"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using {device} for face detection")

            return MTCNN(
                margin=20,
                select_largest=True,
                post_process=True,
                device=device,
                keep_all=True,  # Keep all faces for batch processing
                min_face_size=60,  # Lower threshold for faster detection
                factor=0.707  # Default is 0.709, slightly lower for speed
            )
        except Exception as e:
            self.logger.error(f"Face detector setup failed: {e}")
            return None

    def _create_output_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_path, exist_ok=True)

        # Create directories for real and fake images
        self.real_dir = os.path.join(self.output_path, 'real')
        self.fake_dir = os.path.join(self.output_path, 'fake')

        # Create train/val/test split directories
        self.train_real_dir = os.path.join(self.output_path, 'train', 'real')
        self.train_fake_dir = os.path.join(self.output_path, 'train', 'fake')
        self.val_real_dir = os.path.join(self.output_path, 'val', 'real')
        self.val_fake_dir = os.path.join(self.output_path, 'val', 'fake')
        self.test_real_dir = os.path.join(self.output_path, 'test', 'real')
        self.test_fake_dir = os.path.join(self.output_path, 'test', 'fake')

        # Create all directories
        for directory in [self.real_dir, self.fake_dir,
                          self.train_real_dir, self.train_fake_dir,
                          self.val_real_dir, self.val_fake_dir,
                          self.test_real_dir, self.test_fake_dir]:
            os.makedirs(directory, exist_ok=True)

        os.makedirs(os.path.join(self.output_path,
                    'checkpoints'), exist_ok=True)

        # Checkpoint file path
        self.checkpoint_path = os.path.join(
            self.output_path, 'checkpoints', 'processing_checkpoint.json')

        # Create an index file to track frame numbers
        self.index_file = os.path.join(self.output_path, 'frame_index.json')
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w') as f:
                json.dump({
                    'real_index': 0,
                    'fake_index': 0,
                    'video_to_frames': {},  # Maps video path to list of frame paths
                    'test_videos': []       # List of videos assigned to test set
                }, f)

    def get_next_index(self, is_real):
        """Get next index for saving frames"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)

            if is_real:
                index = index_data['real_index']
                index_data['real_index'] += 1
            else:
                index = index_data['fake_index']
                index_data['fake_index'] += 1

            with open(self.index_file, 'w') as f:
                json.dump(index_data, f)

            return index
        else:
            # Initialize with default values if file doesn't exist
            with open(self.index_file, 'w') as f:
                json.dump({
                    'real_index': 1,
                    'fake_index': 1,
                    'video_to_frames': {},
                    'test_videos': []
                }, f)
            return 0

    def update_video_frames_mapping(self, video_path, frame_path):
        """Update the mapping between videos and their frames"""
        with open(self.index_file, 'r') as f:
            index_data = json.load(f)

        if video_path not in index_data['video_to_frames']:
            index_data['video_to_frames'][video_path] = []

        index_data['video_to_frames'][video_path].append(frame_path)

        with open(self.index_file, 'w') as f:
            json.dump(index_data, f)

    def set_test_videos(self, test_videos):
        """Mark specific videos as part of the test set"""
        with open(self.index_file, 'r') as f:
            index_data = json.load(f)

        index_data['test_videos'] = test_videos

        with open(self.index_file, 'w') as f:
            json.dump(index_data, f)

    def save_checkpoint(self, processed_videos, current_dir=None):
        """Save processing checkpoint"""
        checkpoint_data = {
            'processed_videos': processed_videos,
            'current_dir': current_dir
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(self):
        """Load processing checkpoint if exists"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data
        return None

    def load_test_list(self):
        """Load testing videos list if available"""
        test_list_path = os.path.join(
            self.base_path, 'List_of_testing_videos.txt')
        if os.path.exists(test_list_path):
            with open(test_list_path, 'r') as f:
                test_videos = f.read().splitlines()
            return test_videos
        return None

    def process_video(self, video_path: str, label: int) -> List[Dict[str, Any]]:
        """
        Extract frames from video with batch face detection

        Args:
            video_path (str): Path to video file
            label (int): Label for the video (1 for real, 0 for fake)

        Returns:
            List of processed frame dictionaries
        """
        global stop_processing
        frames_data = []
        batch_size = self.face_batch_size
        frame_batch = []
        frame_info_batch = []

        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened() and not stop_processing:
                ret, frame = cap.read()
                if not ret:
                    # Process any remaining frames in the batch
                    if frame_batch:
                        batch_results = self._process_frame_batch(
                            frame_batch, frame_info_batch)
                        frames_data.extend(batch_results)
                    break

                # Sample frames
                if frame_count % self.frame_sampling_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Add to batch
                    frame_batch.append(frame_rgb)
                    frame_info_batch.append({
                        'video_path': video_path,
                        'frame_number': frame_count,
                        'label': label
                    })

                    # Process batch when it reaches the target size
                    if len(frame_batch) >= batch_size:
                        batch_results = self._process_frame_batch(
                            frame_batch, frame_info_batch)
                        frames_data.extend(batch_results)
                        frame_batch = []
                        frame_info_batch = []

                frame_count += 1

            cap.release()

        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")

        return frames_data

    def _process_frame_batch(self, frame_batch, frame_info_batch):
        """Process a batch of frames with face detection"""
        processed_frames = []

        if not self.use_face_detection or not self.face_detector:
            # Simple resize for all frames in batch
            for i, frame in enumerate(frame_batch):
                resized = cv2.resize(frame, (128, 128))
                info = frame_info_batch[i]
                processed_frames.append({
                    'video_path': info['video_path'],
                    'frame_number': info['frame_number'],
                    'label': info['label'],
                    'face_array': resized
                })
            return processed_frames

        try:
            # Detect faces in all frames at once
            batch_boxes, batch_probs, batch_landmarks = self.face_detector.detect(
                frame_batch, landmarks=True)

            # Process each frame with its detection result
            for i, (frame, boxes) in enumerate(zip(frame_batch, batch_boxes)):
                info = frame_info_batch[i]

                if boxes is not None and len(boxes) > 0:
                    # Get the box with highest probability if multiple faces
                    box_idx = 0  # Default to first face
                    if batch_probs[i] is not None and len(batch_probs[i]) > 1:
                        # Use the face with highest probability
                        box_idx = np.argmax(batch_probs[i])

                    x1, y1, x2, y2 = map(int, boxes[box_idx])
                    face = frame[max(0, y1):y2, max(0, x1):x2]

                    # Handle potential empty crop
                    if face.size > 0:
                        face = cv2.resize(face, (128, 128))
                        processed_frames.append({
                            'video_path': info['video_path'],
                            'frame_number': info['frame_number'],
                            'label': info['label'],
                            'face_array': face
                        })
        except Exception as e:
            self.logger.warning(f"Batch face detection failed: {e}")
            # Fall back to individual processing if batch fails
            for i, frame in enumerate(frame_batch):
                try:
                    info = frame_info_batch[i]
                    result = self._extract_frame(frame, info['video_path'],
                                                 info['frame_number'], info['label'])
                    if result:
                        processed_frames.append(result)
                except Exception as inner_e:
                    self.logger.warning(
                        f"Individual frame processing failed: {inner_e}")

        return processed_frames

    def _extract_frame(self, frame_rgb, video_path, frame_count, label):
        """Extract and process a single frame"""
        if self.use_face_detection and self.face_detector:
            try:
                detection_result = self.face_detector.detect(frame_rgb)

                # Handle different MTCNN detection output formats
                boxes = detection_result[0] if len(
                    detection_result) == 3 else detection_result[0]

                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    face = frame_rgb[y1:y2, x1:x2]
                    face = cv2.resize(face, (128, 128))
                else:
                    return None
            except Exception as e:
                self.logger.warning(
                    f"Face detection failed for frame {frame_count}: {e}")
                return None
        else:
            # Resize entire frame if no face detection
            face = cv2.resize(frame_rgb, (128, 128))

        return {
            'video_path': video_path,
            'frame_number': frame_count,
            'label': label,
            'face_array': face
        }

    def prepare_test_videos(self, video_dirs: Dict[str, int]):
        """
        Prepare test videos either by loading from a predefined list
        or by randomly selecting a portion of videos

        Args:
            video_dirs (dict): Directories with label mapping

        Returns:
            list: List of video paths selected for testing
        """
        # First check if there's a predefined list
        predefined_test_list = self.load_test_list()
        if predefined_test_list:
            self.logger.info(
                f"Found predefined test list with {len(predefined_test_list)} videos")
            # Convert relative paths to absolute paths
            test_videos = []
            for rel_path in predefined_test_list:
                # Try to find matching video in any directory
                found = False
                for dir_name in video_dirs.keys():
                    potential_path = os.path.join(
                        self.base_path, dir_name, rel_path)
                    if os.path.exists(potential_path):
                        test_videos.append(potential_path)
                        found = True
                        break
                if not found:
                    self.logger.warning(
                        f"Could not find test video: {rel_path}")
            return test_videos

        # If no predefined list, randomly select videos
        all_videos = []
        for dir_name, label in video_dirs.items():
            dir_path = os.path.join(self.base_path, dir_name)
            if not os.path.isdir(dir_path):
                continue

            videos = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                      if f.endswith(('.mp4', '.avi'))]
            all_videos.extend(videos)

        # Randomly select test videos
        _, test_videos = train_test_split(
            all_videos, test_size=self.test_size, random_state=42)

        self.logger.info(f"Selected {len(test_videos)} videos for testing")
        return test_videos

    def process_dataset(self, video_dirs: Dict[str, int], resume: bool = False):
        """
        Process entire dataset with checkpoint support, separating test videos

        Args:
            video_dirs (dict): Directories with label mapping 
                              (1 for real videos, 0 for fake videos)
            resume (bool): Whether to resume from previous checkpoint

        Returns:
            Dictionary mapping video paths to frame counts
        """
        global stop_processing
        processed_videos = set()
        current_dir = None
        video_frame_counts = {}

        # Prepare test videos
        test_videos = []
        if not resume:
            test_videos = self.prepare_test_videos(video_dirs)
            self.set_test_videos(test_videos)
        else:
            # Load test videos from index file
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
                test_videos = index_data.get('test_videos', [])

        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                processed_videos = set(checkpoint['processed_videos'])
                current_dir = checkpoint['current_dir']
                self.logger.info(
                    f"Resuming from checkpoint. Processed {len(processed_videos)} videos.")

        # Process videos
        for dir_name, label in video_dirs.items():
            # Skip directories until we reach the one we were processing
            if current_dir and dir_name != current_dir:
                continue
            current_dir = None  # Reset after finding the resume point

            dir_path = os.path.join(self.base_path, dir_name)
            if not os.path.isdir(dir_path):
                self.logger.warning(
                    f"Directory not found: {dir_path}, skipping")
                continue

            videos = [f for f in os.listdir(
                dir_path) if f.endswith(('.mp4', '.avi'))]
            self.logger.info(f"Found {len(videos)} videos in {dir_name}")

            for video_file in tqdm(videos, desc=f"Processing {dir_name}"):
                # Check if stop flag has been set
                if stop_processing:
                    self.logger.info(
                        "Stop command received. Saving progress...")
                    self.save_checkpoint(list(processed_videos), dir_name)
                    return video_frame_counts

                video_path = os.path.join(dir_path, video_file)

                # Skip if already processed
                if video_path in processed_videos:
                    continue

                # Process video
                frames_data = self.process_video(video_path, label)
                video_frame_counts[video_path] = len(frames_data)

                # Save processed frames directly to real/fake folders
                for frame_data in frames_data:
                    self._save_frame_to_category(
                        frame_data, video_path in test_videos)

                # Mark video as processed
                processed_videos.add(video_path)

                # Save checkpoint after each video
                self.save_checkpoint(list(processed_videos), dir_name)

        return video_frame_counts

    def _save_frame_to_category(self, frame_data: Dict[str, Any], is_test_video: bool = False) -> str:
        """
        Save a single frame to the appropriate real/fake folder

        Args:
            frame_data (dict): Dictionary containing frame data
            is_test_video (bool): Whether this frame belongs to a test video

        Returns:
            str: Path to the saved frame
        """
        is_real = frame_data['label'] == 1
        dest_dir = self.real_dir if is_real else self.fake_dir

        # Get next index for this category
        index = self.get_next_index(is_real)
        category = "real" if is_real else "fake"

        # Create a unique filename using video name and frame number
        video_basename = os.path.basename(
            frame_data['video_path']).split('.')[0]
        filename = f"{category}_{video_basename}_{frame_data['frame_number']}_{index}"

        # Save to appropriate directory
        if self.output_format == 'npy':
            frame_path = os.path.join(dest_dir, f"{filename}.npy")
            np.save(frame_path, frame_data['face_array'])
        else:
            frame_path = os.path.join(dest_dir, f"{filename}.jpg")
            face_image = Image.fromarray(frame_data['face_array'])
            face_image.save(frame_path)

        # Save directly to test folder if it's a test video
        if is_test_video:
            if is_real:
                test_path = os.path.join(
                    self.test_real_dir, os.path.basename(frame_path))
            else:
                test_path = os.path.join(
                    self.test_fake_dir, os.path.basename(frame_path))

            if self.output_format == 'npy':
                np.save(test_path, frame_data['face_array'])
            else:
                face_image = Image.fromarray(frame_data['face_array'])
                face_image.save(test_path)

        # Update mapping between videos and their frames
        self.update_video_frames_mapping(frame_data['video_path'], frame_path)

        return frame_path

    def create_train_val_split(self, val_size=0.2):
        """
        Create train/validation split by moving files from real/fake folders
        to train/val subfolders (excluding test videos which are already separated)
        """
        # Load the mapping between videos and their frames
        with open(self.index_file, 'r') as f:
            index_data = json.load(f)

        video_to_frames = index_data['video_to_frames']
        test_videos = set(index_data.get('test_videos', []))

        # Get unique videos (excluding test videos)
        unique_videos = [v for v in list(
            video_to_frames.keys()) if v not in test_videos]

        # Split remaining videos into train and validation sets
        train_videos, val_videos = train_test_split(
            unique_videos, test_size=val_size, random_state=42)

        # Move frames to appropriate folders
        print("Creating train/val split...")
        for video_path in tqdm(train_videos, desc="Processing train videos"):
            for frame_path in video_to_frames[video_path]:
                if os.path.exists(frame_path):
                    # Determine if real or fake
                    is_real = "real" in os.path.basename(frame_path).lower()
                    dest_dir = self.train_real_dir if is_real else self.train_fake_dir

                    # Copy file to train folder
                    dest_path = os.path.join(
                        dest_dir, os.path.basename(frame_path))
                    if not os.path.exists(dest_path):  # Avoid duplicate copies
                        shutil.copy2(frame_path, dest_path)

        for video_path in tqdm(val_videos, desc="Processing validation videos"):
            for frame_path in video_to_frames[video_path]:
                if os.path.exists(frame_path):
                    # Determine if real or fake
                    is_real = "real" in os.path.basename(frame_path).lower()
                    dest_dir = self.val_real_dir if is_real else self.val_fake_dir

                    # Copy file to validation folder
                    dest_path = os.path.join(
                        dest_dir, os.path.basename(frame_path))
                    if not os.path.exists(dest_path):  # Avoid duplicate copies
                        shutil.copy2(frame_path, dest_path)

        # Create summary file
        summary = {
            'train': {
                'real': len(os.listdir(self.train_real_dir)),
                'fake': len(os.listdir(self.train_fake_dir))
            },
            'val': {
                'real': len(os.listdir(self.val_real_dir)),
                'fake': len(os.listdir(self.val_fake_dir))
            },
            'test': {
                'real': len(os.listdir(self.test_real_dir)),
                'fake': len(os.listdir(self.test_fake_dir))
            },
            'test_videos_count': len(test_videos)
        }

        with open(os.path.join(self.output_path, 'dataset_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        return summary


def main(base_path: str, output_path: str, resume: bool = False, output_format: str = 'npy', test_size: float = 0.1):
    """Main processing pipeline"""
    global stop_processing

    # Set up signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    print("Processing started. Press Ctrl+C to stop processing and save progress.")

    video_dirs = {
        'Celeb-real': 1,         # Real videos (label=1)
        'Celeb-synthesis': 0,    # Fake videos (label=0)
        'YouTube-real': 1,       # Real videos (label=1)
        'face-forensics-youtube-real': 1,  # Real videos (label=1)
    }

    processor = VideoDataProcessor(
        base_path=base_path,            # Root directory of video dataset
        output_path=output_path,        # Output directory
        use_face_detection=True,        # Enable face detection
        frame_sampling_rate=15,         # Configurable sampling
        num_workers=None,               # Auto-detect cores
        output_format=output_format,    # Output format ('npy' or 'image')
        test_size=test_size             # Proportion of data for test set
    )

    try:
        # Process dataset with resume option
        video_frame_counts = processor.process_dataset(
            video_dirs, resume=resume)

        # Only create train/val splits if processing completed successfully
        if not stop_processing:
            # Create train/val splits
            split_summary = processor.create_train_val_split(val_size=0.2)

            print("\nDataset processing complete!")
            print(
                f"Train set: {split_summary['train']['real']} real, {split_summary['train']['fake']} fake images")
            print(
                f"Validation set: {split_summary['val']['real']} real, {split_summary['val']['fake']} fake images")
            print(
                f"Test set: {split_summary['test']['real']} real, {split_summary['test']['fake']} fake images")
            print(
                f"(Based on {split_summary['test_videos_count']} unique test videos)")
        else:
            print("Processing stopped. Progress saved in checkpoint file.")

    except Exception as e:
        print(f"Error during processing: {e}")
        # Save current progress on error
        print("Saving progress before exit...")


if __name__ == "__main__":
    # Reset the stop flag at the start
    stop_processing = False

    main('Celeb-DF-v2', 'output_test_2_image',
         resume=True, output_format='image', test_size=0.1)
