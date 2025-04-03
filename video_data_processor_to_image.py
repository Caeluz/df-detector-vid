import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import random
import json
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Any, Optional
from PIL import Image


class VideoDataProcessor:
    def __init__(self,
                 base_path: str,
                 output_path: str,
                 use_face_detection: bool = True,
                 frame_sampling_rate: int = 10,
                 num_workers: Optional[int] = None):
        """
        Initialize video data processor with configurable parameters

        Args:
            base_path (str): Root directory of video dataset
            output_path (str): Directory to save processed data
            use_face_detection (bool): Whether to detect and crop faces
            frame_sampling_rate (int): Process every nth frame
            num_workers (int): Number of parallel processing workers
        """
        self.base_path = base_path
        self.output_path = output_path
        self.use_face_detection = use_face_detection
        self.frame_sampling_rate = frame_sampling_rate
        self.num_workers = num_workers or (os.cpu_count() or 1)

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
            return MTCNN(
                margin=20,
                select_largest=True,
                post_process=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            self.logger.error(f"Face detector setup failed: {e}")
            return None

    def _create_output_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path,
                    'processed_images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path,
                    'checkpoints'), exist_ok=True)

        # Checkpoint file path
        self.checkpoint_path = os.path.join(
            self.output_path, 'checkpoints', 'processing_checkpoint.json')

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
        """Load testing videos list"""
        test_list_path = os.path.join(
            self.base_path, 'List_of_testing_videos.txt')
        with open(test_list_path, 'r') as f:
            test_videos = f.read().splitlines()
        return test_videos

    def process_image(self, image_path: str, label: int) -> Dict[str, Any]:
        """
        Process a single image with optional face detection

        Args:
            image_path (str): Path to image file
            label (int): Label for the image

        Returns:
            Processed image dictionary
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            if self.use_face_detection and self.face_detector:
                detection_result = self.face_detector.detect(image_np)

                if detection_result[0] is not None:
                    x1, y1, x2, y2 = map(int, detection_result[0][0])
                    face = image_np[y1:y2, x1:x2]
                    face = cv2.resize(face, (128, 128))
                else:
                    return None
            else:
                face = cv2.resize(image_np, (128, 128))

            return {
                'image_path': image_path,
                'label': label,
                'face_array': face
            }
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None

    def process_dataset(self, image_dirs: Dict[str, int], resume: bool = False) -> List[Dict[str, Any]]:
        """
        Process entire dataset with checkpoint support

        Args:
            image_dirs (dict): Directories with label mapping
            resume (bool): Whether to resume from previous checkpoint

        Returns:
            List of processed image data
        """
        all_data = []
        processed_images = set()
        current_dir = None

        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                processed_images = set(checkpoint['processed_videos'])
                current_dir = checkpoint['current_dir']
                self.logger.info(
                    f"Resuming from checkpoint. Processed {len(processed_images)} images.")

                # Load previously processed data from metadata
                metadata_path = os.path.join(
                    self.output_path, 'partial_metadata.csv')
                if os.path.exists(metadata_path):
                    partial_df = pd.read_csv(metadata_path)
                    for _, row in partial_df.iterrows():
                        face_array = np.load(row['frame_path'])
                        image_data = {
                            'image_path': row['image_path'],
                            'label': row['label'],
                            'face_array': face_array
                        }
                        all_data.append(image_data)

        # Process images
        for dir_name, label in image_dirs.items():
            # Skip directories until we reach the one we were processing
            if current_dir and dir_name != current_dir:
                continue
            current_dir = None  # Reset after finding the resume point

            dir_path = os.path.join(self.base_path, dir_name)
            images = [f for f in os.listdir(
                dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

            for image_file in images:
                image_path = os.path.join(dir_path, image_file)

                # Skip if already processed
                if image_path in processed_images:
                    continue

                # Process image
                image_data = self.process_image(image_path, label)
                if image_data:
                    all_data.append(image_data)
                    processed_images.add(image_path)

                # Save checkpoint after each image
                self.save_checkpoint(list(processed_images), dir_name)

                # Save partial results periodically
                if len(all_data) % 1000 == 0:
                    self.save_processed_data(
                        all_data, metadata_filename='partial_metadata.csv')

        return all_data

    def save_processed_data(self, processed_data, metadata_filename='metadata.csv'):
        """Save processed data to disk"""
        metadata = []
        for i, data in enumerate(processed_data):
            image_path = os.path.join(
                self.output_path, 'processed_images', f'image_{i}.jpg')

            # Save face array as image
            face_image = Image.fromarray(data['face_array'])
            face_image.save(image_path)

            # Add to metadata
            metadata.append({
                'image_path': image_path,
                'original_image_path': data['image_path'],
                'label': data['label']
            })

        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(
            self.output_path, metadata_filename), index=False)

        return metadata_df

    def create_train_val_split(self, metadata_df, val_size=0.2):
        """Create train/validation split"""
        unique_images = metadata_df['original_image_path'].unique()

        train_images, val_images = train_test_split(
            unique_images, test_size=val_size, random_state=42)

        train_df = metadata_df[metadata_df['original_image_path'].isin(
            train_images)]
        val_df = metadata_df[metadata_df['original_image_path'].isin(
            val_images)]

        return train_df, val_df

# Example usage:


def main(base_path: str, output_path: str, resume: bool = False):
    """Main processing pipeline"""
    image_dirs = {
        'Celeb-real': 1,
        'Celeb-synthesis': 0,
        'YouTube-real': 1,
        'face-forensics-youtube-real': 1, # New dataset
    }

    processor = VideoDataProcessor(
        base_path=base_path,  # Root directory of image dataset
        output_path=output_path,  # Output directory
        use_face_detection=True,  # Enable face detection
        frame_sampling_rate=15,  # Configurable sampling
        num_workers=None  # Auto-detect cores
    )

    # Process dataset with resume option
    processed_data = processor.process_dataset(image_dirs, resume=resume)
    metadata_df = processor.save_processed_data(processed_data)

    train_df, val_df = processor.create_train_val_split(metadata_df)

    # Save train/val splits
    train_df.to_csv(os.path.join(
        output_path, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_metadata.csv'), index=False)


if __name__ == "__main__":
    main('Celeb-DF-v2', 'output_test_2_image', resume=True)
