import torch
from torch import nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
from collections import defaultdict
import os
from tqdm import tqdm
from step_04_deepfake_data import DeepFakeDetector
from facenet_pytorch import MTCNN

import matplotlib.pyplot as plt


class ImprovedDeepFakeVideoPredictor:
    def __init__(self,
                 model_path: str,
                 use_face_detection: bool = True,
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the improved DeepFake video predictor with a trained model.
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = DeepFakeDetector()
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.use_face_detection = use_face_detection

        # Initialize MTCNN with parameters matching training
        if self.use_face_detection:
            self.face_detector = MTCNN(
                image_size=128,  # Match training size
                margin=20,  # No margin to match training
                min_face_size=60,
                thresholds=[0.6, 0.7, 0.9],
                factor=0.709,
                post_process=True,
                device=self.device,
            )

        # Match the exact preprocessing used in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def preprocess_frame(self, frame: np.ndarray) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        """
        Preprocess a single frame for inference, matching training preprocessing.
        """
        if self.use_face_detection:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect and align face
            face_tensor = self.face_detector(frame_rgb)

            if face_tensor is None:
                # If no face detected, return None
                return None

            # Normalize to [0, 1] range
            face_tensor = face_tensor.float() / 255.0
            return face_tensor, frame_rgb
        else:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to match training size
            frame_resized = cv2.resize(frame_rgb, (128, 128))
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(
                frame_resized).float().permute(2, 0, 1) / 255.0
            return frame_tensor, frame_rgb

    def predict_video(self,
                      video_path: str,
                      sample_rate: int = 1,
                      threshold: float = 0.5,
                      temperature: float = 2.0) -> Dict:  # Added temperature parameter
        """
        Predict whether a video is real or fake using improved processing.
        """
        cap = cv2.VideoCapture(video_path)
        real_probs = []
        fake_probs = []
        valid_frames = 0
        total_frames = 0
        processed_frames = []

        with tqdm(desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if total_frames % sample_rate == 0:
                    result = self.preprocess_frame(frame)
                    if result is not None:
                        processed_frame, frame_rgb = result
                    else:
                        processed_frame, frame_rgb = None, None

                    if processed_frame is not None:
                        processed_frame = processed_frame.unsqueeze(
                            0).to(self.device)

                        with torch.no_grad():
                            # Apply temperature scaling
                            output = self.model(
                                processed_frame, temperature=temperature)
                            probabilities = torch.softmax(output, dim=1)

                            real_prob = probabilities[0][0].item()
                            fake_prob = probabilities[0][1].item()
                            real_probs.append(real_prob)
                            fake_probs.append(fake_prob)
                            valid_frames += 1

                    # Collect the processed frame
                    processed_frames.append(frame_rgb)

                total_frames += 1
                pbar.update(1)

        cap.release()

        if valid_frames == 0:
            return {
                'error': 'No valid frames processed',
                'prediction': 'UNKNOWN',
                'confidence': 0.0
            }

        # Calculate metrics
        avg_real_prob = np.mean(real_probs)
        avg_fake_prob = np.mean(fake_probs)
        std_real = np.std(real_probs)
        std_fake = np.std(fake_probs)

        # More nuanced prediction logic
        if avg_fake_prob > threshold:
            prediction = 'FAKE'
            confidence = avg_fake_prob
        else:
            prediction = 'REAL'
            confidence = avg_real_prob

        # Adjust confidence based on prediction consistency
        confidence = confidence * (1 - (std_real + std_fake) / 2)

        results = {
            'prediction': prediction,
            'confidence': confidence,
            'fake_ratio': avg_fake_prob,
            'total_frames': total_frames,
            'valid_frames': valid_frames,
            'avg_real_prob': avg_real_prob,
            'avg_fake_prob': avg_fake_prob,
            'std_real': std_real,
            'std_fake': std_fake
        }

        # self.visualize_frames(processed_frames)

        return results


    def visualize_frames(self, frames: List[np.ndarray], grid_size: int = 4):
        """
        Visualize the processed frames using Matplotlib.
        """
        num_frames = len(frames)
        num_plots = grid_size * grid_size

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 8))
        axes = axes.flatten()

        for ax in axes:
            ax.axis('off')

        for i, frame in enumerate(frames):
            ax = axes[i % num_plots]
            ax.imshow(frame)
            ax.axis('off')

            if (i + 1) % num_plots == 0 or i == num_frames - 1:
                plt.tight_layout()
                # Use pause instead of show to keep the figure open
                plt.pause(0.001)
                if i != num_frames - 1:
                    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 8))
                    axes = axes.flatten()
                    for ax in axes:
                        ax.axis('off')


# Example usage
if __name__ == "__main__":
    predictor = ImprovedDeepFakeVideoPredictor(
        model_path='training_output_2/checkpoints/best_model.pth',
        use_face_detection=False
    )

    # video_path = r"path/to/your/video.mp4"
    # List of video paths to process
    video_paths = [
    ]
    for video_path in video_paths:
        results = predictor.predict_video(
            video_path=video_path,
            sample_rate=2,
            threshold=0.5
        )

        print(f"Video: {video_path}")
        print(f"\nVideo Analysis Results:")
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Fake ratio: {results['fake_ratio']:.2%}")
        print(
            f"Valid frames: {results['valid_frames']} / {results['total_frames']}")
        print('Average real probability:', results['avg_real_prob'])
        print('Average fake probability:', results['avg_fake_prob'])
        print('Real probability standard deviation:', results['std_real'])
        print('Fake probability standard deviation:', results['std_fake'])
