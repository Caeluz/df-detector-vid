import torch
from torch import nn
import numpy as np
import cv2
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
from collections import defaultdict
import os
from tqdm import tqdm
from step_04_deepfake_data import DeepFakeDetector
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt


class ImprovedDeepFakePredictor:
    def __init__(self,
                 model_path: str,
                 use_face_detection: bool = True,
                 use_face_only: bool = True,
                 device: Optional[str] = None,
                 visualize_frames_bool: bool = False):
        """
        Initialize the improved DeepFake predictor with a trained model.
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

        self.use_face_detection = use_face_detection
        self.use_face_only = use_face_only

        # For visualization
        self.visualize_frames_bool = visualize_frames_bool

        # Initialize MTCNN with parameters matching training
        if self.use_face_detection:
            self.face_detector = MTCNN(
                image_size=128,  # Match training size
                margin=20,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                keep_all=False,
                device=self.device,
            )

        # Match the exact preprocessing used in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def preprocess_frame(self, frame: np.ndarray) -> Optional[Tuple[Optional[torch.Tensor], np.ndarray]]:
        """
        Preprocess a frame (from video or image) for deepfake detection.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_face_detection:
            try:
                # Detect faces
                boxes, _ = self.face_detector.detect(frame_rgb)  # type: ignore
            except Exception:
                return None, frame_rgb

            if boxes is not None and len(boxes) > 0:
                # Extract the first detected face
                x1, y1, x2, y2 = map(int, boxes[0])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(
                    x2, frame_rgb.shape[1]), min(y2, frame_rgb.shape[0])

                try:
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    face_resized = cv2.resize(
                        face_crop, (128, 128), interpolation=cv2.INTER_AREA)
                    face_tensor = torch.from_numpy(
                        face_resized).permute(2, 0, 1).float() / 255.0

                    if self.use_face_only:
                        return face_tensor, frame_rgb
                    else:
                        frame_tensor = torch.from_numpy(
                            frame_rgb).permute(2, 0, 1).float() / 255.0
                        return frame_tensor, frame_rgb
                except Exception:
                    return None, frame_rgb

            return None, frame_rgb
        else:
            # Non-face detection path
            frame_resized = cv2.resize(
                frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            frame_tensor = torch.from_numpy(
                frame_resized).permute(2, 0, 1).float() / 255.0
            return frame_tensor, frame_rgb

    def predict_image(self, image_path: str, threshold: float = 0.5, temperature: float = 2.0) -> Dict:
        """
        Predict whether an image is real or fake.
        """
        # Load the image
        if not os.path.exists(image_path):
            return {
                'error': 'File not found',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': f'Image file not found: {image_path}'
            }

        frame = cv2.imread(image_path)
        if frame is None:
            return {
                'error': 'Invalid image',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': f'Could not read image: {image_path}'
            }

        # Preprocess the image
        result = self.preprocess_frame(frame)
        if result is None:
            return {
                'error': 'No valid image processed',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': 'No faces detected in the image.'
            }

        processed_tensor, frame_rgb = result
        if processed_tensor is None:
            return {
                'error': 'No valid image processed',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': 'No faces detected in the image.'
            }

        # Perform inference
        processed_tensor = processed_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(processed_tensor, temperature=temperature)
            probabilities = torch.softmax(output, dim=1)

            # Index 0 is fake (0), index 1 is real (1)
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()

        # Determine prediction based on real probability threshold
        if real_prob > threshold:
            prediction = 'REAL'
            confidence = real_prob
        else:
            prediction = 'FAKE'
            confidence = fake_prob

        results = {
            'prediction': prediction,
            'confidence': confidence,
            'fake_ratio': fake_prob,
            'real_prob': real_prob,
        }

        # Visualize heatmap for the image
        if self.visualize_frames_bool:
            self.visualize_heatmap(processed_tensor.squeeze(0), self.model)

        return results

    def predict_video(self, video_path: str, sample_rate: int = 1, threshold: float = 0.5, temperature: float = 2.0) -> Dict:
        """
        Predict whether a video is real or fake.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'error': 'Invalid video',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': f'Could not open video: {video_path}'
            }

        real_probs = []
        fake_probs = []
        valid_frames = 0
        total_frames = 0
        last_frame_tensor = None  # Store the last frame tensor for heatmap

        with tqdm(desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if total_frames % sample_rate == 0:
                    result = self.preprocess_frame(frame)
                    if result is not None:
                        processed_tensor, frame_rgb = result
                    else:
                        processed_tensor, frame_rgb = None, None

                    if processed_tensor is not None:
                        processed_tensor = processed_tensor.unsqueeze(
                            0).to(self.device)

                        with torch.no_grad():
                            # Apply temperature scaling
                            output = self.model(
                                processed_tensor, temperature=temperature)
                            probabilities = torch.softmax(output, dim=1)

                            # Index 0 is fake (0), index 1 is real (1)
                            fake_prob = probabilities[0][0].item()
                            real_prob = probabilities[0][1].item()
                            real_probs.append(real_prob)
                            fake_probs.append(fake_prob)
                            valid_frames += 1

                        # Store the last frame tensor for heatmap
                        last_frame_tensor = processed_tensor

                total_frames += 1
                pbar.update(1)

        cap.release()

        if valid_frames == 0:
            return {
                'error': 'No valid frames processed',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': 'No faces detected in the video.'
            }

        # Calculate metrics
        avg_real_prob = np.mean(real_probs)
        avg_fake_prob = np.mean(fake_probs)
        std_real = np.std(real_probs)
        std_fake = np.std(fake_probs)

        # Prediction logic based on real probability
        if avg_real_prob > threshold:
            prediction = 'REAL'
            confidence = avg_real_prob
        else:
            prediction = 'FAKE'
            confidence = avg_fake_prob

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
            'std_fake': std_fake,
        }

        # Visualize heatmap for the last frame
        if last_frame_tensor is not None and self.visualize_frames_bool:
            self.visualize_heatmap(last_frame_tensor.squeeze(0), self.model)

        return results

    def visualize_heatmap(self, face_tensor: torch.Tensor, model: nn.Module, save_path: Optional[str] = None):
        """
        Generate a class activation heatmap to show which regions contribute most to the prediction.
        """
        # Ensure the model is in evaluation mode
        model.eval()

        # Hook into the last convolutional layer
        # Assuming the last conv layer is used
        target_layer = model.features[-1]
        activations = None
        gradients = None

        def hook_fn(module, input, output):
            nonlocal activations
            activations = output

        def backward_hook_fn(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        # Register hooks
        hook = target_layer.register_forward_hook(hook_fn)
        backward_hook = target_layer.register_backward_hook(backward_hook_fn)

        # Forward pass
        outputs = model(face_tensor.unsqueeze(0))
        prediction = torch.argmax(outputs, dim=1)

        # Backward pass
        model.zero_grad()
        outputs[0, prediction].backward()

        # Remove hooks
        hook.remove()
        backward_hook.remove()

        # Compute Grad-CAM
        if activations is not None and gradients is not None:
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(
                activations, dim=1).squeeze().detach().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)  # ReLU
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min())  # Normalize
            # Resize to match input size
            heatmap = cv2.resize(heatmap, (128, 128))
            heatmap = np.uint8(255 * heatmap)  # Convert to 8-bit
            heatmap = cv2.applyColorMap(
                heatmap, cv2.COLORMAP_JET)  # Apply colormap

            # Overlay heatmap on original image
            original_img = (face_tensor.permute(
                1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            output_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

            # Plot results
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(original_img)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Activation Heatmap')
            plt.imshow(heatmap)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Overlaid Heatmap')
            plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            if save_path:
                plt.savefig(save_path)
            plt.show()


# Example usage
if __name__ == "__main__":
    predictor = ImprovedDeepFakePredictor(
        # model_path='training_output_test_2_image/checkpoints/best_model.pth',
        model_path='training_output_test_2_image/checkpoints/checkpoint_epoch_8.pth',
        use_face_detection=True,
        use_face_only=True,
        visualize_frames_bool=True,
    )

    # Predict for an image
    image_path = r"C:\Users\aaron\Pictures\Screenshots\Screenshot 2024-10-16 111520.png"
    image_results = predictor.predict_image(image_path)
    print("Image Results:", image_results)

    # Predict for a video
    # video_path = r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker 4_1728357437.mp4"
    # video_results = predictor.predict_video(video_path)
    # print("Video Results:", video_results)
