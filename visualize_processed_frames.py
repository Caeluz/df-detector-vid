import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random


def load_metadata(metadata_path):
    """Load metadata from a CSV file."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def load_frame(frame_path):
    """Load a frame from a .npy file."""
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Frame file not found: {frame_path}")
    return np.load(frame_path)


def plot_frames(frames, titles, rows, cols):
    """Plot multiple frames using matplotlib."""
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # type: ignore

    for i, (frame, title) in enumerate(zip(frames, titles)):
        axes[i].imshow(frame)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def print_metadata_info(metadata):
    """Print metadata information."""
    print("Metadata Head:")
    print(metadata.head())
    print("\nMetadata Shape:")
    print(metadata.shape)
    print("\nUnique Labels:")
    print(metadata['label'].unique())
    print("\nLabel Distribution:")
    print(metadata['label'].value_counts())


def main(metadata_path, num_frames):
    # Load metadata
    metadata = load_metadata(metadata_path)

    # Print metadata information
    print_metadata_info(metadata)

    # Ensure num_frames is within bounds
    num_frames = min(num_frames, len(metadata))

    # Randomly sample frame indices
    sampled_indices = random.sample(range(len(metadata)), num_frames)

    frames = []
    titles = []

    for frame_index in sampled_indices:
        # Get frame path
        frame_path = metadata['frame_path'].values[frame_index]

        # Load frame
        frame = load_frame(frame_path)
        frames.append(frame)
        titles.append(
            f"Frame {frame_index} - Label: {metadata['label'].values[frame_index]}")

        # Print additional metadata for the selected frame
        print("\nSelected Frame Metadata:")
        print(f"Video Path: {metadata['video_path'].values[frame_index]}")
        print(f"Frame Number: {metadata['frame_number'].values[frame_index]}")
        print(f"Frame Path: {metadata['frame_path'].values[frame_index]}")

    # Determine the number of rows and columns for the subplot grid
    cols = 3  # Number of columns in the grid
    # Calculate the number of rows needed
    rows = (num_frames + cols - 1) // cols

    # Plot all frames in a single figure
    plot_frames(frames, titles, rows, cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan and visualize processed frames from metadata.")
    parser.add_argument('--metadata_path', type=str,
                        default='output_face_test_dataset/test_metadata.csv', help='Path to the metadata CSV file.')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to visualize.')

    args = parser.parse_args()
    main(args.metadata_path, args.num_frames)
