import os
import pandas as pd
from sklearn.model_selection import train_test_split
__file__

def create_train_val_split(metadata_path, output_path, val_size=0.2):
    """Create train/validation split from existing metadata"""
    # Load the existing metadata
    metadata_df = pd.read_csv(metadata_path)

    # Get unique video paths
    unique_videos = metadata_df['video_path'].unique()

    # Split into training and validation sets
    train_videos, val_videos = train_test_split(
        unique_videos,
        test_size=val_size,
        random_state=42
    )

    # Create DataFrames for training and validation sets
    train_df = metadata_df[metadata_df['video_path'].isin(train_videos)]
    val_df = metadata_df[metadata_df['video_path'].isin(val_videos)]

    # Save the splits to separate CSV files
    train_df.to_csv(os.path.join(
        output_path, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_metadata.csv'), index=False)

    print(
        f"Training metadata saved to {os.path.join(output_path, 'train_metadata.csv')}")
    print(
        f"Validation metadata saved to {os.path.join(output_path, 'val_metadata.csv')}")


if __name__ == "__main__":
    # Define paths
    # Path to the existing metadata file
    metadata_path = 'output_test_1/metadata.csv'
    output_path = 'output_test_1'  # Output directory

    # Create train/val split
    create_train_val_split(metadata_path, output_path)
