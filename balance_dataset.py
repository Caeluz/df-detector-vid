import os
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def check_dataset_balance(metadata_path):
    df = pd.read_csv(metadata_path)
    real_count = df[df['label'] == 1].shape[0]
    fake_count = df[df['label'] == 0].shape[0]
    print(f"Real video frames: {real_count}, Fake video frames: {fake_count}")


def balance_dataset(metadata_path, output_path, method='oversample'):
    df = pd.read_csv(metadata_path)
    real_videos = df[df['label'] == 1]
    fake_videos = df[df['label'] == 0]

    print(
        f"Before balancing: Real: {len(real_videos)}, Fake: {len(fake_videos)}")

    if method == 'oversample':
        # Oversample real videos
        real_videos_oversampled = resample(real_videos,
                                           replace=True,
                                           n_samples=len(fake_videos),
                                           random_state=42)
        balanced_df = pd.concat([real_videos_oversampled, fake_videos])
    elif method == 'undersample':
        # Undersample fake videos
        fake_videos_undersampled = resample(fake_videos,
                                            replace=False,
                                            n_samples=len(real_videos),
                                            random_state=42)
        balanced_df = pd.concat([real_videos, fake_videos_undersampled])
    else:
        raise ValueError("Method should be 'oversample' or 'undersample'")

    print(
        f"After balancing: Real: {len(balanced_df[balanced_df['label'] == 1])}, Fake: {len(balanced_df[balanced_df['label'] == 0])}")

    balanced_df = balanced_df.sample(
        frac=1, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(output_path, index=False)
    print(f"Balanced dataset saved to {output_path}")


def split_dataset(metadata_path, train_output_path, val_output_path, val_size=0.2):
    df = pd.read_csv(metadata_path)
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=42, stratify=df['label'])
    train_df.to_csv(train_output_path, index=False)
    val_df.to_csv(val_output_path, index=False)
    print(f"Training dataset saved to {train_output_path}")
    print(f"Validation dataset saved to {val_output_path}")


# check_dataset_balance('output_test_1/balanced_metadata.csv')
check_dataset_balance('output_test_1_image/new_train_metadata.csv')
check_dataset_balance('output_test_1_image/new_val_metadata.csv')

balance_dataset('output_test_1_image/new_train_metadata.csv',
                'output_test_1_image/balanced_new_train_metadata.csv', method='undersample')

check_dataset_balance('output/balanced_metadata.csv')

# split_dataset('output_test_1_image/balanced_metadata.csv',
#               'output_test_1_image/balanced_train_metadata.csv', 'output_test_1_image/balanced_val_metadata.csv', val_size=0.2)
