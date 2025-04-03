import csv
import os
from pathlib import Path


def split_metadata(train_csv_path, list_of_testing_videos_path, output_train_csv_path, output_test_csv_path):
    """
    Split train_metadata.csv into train and test CSVs based on testing videos list

    Parameters:
    train_csv_path (str): Path to the original train_metadata.csv
    list_of_testing_videos_path (str): Path to the list_of_testing_videos.txt
    output_train_csv_path (str): Path to save the new train_metadata.csv
    output_test_csv_path (str): Path to save the test_metadata.csv
    """
    # Read testing videos list
    testing_videos = []
    with open(list_of_testing_videos_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(' ', 1)
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue

            label, video_path = parts
            # Remove file system prefix and normalize slashes
            video_path = video_path.replace('\\', '/').lower()
            testing_videos.append((label, video_path))

    print(f"Loaded {len(testing_videos)} testing videos")

    # Process testing videos to create lookup dictionary by filename
    # This helps match regardless of directory structure
    test_video_lookup = {}
    for label, path in testing_videos:
        filename = os.path.basename(path)
        # Store both the filename and the path to handle potential filename duplicates
        if filename not in test_video_lookup:
            test_video_lookup[filename] = []
        test_video_lookup[filename].append((label, path))

    # Track which videos we find and process
    found_testing_videos = set()
    test_rows = []
    train_rows = []
    headers = None

    # Process the train_metadata.csv
    with open(train_csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Get headers

        for row in reader:
            if len(row) < 4:
                print(f"Skipping row with insufficient data: {row}")
                continue

            frame_path, video_path, frame_number, label = row

            # Normalize the path for comparison
            normalized_path = video_path.replace('\\', '/').lower()
            filename = os.path.basename(normalized_path)

            # Check if this file is in our test set
            is_test_video = False

            # First try exact match after base directory
            for test_label, test_path in testing_videos:
                if test_path.lower() in normalized_path:
                    found_testing_videos.add((test_label, test_path))
                    is_test_video = True
                    break

            # If no exact match, try matching by filename
            if not is_test_video and filename in test_video_lookup:
                for test_label, test_path in test_video_lookup[filename]:
                    # Check if the filename and parts of the path match
                    test_path_parts = test_path.split('/')
                    normalized_path_parts = normalized_path.split('/')

                    # Check if key parts of the path match
                    for part in test_path_parts:
                        if part.lower() in normalized_path.lower():
                            found_testing_videos.add((test_label, test_path))
                            is_test_video = True
                            break

                    if is_test_video:
                        break

            if is_test_video:
                test_rows.append(row)
            else:
                train_rows.append(row)

    # If no matches found, try more aggressive matching
    if not found_testing_videos:
        print("No matches found with standard approach. Trying more aggressive matching...")

        # Reprocess with more aggressive matching
        test_rows = []
        train_rows = []

        with open(train_csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 4:
                    continue

                frame_path, video_path, frame_number, label = row
                normalized_path = video_path.replace('\\', '/').lower()
                filename = os.path.basename(normalized_path)

                # Look for any part of test path in video path
                is_test_video = False
                for test_label, test_path in testing_videos:
                    # Extract video ID from test path (e.g., "00170" from "YouTube-real/00170.mp4")
                    test_parts = test_path.split('/')
                    test_filename = test_parts[-1]  # e.g., "00170.mp4"
                    test_type = test_parts[0] if len(
                        test_parts) > 1 else ""  # e.g., "YouTube-real"

                    # Check if both filename and type match
                    if test_filename.lower() in normalized_path.lower() and test_type.lower() in normalized_path.lower():
                        found_testing_videos.add((test_label, test_path))
                        is_test_video = True
                        break

                if is_test_video:
                    test_rows.append(row)
                else:
                    train_rows.append(row)

    # Write the new train_metadata.csv
    with open(output_train_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(train_rows)

    # Write the test_metadata.csv
    with open(output_test_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(test_rows)

    # Report statistics
    print(
        f"Original train_metadata entries: {len(train_rows) + len(test_rows)}")
    print(f"New train_metadata entries: {len(train_rows)}")
    print(f"Test_metadata entries: {len(test_rows)}")
    print(
        f"Found {len(found_testing_videos)} out of {len(testing_videos)} testing videos")

    # Report missing testing videos
    found_testing_paths = {path for _, path in found_testing_videos}
    missing_count = 0
    missing_examples = []

    for label, path in testing_videos:
        if path not in found_testing_paths:
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append((label, path))

    if missing_count > 0:
        print(
            f"Warning: {missing_count} testing videos were not found in train_metadata.csv")
        print("Example missing videos:")
        for label, path in missing_examples:
            print(f"  {label} {path}")
        if missing_count > 5:
            print(f"  ... and {missing_count - 5} more")

    # Show first few entries of each result file
    print("\nSample entries in new train_metadata.csv:")
    with open(output_train_csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"  {','.join(row)}")

    print("\nSample entries in test_metadata.csv:")
    if test_rows:
        with open(output_test_csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(f"  {','.join(row)}")
    else:
        print("  No entries found")


def debug_matches(train_csv_path, list_of_testing_videos_path):
    """
    Debug specific matching cases to identify issues
    """
    print("\nDEBUG: Checking specific matches")

    # Get a list of test videos
    test_videos = []
    with open(list_of_testing_videos_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                label, path = parts
                test_videos.append((label, path))

    # Select a few test videos to debug
    debug_cases = test_videos[:5]

    # Look for these videos in the train metadata
    with open(train_csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        # Store first 1000 entries to check
        train_entries = []
        for i, row in enumerate(reader):
            if i >= 1000:
                break
            if len(row) >= 2:
                train_entries.append(row)

        for label, test_path in debug_cases:
            print(f"\nLooking for matches for: {label} {test_path}")
            test_filename = os.path.basename(test_path)
            test_path_norm = test_path.replace('\\', '/').lower()

            found_matches = False
            for row in train_entries:
                train_path = row[1]
                train_path_norm = train_path.replace('\\', '/').lower()

                if test_filename.lower() in train_path_norm:
                    print(f"  Potential match: {train_path}")
                    found_matches = True

                if test_path_norm in train_path_norm:
                    print(f"  Direct path match: {train_path}")
                    found_matches = True

            if not found_matches:
                print("  No matches found")


if __name__ == "__main__":
    # Configuration
    train_csv_path = "output_test_1_image/val_metadata.csv"  # Original train metadata
    list_of_testing_videos_path = "output_test_1_image/List_of_testing_videos.txt"
    # Output train metadata (without test videos)
    output_train_csv_path = "output_test_1_image/new_val_metadata.csv"
    # Output test metadata
    output_test_csv_path = "output_test_1_image/new_test_val_metadata.csv"

    # Debug specific matches to understand the issue
    debug_matches(train_csv_path, list_of_testing_videos_path)

    # Split the metadata
    split_metadata(train_csv_path, list_of_testing_videos_path,
                   output_train_csv_path, output_test_csv_path)
