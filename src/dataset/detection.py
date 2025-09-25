import cv2
import json
import os
import sys
from pathlib import Path
import re
import yaml

# Customize this mapping for your categories
CATEGORY_MAPPING = {2: 0, 4: 1, 3: 2}  # In the original JSON file, class 2 is bystander, 4 is player and 3 is ball.

# Class names corresponding to the mapped indices
CLASS_NAMES = ["bystander", "player", "ball"]


def extract_video_frames(video_path, output_dir, subset_name):
    """
    Extract frames from a video and save them as JPG images.

    Args:
        video_path (Path): Path to the input video file
        output_dir (Path): Directory to save extracted frames
        subset_name (str): Name of the subset (train/val/test)
    """
    # Check if video file exists
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' does not exist!")
        return False

    video_name = video_path.stem  # filename without extension

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path.name}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print("Extracting frames...")

    frame_count = 0
    extracted_count = 0

    while True:
        # Read frame from video
        ret, frame = cap.read()

        # Break if no more frames
        if not ret:
            break

        # Create filename: video_name_frame_i.jpg with 6-digit frame number
        frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
        frame_path = output_dir / frame_filename

        # Save frame as JPG
        success = cv2.imwrite(str(frame_path), frame)

        if success:
            extracted_count += 1
            # Print progress every 100 frames
            if extracted_count % 100 == 0:
                print(f"Extracted {extracted_count}/{total_frames} frames...")
        else:
            print(f"Failed to save frame {frame_count}")

        frame_count += 1

    # Release video capture object
    cap.release()

    print(f"Extraction complete!")
    print(f"Total frames extracted: {extracted_count}")
    print(f"Frames saved in: {output_dir}")

    return True


def convert_json_to_yolo(json_file_path: Path, output_dir: Path):
    """
    Convert one JSON file to YOLO format labels.
    Label filenames use the json file stem as prefix:
        <json_stem>_frame_<6-digit>.txt
    Frame number is extracted from each image filename (last digit group).
    If no digits found, a fallback sequential index is used.
    """
    json_stem = json_file_path.stem
    print(f"\n Processing {json_file_path}  to labels in {output_dir}")

    try:
        with json_file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f" Error reading {json_file_path}: {e}")
        return

    # Validate required keys
    for key in ('categories', 'images', 'annotations'):
        if key not in data:
            print(f" Missing key '{key}' in {json_file_path}")
            return

    # Build image_id -> info (include original filename stem)
    image_info = {}
    for img in data['images']:
        fid = img['id']
        orig_name_stem = Path(img['file_name']).stem
        image_info[fid] = {
            "orig_name": orig_name_stem,
            "width": img['width'],
            "height": img['height']
        }

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        cid = ann['category_id']
        if cid in CATEGORY_MAPPING:
            annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_labels = 0
    total_annotations = 0
    fallback_counter = 1  # used when no digits in filename

    for image_id, anns in annotations_by_image.items():
        info = image_info.get(image_id)
        if not info:
            print(f"  image_id {image_id} not found in images list — skipping")
            continue

        orig_stem = info["orig_name"]

        # Try to extract the last numeric group from the image filename stem
        m = re.search(r"(\d+)$", orig_stem)
        if m:
            frame_num = int(m.group(1))
        else:
            frame_num = fallback_counter
            fallback_counter += 1

        label_filename = f"{json_stem}_frame_{frame_num:06d}.txt"
        label_path = output_dir / label_filename

        with label_path.open("w", encoding="utf-8") as lf:
            for ann in anns:
                cid = ann['category_id']
                x, y, w, h = ann['bbox']  # COCO top-left x,y,width,height

                # Convert to YOLO normalized (x_center, y_center, w, h)
                x_center = (x + w / 2) / info["width"]
                y_center = (y + h / 2) / info["height"]
                w_norm = w / info["width"]
                h_norm = h / info["height"]

                lf.write(f"{CATEGORY_MAPPING[cid]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                total_annotations += 1

        total_labels += 1

    print(f" {json_file_path.name}: created {total_labels} label files ({total_annotations} annotations).")


def create_yaml_config(detection_dataset_dir: Path, subsets: list):
    """
    Create a YAML configuration file for the YOLO dataset.

    Args:
        detection_dataset_dir (Path): Path to the detection dataset directory
        subsets (list): List of subset names (train, val, test)
    """
    images_dir = detection_dataset_dir / "images"

    # Create the YAML configuration
    yaml_config = {
        'nc': len(CLASS_NAMES),  # number of classes
        'names': CLASS_NAMES
    }

    # Add paths for each existing subset
    for subset in subsets:
        subset_images_dir = images_dir / subset
        if subset_images_dir.exists():
            # Convert to absolute path and use forward slashes for cross-platform compatibility
            abs_path = subset_images_dir.resolve()
            yaml_config[subset] = str(abs_path).replace('\\', '/')

    # Write the YAML file
    yaml_path = detection_dataset_dir / "dataset.yaml"

    try:
        with yaml_path.open('w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        print(f"\n YAML configuration created: {yaml_path}")
        print("YAML contents:")
        print("-" * 40)
        with yaml_path.open('r', encoding='utf-8') as f:
            print(f.read())
        print("-" * 40)

        return True

    except Exception as e:
        print(f" Error creating YAML config: {e}")
        return False


def process_videos_and_annotations():
    """
    Main pipeline function to process videos and convert annotations to YOLO format.
    """
    # Get the root directory (two levels up from current script location: root/src/dataset)
    script_dir = Path(__file__).parent  # root/src/dataset
    root_dir = script_dir.parent.parent  # root

    # Input directories
    data_dir = root_dir / "data"

    # Output directory structure
    detection_dataset_dir = data_dir / "detection_dataset"
    images_dir = detection_dataset_dir / "images"
    labels_dir = detection_dataset_dir / "labels"

    # Define subsets
    subsets = ["train", "val", "test"]

    print(f"Root directory: {root_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output dataset directory: {detection_dataset_dir}")

    # Check if data directory exists
    if not data_dir.exists():
        print(f" Data directory '{data_dir}' does not exist!")
        return False

    # Keep track of which subsets actually have data
    processed_subsets = []

    for subset in subsets:
        subset_input_dir = data_dir / f"football_{subset}"
        subset_images_dir = images_dir / subset
        subset_labels_dir = labels_dir / subset

        print(f"\n=== Processing {subset} subset ===")

        # Check if subset input directory exists
        if not subset_input_dir.exists():
            print(f" {subset_input_dir} not found — skipping {subset}")
            continue

        subset_has_data = False

        # Process videos (extract frames)
        video_files = list(subset_input_dir.rglob("*.mp4")) + list(subset_input_dir.rglob("*.avi")) + list(
            subset_input_dir.rglob("*.mov"))

        if video_files:
            print(f"Found {len(video_files)} video file(s) in {subset_input_dir}")
            for video_file in video_files:
                print(f"Extracting frames from: {video_file}")
                if extract_video_frames(video_file, subset_images_dir, subset):
                    subset_has_data = True
        else:
            print(f"No video files found in {subset_input_dir}")

        # Process JSON annotations
        print(f"\n--- Converting JSON annotations for {subset} ---")
        json_files = list(subset_input_dir.rglob("*.json"))

        if not json_files:
            print(f" No JSON files found under {subset_input_dir} (including subfolders).")
        else:
            print(f"Found {len(json_files)} JSON file(s)")
            for jf in json_files:
                convert_json_to_yolo(jf, subset_labels_dir)
            subset_has_data = True

        if subset_has_data:
            processed_subsets.append(subset)

    # Create YAML configuration file
    if processed_subsets:
        print(f"\n=== Creating YAML Configuration ===")
        create_yaml_config(detection_dataset_dir, processed_subsets)
    else:
        print(f"\n No subsets were processed, skipping YAML creation")

    print(f"\n Pipeline complete! YOLO dataset created in: {detection_dataset_dir}")
    return True


def main():
    """Main function to run the complete pipeline."""
    print(" Starting YOLO Dataset Creation Pipeline")
    print("=" * 50)

    success = process_videos_and_annotations()

    if not success:
        print(" Pipeline failed!")
        sys.exit(1)
    else:
        print(" Pipeline completed successfully!")


if __name__ == "__main__":
    main()