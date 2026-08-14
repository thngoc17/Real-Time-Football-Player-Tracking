import os
import json
import shutil
import cv2
from pathlib import Path


def create_dataset_folders():
    """Create the required folder structure for both datasets"""
    # Get the root directory (two levels up from current script location: root/src/dataset)
    script_dir = Path(__file__).parent  # root/src/dataset
    root_dir = script_dir.parent.parent  # root
    data_dir = root_dir / "data"

    datasets = ['color_dataset', 'visibility_dataset']
    splits = ['train', 'val', 'test']

    for dataset in datasets:
        for split in splits:
            dataset_path = data_dir / dataset / split
            dataset_path.mkdir(parents=True, exist_ok=True)

    print("Created dataset folders structure")


def get_jersey_color_class(jersey_color):
    """Map jersey colors to class names"""
    # You can modify this mapping based on your specific needs
    color_mapping = {
        'white': 'white',
        'black': 'black',
        'red': 'red',
        'blue': 'blue',
        'yellow': 'yellow',
        'green': 'green',
        'orange': 'orange',
        'purple': 'purple',
        'pink': 'pink',
        'gray': 'gray',
        'grey': 'gray',  # Handle both spellings
        'brown': 'brown'
    }

    # Convert to lowercase for consistent matching
    color_lower = jersey_color.lower() if jersey_color else 'unknown'
    return color_mapping.get(color_lower, 'other')


def get_visibility_class(number_visible):
    """Map number visibility to class names"""
    if number_visible in ['visible', 'partly_visible']:
        return 'visible'
    else:  # 'invisible' or any other value
        return 'invisible'


def crop_and_save_image(image_path, bbox, output_path, min_size=32):
    """Crop image using bounding box and save to output path"""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return False

        # Extract bounding box coordinates [x, y, width, height]
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Ensure coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))

        # Crop the image
        cropped = image[y:y + h, x:x + w]

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cropped image
        cv2.imwrite(str(output_path), cropped)
        return True

    except Exception as e:
        print(f"Error cropping image {image_path}: {str(e)}")
        return False


def process_json_folder(json_folder_path, split_name, root_dir):
    """Process a single JSON folder (train/val/test)"""
    processed_count = 0
    data_dir = root_dir / "data"

    # Get all subdirectories in the json folder
    for subdir in os.listdir(json_folder_path):
        subdir_path = os.path.join(json_folder_path, subdir)

        if not os.path.isdir(subdir_path):
            continue

        # Find JSON files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith('.json'):
                json_file_path = os.path.join(subdir_path, filename)

                # Extract match name from JSON filename (remove .json extension)
                match_name = filename.replace('.json', '')

                try:
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)

                    # Create mappings
                    images_dict = {img['id']: img for img in data.get('images', [])}

                    # Process annotations - every 7th annotation
                    annotations = [ann for ann in data.get('annotations', []) if ann.get('category_id') == 4]
                    for i, annotation in enumerate(annotations):
                        # Process every 4th annotation (0, 4, 8, 12, ...)
                        if i % 4 != 0:
                            continue

                        image_id = annotation.get('image_id')
                        if image_id not in images_dict:
                            continue

                        image_info = images_dict[image_id]
                        frame_name = image_info.get('file_name', '')

                        # Construct the full image filename
                        # Format: "Match_1824_1_0_subclip_3_frame_000000.jpg"
                        full_image_name = f"{match_name}_{frame_name}"
                        # Change extension to .jpg if it's .PNG
                        if full_image_name.lower().endswith('.png'):
                            full_image_name = full_image_name[:-4] + '.jpg'

                        # Find the actual image file in detection_dataset
                        image_path = data_dir / "detection_dataset" / "images" / split_name / full_image_name

                        # Also try with original extension if jpg doesn't exist
                        if not image_path.exists():
                            original_image_name = f"{match_name}_{frame_name}"
                            image_path = data_dir / "detection_dataset" / "images" / split_name / original_image_name

                        if not image_path.exists():
                            print(f"Warning: Image not found: {image_path}")
                            continue

                        # Get attributes
                        attributes = annotation.get('attributes', {})
                        jersey_color = attributes.get('team_jersey_color', '')
                        number_visible = attributes.get('number_visible', '')

                        if not jersey_color or not number_visible:
                            continue

                        # Get bounding box
                        bbox = annotation.get('bbox', [])
                        if len(bbox) != 4:
                            continue

                        # Create class labels
                        jersey_class = get_jersey_color_class(jersey_color)
                        visibility_class = get_visibility_class(number_visible)

                        # Create output filenames
                        annotation_id = annotation.get('id', processed_count)
                        base_filename = f"{match_name}_ann_{annotation_id}.jpg"

                        # Dataset 1: Jersey Color (color_dataset)
                        jersey_output_dir = data_dir / "color_dataset" / split_name / jersey_class
                        jersey_output_dir.mkdir(parents=True, exist_ok=True)
                        jersey_output_path = jersey_output_dir / base_filename

                        # Dataset 2: Number Visibility (visibility_dataset)
                        visibility_output_dir = data_dir / "visibility_dataset" / split_name / visibility_class
                        visibility_output_dir.mkdir(parents=True, exist_ok=True)
                        visibility_output_path = visibility_output_dir / base_filename

                        # Crop and save for both datasets
                        if crop_and_save_image(image_path, bbox, jersey_output_path):
                            # Copy the same cropped image for visibility dataset
                            shutil.copy2(str(jersey_output_path), str(visibility_output_path))
                            processed_count += 1

                except Exception as e:
                    print(f"Error processing {json_file_path}: {str(e)}")
                    continue

    return processed_count


def main():
    """Main function to process the entire dataset"""
    print("Starting football dataset processing...")

    # Get the root directory (two levels up from current script location: root/src/dataset)
    script_dir = Path(__file__).parent  # root/src/dataset
    root_dir = script_dir.parent.parent  # root
    data_dir = root_dir / "data"

    print(f"Root directory: {root_dir}")
    print(f"Data directory: {data_dir}")

    # Check if we're in the right directory
    required_folders = ['detection_dataset', 'football_train', 'football_val', 'football_test']
    for folder in required_folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(
                f"Error: Required folder '{folder_path}' not found. Make sure the detection_dataset and football_* folders exist in {data_dir}.")
            return

    # Create dataset folder structure
    create_dataset_folders()

    # Process each split
    splits = [
        ('football_train', 'train'),
        ('football_val', 'val'),
        ('football_test', 'test')
    ]

    total_processed = 0

    for json_folder, split_name in splits:
        print(f"\nProcessing {split_name} split...")

        # Check if images folder exists for this split in detection_dataset
        images_folder = data_dir / "detection_dataset" / "images" / split_name
        if not images_folder.exists():
            print(f"Warning: Images folder not found: {images_folder}")
            continue

        # Process the JSON folder
        json_folder_path = data_dir / json_folder
        count = process_json_folder(json_folder_path, split_name, root_dir)
        total_processed += count
        print(f"Processed {count} annotations from {split_name} split")

    print(f"\nDataset processing completed!")
    print(f"Total processed annotations: {total_processed}")

    # Print summary of created datasets
    print("\nDataset Summary:")
    dataset_info = [
        ("color_dataset", "Jersey Color"),
        ("visibility_dataset", "Number Visibility")
    ]

    for dataset_name, dataset_type in dataset_info:
        print(f"\n{dataset_type} Dataset ({dataset_name}):")

        for split in ['train', 'val', 'test']:
            split_path = data_dir / dataset_name / split
            if split_path.exists():
                classes = [d.name for d in split_path.iterdir() if d.is_dir()]
                print(f"  {split}: {len(classes)} classes - {classes}")

                # Count total images in this split
                total_images = 0
                for class_name in classes:
                    class_path = split_path / class_name
                    if class_path.is_dir():
                        images_count = len(
                            [f for f in class_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
                        total_images += images_count
                        print(f"    {class_name}: {images_count} images")
                print(f"  Total {split} images: {total_images}")


if __name__ == "__main__":
    main()