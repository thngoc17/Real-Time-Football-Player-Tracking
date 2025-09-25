from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_config(config_path: Path):
    """
    Load training configuration from YAML file.

    Args:
        config_path (Path): Path to the configuration YAML file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


def get_data_path():
    """
    Get the path to the dataset YAML file.
    Assumes script is running from root/src/trainer and dataset is in root/data/detection_dataset/

    Returns:
        Path: Path to the dataset YAML file
    """
    # Get current script directory (root/src/trainer)
    script_dir = Path(__file__).parent

    # Navigate to root directory (two levels up)
    root_dir = script_dir.parent.parent

    # Construct path to dataset YAML
    dataset_yaml_path = root_dir / "data" / "detection_dataset" / "dataset.yaml"

    return dataset_yaml_path


def train_model():
    """
    Train YOLO model using configuration from YAML file.
    """
    # Get paths
    script_dir = Path(__file__).parent  # root/src/trainer
    root_dir = script_dir.parent.parent  # root
    config_path = root_dir / "config" / "train_detection.yaml"
    data_path = get_data_path()

    print("Starting YOLO Model Training")
    print("=" * 50)
    print(f"Root directory: {root_dir}")
    print(f"Config file: {config_path}")
    print(f"Dataset file: {data_path}")

    # Check if dataset file exists
    if not data_path.exists():
        print(f"Dataset file not found: {data_path}")
        print("Please make sure you have run the dataset creation script first.")
        sys.exit(1)

    # Load configuration
    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration. Please check your config file.")
        sys.exit(1)

    # Extract model and training parameters
    model_config = config.get('model', {})
    train_config = config.get('training', {})

    # Get model path/name
    model_name = model_config.get('name', 'yolo11n.pt')
    print(f"Loading model: {model_name}")

    try:
        # Load the model
        model = YOLO(model_name)
        print(f"Model loaded successfully")

        # Prepare training arguments
        train_args = {
            'data': str(data_path),
            **train_config  # Unpack all training parameters from config
        }

        print(f"\nTraining Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {data_path.name}")
        for key, value in train_config.items():
            print(f"  {key}: {value}")

        print(f"\nStarting training...")

        # Train the model
        results = model.train(**train_args)

        print(f"\nTraining completed successfully!")
        print(f"Results: {results}")

    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


def create_sample_config():
    """
    Create a sample configuration file if it doesn't exist.
    """
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent
    config_dir = root_dir / "config"
    config_path = config_dir / "train_detection.yaml"

    if config_path.exists():
        return

    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Sample configuration
    sample_config = {
        'model': {
            'name': 'yolo11n.pt'  # Can be yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
        },
        'training': {
            'epochs': 100,
            'imgsz': 1280,
            'workers': 8,
            'batch': 4,
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'patience': 50,
            'save_period': 10,
            'val': True,
            'plots': True,
            'device': 'auto',  # 'auto', 'cpu', '0', '0,1,2,3'
            'project': 'runs/detect',
            'name': 'football_detection'
        }
    }

    try:
        with config_path.open('w', encoding='utf-8') as f:
            yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)

        print(f"Sample configuration created: {config_path}")
        print("Please review and modify the configuration before training.")

    except Exception as e:
        print(f"Error creating sample config: {e}")


if __name__ == "__main__":
    # Create sample config if it doesn't exist
    create_sample_config()

    # Train the model
    train_model()
