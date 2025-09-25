import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path


class JerseyVisibilityDataset(Dataset):
    """Custom dataset for jersey number visibility classification"""

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.data_dir)
                               if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load all image paths and labels
        self.samples = []
        self.class_counts = {cls_name: 0 for cls_name in self.classes}

        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.class_counts[class_name] += 1

        print(f"{split} dataset: {len(self.samples)} samples, {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
        print(f"Class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


class JerseyVisibilityClassifier:
    """Configurable jersey number visibility classifier"""

    def __init__(self, config, num_classes, results_dir, class_weights=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.results_dir = results_dir
        self.config = config

        # Initialize model based on config
        self.model = self._create_model(config['model'])
        self.model = self.model.to(device)

        # Define loss function and optimizer based on config
        self.criterion = self._create_criterion(config['training']['loss'], class_weights)
        self.optimizer = self._create_optimizer(config['training']['optimizer'])
        self.scheduler = self._create_scheduler(config['training']['scheduler'])

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def _create_model(self, model_config):
        """Create model based on configuration"""
        model_name = model_config['name'].lower()
        pretrained = model_config.get('pretrained', True)

        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def _create_criterion(self, loss_config, class_weights):
        """Create loss function based on configuration"""
        loss_name = loss_config['name'].lower()

        if loss_name == 'crossentropyloss':
            if class_weights is not None:
                class_weights = torch.FloatTensor(class_weights).to(self.device)
                print(f"Using class weights: {class_weights}")
                return nn.CrossEntropyLoss(weight=class_weights)
            else:
                return nn.CrossEntropyLoss()
        elif loss_name == 'focalloss':
            # Simple implementation of Focal Loss
            alpha = loss_config.get('alpha', 1.0)
            gamma = loss_config.get('gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights, device=self.device)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _create_optimizer(self, optimizer_config):
        """Create optimizer based on configuration"""
        optimizer_name = optimizer_config['name'].lower()
        lr = optimizer_config['lr']
        weight_decay = optimizer_config.get('weight_decay', 0.0)

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self, scheduler_config):
        """Create learning rate scheduler based on configuration"""
        if scheduler_config is None:
            return None

        scheduler_name = scheduler_config['name'].lower()

        if scheduler_name == 'steplr':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosineannealinglr':
            T_max = scheduler_config.get('T_max', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'reducelronplateau':
            factor = scheduler_config.get('factor', 0.1)
            patience = scheduler_config.get('patience', 10)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=factor, patience=patience)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use ascii=True to avoid fancy unicode icons in tqdm
        progress_bar = tqdm(dataloader, desc='Training', ascii=True)
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            accuracy = 100 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{running_loss / len(progress_bar):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            # Use ascii=True to avoid fancy unicode icons in tqdm
            progress_bar = tqdm(dataloader, desc='Validation', ascii=True)
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                accuracy = 100 * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{running_loss / len(progress_bar):.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy, all_predictions, all_labels

    def train(self, train_loader, val_loader, num_epochs=20, save_path='jersey_visibility_model.pth'):
        """Train the model"""
        print(f"Training on {self.device}")
        print(f"Model: {self.config['model']['name']}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader)

            # Validate
            val_loss, val_accuracy, _, _ = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Ensure parent dir exists before saving
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'class_names': train_loader.dataset.classes,
                    'config': self.config
                }, save_path)
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}% -> {save_path}")

        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        return best_val_accuracy

    def test(self, test_loader, model_path='jersey_visibility_model.pth'):
        """Test the model"""
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint['class_names']

        print("Testing the model...")
        test_loss, test_accuracy, predictions, true_labels = self.validate(test_loader)

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        self.plot_confusion_matrix(cm, class_names)

        return test_accuracy, predictions, true_labels

    def plot_training_history(self):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy', marker='o')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', marker='s')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        training_history_path = self.results_dir / 'visibility_training_history.png'
        plt.savefig(training_history_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history plot saved to {training_history_path}")

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Jersey Number Visibility Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        confusion_matrix_path = self.results_dir / 'visibility_confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {confusion_matrix_path}")


class FocalLoss(nn.Module):
    """Focal Loss implementation for imbalanced datasets"""

    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, device='cuda', reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"Using class weights in FocalLoss: {class_weights}")

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calculate_class_weights(train_dataset):
    """Calculate class weights based on inverse frequency"""
    class_counts = train_dataset.class_counts
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    # Calculate weights as inverse frequency
    class_weights = []
    for class_name in train_dataset.classes:
        weight = total_samples / (num_classes * class_counts[class_name])
        class_weights.append(weight)

    print(f"Class counts: {class_counts}")
    print(f"Calculated weights: {dict(zip(train_dataset.classes, class_weights))}")

    return class_weights


def get_transforms(config):
    """Get data transforms based on configuration"""
    transform_config = config.get('transforms', {})

    # Training transforms
    train_transforms = [
        transforms.Resize((transform_config.get('resize', 256), transform_config.get('resize', 256))),
        transforms.RandomCrop((transform_config.get('crop_size', 224), transform_config.get('crop_size', 224))),
    ]

    # Add data augmentation if specified
    if transform_config.get('horizontal_flip', True):
        train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))

    if transform_config.get('color_jitter', True):
        train_transforms.append(transforms.ColorJitter(
            brightness=transform_config.get('brightness', 0.2),
            contrast=transform_config.get('contrast', 0.2),
            saturation=transform_config.get('saturation', 0.2),
            hue=transform_config.get('hue', 0.1)
        ))

    if transform_config.get('rotation', True):
        train_transforms.append(transforms.RandomRotation(degrees=transform_config.get('rotation_degrees', 10)))

    # Always add normalization
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=transform_config.get('mean', [0.485, 0.456, 0.406]),
            std=transform_config.get('std', [0.229, 0.224, 0.225])
        )
    ])

    # Validation/Test transforms (no augmentation)
    val_test_transforms = [
        transforms.Resize((transform_config.get('crop_size', 224), transform_config.get('crop_size', 224))),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=transform_config.get('mean', [0.485, 0.456, 0.406]),
            std=transform_config.get('std', [0.229, 0.224, 0.225])
        )
    ]

    train_transform = transforms.Compose(train_transforms)
    val_test_transform = transforms.Compose(val_test_transforms)

    return train_transform, val_test_transform


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config(config_path):
    """Create a default train_visibility.yaml configuration file"""
    default_config = {
        'model': {
            'name': 'resnet50',
            'pretrained': True
        },
        'training': {
            'batch_size': 32,
            'epochs': 20,
            'use_class_weights': True,
            'loss': {
                'name': 'CrossEntropyLoss'
            },
            'optimizer': {
                'name': 'Adam',
                'lr': 0.001,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'name': 'StepLR',
                'step_size': 10,
                'gamma': 0.1
            }
        },
        'transforms': {
            'resize': 256,
            'crop_size': 224,
            'horizontal_flip': True,
            'color_jitter': True,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'rotation': True,
            'rotation_degrees': 10,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Created default configuration file: {config_path}")
    return default_config


def main():
    parser = argparse.ArgumentParser(description='Jersey Number Visibility Classification with YAML Configuration')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (if not provided, will use root/config/train_visibility.yaml)')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing with pre-trained model')
    parser.add_argument('--create_config', action='store_true',
                        help='Create a default configuration file and exit')

    args = parser.parse_args()

    # Get the root directory (two levels up from current script location: root/src/trainer)
    script_dir = Path(__file__).parent  # root/src/trainer
    root_dir = script_dir.parent.parent  # root
    data_dir = root_dir / "data"
    config_dir = root_dir / "config"
    results_dir = config_dir / "results"

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create outputs directory (for model .pth files) if it doesn't exist
    outputs_dir = config_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Set default config path
    if args.config is None:
        args.config = str(config_dir / "train_visibility.yaml")

    print(f"Root directory: {root_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Config directory: {config_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Outputs directory (models): {outputs_dir}")
    print(f"Configuration file: {args.config}")

    # Create default config if requested
    if args.create_config:
        config_dir.mkdir(parents=True, exist_ok=True)
        create_default_config(args.config)
        return

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found!")
        print("Creating default configuration file...")
        config_dir.mkdir(parents=True, exist_ok=True)
        config = create_default_config(args.config)
        print(f"Please modify {args.config} as needed and run again.")
        return
    else:
        config = load_config(args.config)
        print("Configuration loaded successfully!")

    # Set paths
    data_path = str(data_dir / "visibility_dataset")
    # Save model in root/config/outputs as requested
    model_save_path = str(outputs_dir / "jersey_visibility_model.pth")

    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset directory '{data_path}' not found!")
        print("Please make sure you've run the dataset preparation script first.")
        return

    # Get transforms based on config
    train_transform, val_test_transform = get_transforms(config)

    # Create datasets
    train_dataset = JerseyVisibilityDataset(data_path, 'train', train_transform)
    val_dataset = JerseyVisibilityDataset(data_path, 'val', val_test_transform)
    test_dataset = JerseyVisibilityDataset(data_path, 'test', val_test_transform)

    # Create data loaders with batch size from config
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Calculate class weights if enabled
    class_weights = None
    if config['training'].get('use_class_weights', True):
        class_weights = calculate_class_weights(train_dataset)

    # Initialize classifier with config
    num_classes = len(train_dataset.classes)
    classifier = JerseyVisibilityClassifier(config, num_classes, results_dir, class_weights)

    if args.test_only:
        # Test only
        if not os.path.exists(model_save_path):
            print(f"Error: Model file '{model_save_path}' not found!")
            return

        test_accuracy, predictions, true_labels = classifier.test(test_loader, model_save_path)
    else:
        # Train and test
        epochs = config['training']['epochs']
        print(f"Starting training with {num_classes} classes for {epochs} epochs...")
        best_val_acc = classifier.train(train_loader, val_loader, epochs, model_save_path)

        # Plot training history
        classifier.plot_training_history()

        # Test the model
        test_accuracy, predictions, true_labels = classifier.test(test_loader, model_save_path)

        # Save training results
        results = {
            'config': config,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_accuracy,
            'num_classes': num_classes,
            'class_names': train_dataset.classes,
            'class_weights': class_weights,
            'train_losses': classifier.train_losses,
            'train_accuracies': classifier.train_accuracies,
            'val_losses': classifier.val_losses,
            'val_accuracies': classifier.val_accuracies
        }

        results_json_path = results_dir / 'visibility_training_results.json'
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save config used for training
        config_save_path = results_dir / 'visibility_training_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"\nTraining results saved to {results_json_path}")
        print(f"Training configuration saved to {config_save_path}")
        print(f"Model saved to {model_save_path}")
        print(f"All results saved in: {results_dir}")


if __name__ == "__main__":
    main()
