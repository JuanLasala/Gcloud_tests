from torchvision import transforms
from torchvision.transforms import (
    RandomHorizontalFlip, RandomResizedCrop, ColorJitter,
    RandomRotation, RandomAutocontrast
)

train_augmentations = transforms.Compose([
    RandomHorizontalFlip(),
    RandomResizedCrop(size=(380, 380), scale=(0.8, 1.0)), #para vit 240x240
    ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
    RandomRotation(degrees=15),
    RandomAutocontrast(),
])
