from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode

train_augmentations = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomResizedCrop(size=(240, 240), scale=(0.8, 1.0)), #vit 240x240, effnet 380x380
    v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomRotation(degrees=15),
    v2.RandomAutocontrast(),
])

train_augmentations_multiband = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
    v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR, fill=0),
])

eval_augmentations_multiband = v2.Compose([
    v2.Resize((384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True),
])
