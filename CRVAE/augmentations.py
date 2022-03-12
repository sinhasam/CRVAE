from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]


SimpleAugmentation = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
    transforms.ToTensor(),
])


LargeAugmentation = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
    transforms.ToTensor(),
])


SimpleVerticalFlipAugmentation = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
    transforms.ToTensor(),
])


LargeVerticalFlipAugmentation = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
    transforms.ToTensor(),
])


SimpleJitterAugmentation = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
    transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
    transforms.ToTensor(),
])


LargeJitterAugmentation = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
    transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
    transforms.ToTensor(),
])


SimpleVerticalFlipJitterAugmentation = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
    transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
    transforms.ToTensor(),
])


LargeVerticalFlipJitterAugmentation = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
    transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
    transforms.ToTensor(),
])

augmentations = {
    'simple': SimpleAugmentation,
    'large': LargeAugmentation,
    'simple_vertical_flip': SimpleVerticalFlipAugmentation,
    'large_vertical_flip': LargeVerticalFlipAugmentation,
    'simple_jitter': SimpleJitterAugmentation,
    'large_jitter': LargeJitterAugmentation,
    'simple_vertical_flip_jitter': SimpleVerticalFlipJitterAugmentation,
    'large_vertical_flip_jitter': LargeVerticalFlipJitterAugmentation,
}


def get_augmentation(aug_type, normalize=False, mean=None, std=None):
    augmentation = augmentations[aug_type]    
    if normalize:
        mean = mean if mean is not None else IMAGENET_MEAN
        std = std if std is not None else IMAGENET_STD
        normalize_aug = transforms.Normalize(mean, std)
        augmentation.transforms.append(normalize_aug)

    return augmentation
