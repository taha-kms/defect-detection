import torchvision.transforms as T
import torch


def get_image_transform(image_size: int = 256, center_crop: int = 224):
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(center_crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]),
    ])


def get_mask_transform(image_size: int = 256, center_crop: int = 224):
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(center_crop),
        T.ToTensor(),  # mask will be in [0,1]
        T.Lambda(lambda x: (x > 0.5).float()),  # binarize
    ])
