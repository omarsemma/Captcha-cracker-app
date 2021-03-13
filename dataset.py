import numpy as np
import albumentations

from PIL import Image

import torch


class ClassificationDataset:
    def __init__(self, image_paths, labels = None, resize=None, test=False):
        
        self.image_paths = image_paths
        self.labels = labels
        self.resize = resize

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.labels is not None:
            labels = self.labels[item]
            return {
            "images": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
            }

        return {
            "images": torch.tensor(image, dtype=torch.float)
        }
