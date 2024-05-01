from typing import Optional, Tuple, Union, List

import albumentations as albu
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import jpeg4py as jpeg


class PlateDataset(Dataset):
    def __init__(
            self,
            image_paths: List,
            image_bboxes: List,
            image_labels: List,
            classes: Optional[list]=None,
            transforms: Optional[albu.BaseCompose]=None,
    ):
        self.image_paths = image_paths
        self.image_bboxes = image_bboxes
        self.image_labels = image_labels
        self.transforms = transforms
        self.classes = classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = Path(self.image_paths[idx])
        image = jpeg.JPEG(img_path).decode()

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for bbox_i, bbox in enumerate(self.image_bboxes[idx]):
            if self.image_labels[idx][bbox_i] == 1:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(
                    mask,
                    (x_min, y_min),
                    (x_max, y_max),
                    1,
                    -1
                )
        # mask = np.stack([mask], axis=-1).astype('float')

        if self.transforms is not None:
            out = self.transforms(
                image=image,
                mask=mask
            )
            image = out["image"]
            mask = out["mask"]

        return image, mask
