"""Module for Dataset preporation"""

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as albu
import cv2
import jpeg4py as jpeg
import numpy as np
from torch.utils.data import Dataset


class PlateDataset(Dataset):
    """
    Dataset module for Plate detection data preporation
    """

    def __init__(
        self,
        image_paths: List,
        image_bboxes: List,
        image_labels: List,
        classes: Optional[list] = None,
        transforms: Optional[albu.BaseCompose] = None,
    ):
        self.image_paths = image_paths
        self.image_bboxes = image_bboxes
        self.image_labels = image_labels
        self.transforms = transforms
        self.classes = classes

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset
        :return: number of images in dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Select and prepare one image with mask
        :param idx: index of the image in dataset
        :return: preprocessed image and binary mask
        """
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
                    -1,
                )

        if self.transforms is not None:
            out = self.transforms(
                image=image,
                mask=mask,
            )
            image = out["image"]
            mask = out["mask"]

        return image, mask
