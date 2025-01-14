"""Module for set augmentations"""

from typing import Callable, Tuple

import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch import Tensor


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
    preprocessing_fn: Callable = None,
) -> albu.BaseCompose:
    """
    Set transforms of image and mask
    :param width: width of image
    :param height: height of image
    :param preprocessing: flag for preprocessing
    :param augmentations: flag for main augmentations
    :param postprocessing: flag for postprocessing augmentations
    :param preprocessing_fn: function for normalization according to selected model
    :return: composition of transformations
    """
    transforms = []

    if preprocessing:
        transforms.append(
            albu.Resize(height=height, width=width),
        )

    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.2),
                albu.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3,
                ),
                albu.CLAHE(clip_limit=7.0, tile_grid_size=(5, 5), p=0.5),
                albu.OneOf(
                    [
                        albu.MotionBlur(blur_limit=5),
                        albu.GaussianBlur(
                            blur_limit=(3, 7),
                            sigma_limit=0,
                            always_apply=False,
                            p=0.5,
                        ),
                        albu.ISONoise(
                            color_shift=(0.01, 0.05),
                            intensity=(0.1, 0.5),
                            always_apply=False,
                        ),
                        albu.ImageCompression(
                            quality_lower=60,
                            quality_upper=100,
                        ),
                    ],
                    p=0.5,
                ),
            ],
        )

    if postprocessing:
        if preprocessing_fn is not None:
            transforms.extend(
                [
                    albu.Lambda(image=preprocessing_fn),
                    ToTensorV2(),
                ],
            )
        else:
            transforms.extend(
                [
                    albu.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                        always_apply=False,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
            )

    return albu.Compose(
        transforms,
    )


def cv_image_to_tensor(img: NDArray[float], normalize: bool = True) -> Tensor:
    """
    Converts a image to tensor
    :param img: array with image data
    :param normalize: flag for normalization
    :return: tensor for model
    """
    ops = [ToTensorV2()]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)["image"]


def denormalize(
    img: NDArray[float],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
) -> NDArray[int]:
    """
    Apply denormalization to image
    :param img: array with image data
    :param mean: means for each image channels
    :param std: standard deviation for each image channels
    :param max_value: maximum value of pixel in image
    :return: RGB image
    """
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)["image"] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray[float]:
    """
    Converts a tensor shape to image shape
    :param tensor: tensor with image (input to model)
    :return: array with image
    """
    return tensor.permute(1, 2, 0).cpu().numpy()
