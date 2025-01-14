"""Module for load and split data"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np


def preprocess_and_split(
    data_path: Union[str, Path],
    valid_part: float = 0.2,
    test_part: float = 0.1,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Initiate preprocessing and split the dataset
    :param data_path: Path to the COCO datafolder with annotation file and images
    :param valid_part: Part of the validation data
    :param test_part: Part of the test data
    :return: Tuple of dictionaries with train, validate and test data for torch.Dataset
    """
    (
        train_data_dict,
        validate_data_dict,
        test_data_dict,
        classes,
    ) = get_split_data(
        data_path=Path(data_path),
        valid_part=valid_part,
        test_part=test_part,
    )
    return (
        train_data_dict,
        validate_data_dict,
        test_data_dict,
        classes,
    )


def get_split_data(
    data_path: Union[str, Path],
    valid_part: float = 0.2,
    test_part: float = 0.1,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Get data from COCO format and form dictionary for NN model
    """

    images_path = data_path / "single_COCO"
    json_coco_path = data_path / "COCO_plate_dataset.json"

    image_data_dict = {
        "Id": [],
        "Full_paths": [],
        "Bboxes": [],
        "Labels": [],
    }

    # Put here images for skip, TODO: set in config file
    skip_imgs_path = []
    skip_ids = []

    # Get classes and encode them
    all_classes = {}
    with open(json_coco_path, "r", encoding="utf-8") as json_file:
        plate_coco_data = json.load(json_file)

        for categories_data in plate_coco_data["categories"]:
            if categories_data["id"] != 0 and categories_data["id"] not in all_classes:
                all_classes[categories_data["id"]] = categories_data["name"]
            elif categories_data["id"] == 0:
                print("0 is not background!")
                all_classes[categories_data["id"]] = categories_data["name"]
        print(f"Classes: {all_classes}")

        # Process image data
        for image_data in plate_coco_data["images"]:
            real_img_id = image_data["id"]
            if image_data["file_name"].split("/")[-1] in skip_imgs_path:
                skip_ids.append(real_img_id)
                continue
            for k in image_data_dict:
                image_data_dict[k].append([])
            image_data_dict["Id"][-1] = real_img_id
            img_path = images_path / image_data["file_name"]
            image_data_dict["Full_paths"][-1] = str(img_path)
        print(
            f"Total images: "
            f"{len(image_data_dict['Full_paths'])} == {len(image_data_dict['Id'])}",
        )

        # Process annotations data
        for annotation_data in plate_coco_data["annotations"]:
            # Data may be numerated to image of have through numeration
            process_image_id = annotation_data["image_id"]
            if process_image_id in skip_ids:
                continue

            # Calculate index for past annotation data
            image_data_indx = image_data_dict["Id"].index(process_image_id)
            image_data_dict["Labels"][image_data_indx].append(
                annotation_data["category_id"],
            )
            # Process each granule
            if isinstance(annotation_data["bbox"], list):
                x, y, w, h = annotation_data["bbox"]
                image_data_dict["Bboxes"][image_data_indx].append(
                    [x, y, x + w, y + h],
                )
            else:
                print(annotation_data["bbox"])
                continue

    for image_key in image_data_dict:
        print(f"\t{image_key} len: {len(image_data_dict[image_key])}")
    print(f"Labels example: {image_data_dict['Labels'][0]}")
    print(f"Bboxes example: {image_data_dict['Bboxes'][0]}")
    print(f"Sckipped data: {len(skip_ids)}")

    train_indexes = np.array([])
    test_indexes = np.array([])
    val_indexes = np.array([])
    group_ids = np.array(image_data_dict["Id"])

    np.random.shuffle(group_ids)
    train, val, test, _ = np.split(
        group_ids,
        [
            int((1 - valid_part - test_part) * group_ids.shape[0]),
            int((1 - test_part) * group_ids.shape[0]),
            group_ids.shape[0],
        ],
    )
    train_indexes = np.append(train_indexes, train)
    val_indexes = np.append(val_indexes, val)
    test_indexes = np.append(test_indexes, test)

    train_data_dict = {}
    validate_data_dict = {}
    test_data_dict = {}
    for fkey in image_data_dict:
        train_data_dict[fkey] = []
        validate_data_dict[fkey] = []
        test_data_dict[fkey] = []

    _data_dict = {
        0: train_data_dict,
        1: validate_data_dict,
        2: test_data_dict,
    }
    for indexes_i, indexes in enumerate(
        (train_indexes, val_indexes, test_indexes),
    ):
        for index_i in indexes:
            for fkey in image_data_dict:
                _data_dict[indexes_i][fkey].append(
                    image_data_dict[fkey][int(index_i)],
                )
    print(
        f"Split {group_ids.shape} into: "
        f"{len(train_data_dict['Full_paths'])} train, "
        f"{len(validate_data_dict['Full_paths'])} validate, "
        f"{len(test_data_dict['Full_paths'])} test",
    )

    return train_data_dict, validate_data_dict, test_data_dict, all_classes
