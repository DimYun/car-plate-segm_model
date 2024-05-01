from typing import Any, Optional
import segmentation_models_pytorch as smp

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from configs.config import Config
from src.augmentations import get_transforms
from src.dataset import PlateDataset
from src.dataset_splitter import preprocess_and_split


class PlateDM(LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self.classes = {}
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        if not self.classes:
            (
                self.train_data,
                self.validation_data,
                self.test_data,
                self.classes
            ) = preprocess_and_split(
                self._config.data_config.data_path,
                self._config.data_config.valid_size,
                self._config.data_config.test_size
            )
        else:
            pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = PlateDataset(
                image_paths=self.train_data["Full_paths"],
                image_bboxes=self.train_data["Bboxes"],
                image_labels=self.train_data["Labels"],
                classes=['plate'],
                transforms=get_transforms(
                    width=self._config.data_config.width,
                    height=self._config.data_config.height,
                    preprocessing=True,
                    augmentations=True,
                    postprocessing=smp.encoders.get_preprocessing_fn(
                        self._config.train_config.encoder_name,
                        self._config.train_config.encoder_weights
                    )
                )
            )
            self.valid_dataset = PlateDataset(
                image_paths=self.validation_data["Full_paths"],
                image_bboxes=self.validation_data["Bboxes"],
                image_labels=self.validation_data["Labels"],
                classes=['plate'],
                transforms=get_transforms(
                    width=self._config.data_config.width,
                    height=self._config.data_config.height,
                    preprocessing=True,
                    augmentations=False,
                    postprocessing=smp.encoders.get_preprocessing_fn(
                        self._config.train_config.encoder_name,
                        self._config.train_config.encoder_weights
                    )
                )
            )
        elif stage == "test":
            self.test_dataset = PlateDataset(
                image_paths=self.test_data["Full_paths"],
                image_bboxes=self.test_data["Bboxes"],
                image_labels=self.test_data["Labels"],
                classes=['plate'],
                transforms=get_transforms(
                    width=self._config.data_config.width,
                    height=self._config.data_config.height,
                    preprocessing=True,
                    augmentations=False,
                    postprocessing=smp.encoders.get_preprocessing_fn(
                        self._config.train_config.encoder_name,
                        self._config.train_config.encoder_weights
                    )
                )
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
