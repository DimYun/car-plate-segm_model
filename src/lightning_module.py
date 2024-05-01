import pytorch_lightning as pl
import torch
from typing import List

from configs.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.train_utils import load_object


class PlateModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._model = load_object(self._config.train_config.model_type)(
            encoder_name=self._config.train_config.encoder_name,  # 'mit_b3', You can choose a different encoder as needed
            encoder_weights=self._config.train_config.encoder_weights,  # 'imagenet',
            classes=1,  # For binary segmentation
            activation=None  # No activation, as we'll apply sigmoid in the training step
        )
        self._losses = get_losses(self._config.seg_losses)
        self._valid_metrics = get_metrics()
        self._test_metrics = get_metrics()
        self.save_hyperparameters(self._config.dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.train_config.optimizer)(
            self._model.parameters(),
            **self._config.train_config.optimizer_kwargs,
        )
        scheduler = load_object(
            self._config.train_config.scheduler,
        )(optimizer, **self._config.train_config.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'val_loss',
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Считаем лосс.
        """
        images, gt_masks = batch
        pred_masks_logits = self(images)
        # print(gt_masks.shape)
        # print(pred_masks_logits.shape)
        loss = self._calculate_loss(pred_masks_logits, gt_masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Считаем лосс и метрики.
        """
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)

        # print(f"Shape of pred_masks_logits: {pred_masks_logits.shape}")
        # print(f"Shape of gt_masks: {gt_masks.shape}")
        # print(f"Unique values in gt_masks: {torch.unique(gt_masks)}")

        loss = self._calculate_loss(pred_masks_logits, gt_masks)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._valid_metrics.update(pred_masks, gt_masks)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Считаем метрики.
        """
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._test_metrics.update(pred_masks, gt_masks)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        metrics = self._valid_metrics.compute()
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_epoch=True, prog_bar=True)
        self._valid_metrics.reset()

    def on_test_epoch_end(self) -> None:
        metrics = self._test_metrics.compute()
        for key, value in metrics.items():
            self.log(f'test_{key}', value, on_epoch=True)
        self._test_metrics.reset()

    def _calculate_loss(
            self,
            pred_masks_logits: torch.Tensor,
            gt_masks: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0
        for loss_wrapper in self._losses:
            loss = loss_wrapper.loss(pred_masks_logits, gt_masks)
            total_loss += loss_wrapper.weight * loss
        return total_loss
