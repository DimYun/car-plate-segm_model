from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    n_workers: int
    valid_size: float
    test_size: float
    width: int
    height: int


class TrainConfig(BaseModel):
    n_epochs: int
    accelerator: str
    device: int
    model_type: str
    encoder_name: str
    encoder_weights: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict


class Config(BaseModel):
    project_name: str
    experiment_name: str
    monitor_metric: str
    monitor_mode: str
    cls_losses: List[LossConfig]
    seg_losses: List[LossConfig]
    data_config: DataConfig
    train_config: TrainConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
