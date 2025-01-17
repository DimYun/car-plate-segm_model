"""Module for load losses function from config file"""

from dataclasses import dataclass
from typing import List

from torch import nn

from configs.config import LossConfig
from src.train_utils import load_object


@dataclass
class Loss:
    """
    Class for loading loss config data
    """
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: List[LossConfig]) -> List[Loss]:
    """
    Create object from config definition of losses
    :param losses_cfg: config parameters from config.py
    :return: list with losses objects
    """
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        )
        for loss_cfg in losses_cfg
    ]
