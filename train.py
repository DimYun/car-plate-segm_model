import argparse
import os

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint
)
from configs.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import PlateDM
from src.lightning_module import PlateModule


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train(config: Config):
    datamodule = PlateDM(config)
    model = PlateModule(config)

    Task.init(
        project_name=config.project_name,
        task_name=f"{config.experiment_name}",
        auto_connect_frameworks=True,
    )
    # task.connect(config.dict())  # TODO: ?

    experiment_save_path = EXPERIMENTS_PATH / config.experiment_name
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f"epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}",
    )
    trainer = pl.Trainer(
        max_epochs=config.train_config.n_epochs,
        accelerator=config.train_config.accelerator,
        devices=[config.train_config.device],
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback,
            # EarlyStopping(
            #     monitor=config.monitor_metric,
            #     patience=4,
            #     mode=config.monitor_mode,
            # ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        # deterministic=True,  TODO: ?
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == "__main__":
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
