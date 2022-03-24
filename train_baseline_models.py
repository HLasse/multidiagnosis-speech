import os

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.baseline_utils.dataloader import MultiDiagnosisDataset
from src.baseline_utils.embedding_fns import get_embedding_fns
from src.baseline_utils.baseline_pl_model import BaselineClassifier

from src.util import create_argparser

import wandb

from typing import Tuple


def create_dataloaders(config, embedding_fn) -> Tuple[DataLoader, DataLoader]:
    training_data = MultiDiagnosisDataset(
        train["file"].tolist(), train["label_id"].tolist(), embedding_fn=embedding_fn
    )

    val_data = MultiDiagnosisDataset(
        val["file"].tolist(), val["label_id"].tolist(), embedding_fn=embedding_fn
    )

    train_loader = DataLoader(
        training_data, batch_size=config.batch_size, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, num_workers=config.num_workers
    )

    return train_loader, val_loader


def create_trainer(config) -> pl.Trainer:
    wandb_cb = WandbLogger(name=config.run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config.default_root_dir, config.run_name),
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            every_n_epochs=1,
        )
    ]
    if config.patience:
        early_stopping = EarlyStopping("val_loss", patience=config.patience)
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        logger=wandb_cb,
        log_every_n_steps=config.log_step,
        val_check_interval=config.val_check_interval,
        callbacks=callbacks,
        gpus=config.gpus,
        profiler=config.profiler,
        max_epochs=config.max_epochs,
        #  default_root_dir=config.default_root_dir,
        #  weights_save_path=os.path.join(config.default_root_dir, config.run_name),
        precision=config.precision,
        auto_lr_find=config.auto_lr_find,
    )
    return trainer

if __name__ == "__main__":
    # Load hyperparam config
    yml_path = os.path.join(
        os.path.dirname(__file__), "configs", "baseline_configs", "default_config.yaml"
    )
    parser = create_argparser(yml_path)
    arguments = parser.parse_args()

    # Load data files
    train_files = "data/audio_file_splits/audio_train_split.csv"
    val_files = "data/audio_file_splits/audio_val_split.csv"

    train = pd.read_csv(train_files)
    train = train[train["duration"] >= 5]

    val = pd.read_csv(val_files)
    val = val[val["duration"] >= 5]

    mapping = {"ASD": 0, "DEPR": 1, "SCHZ": 2, "TD": 3}

    train["label_id"] = train.label.replace(mapping)
    val["label_id"] = val.label.replace(mapping)

    embedding_fn_dict = get_embedding_fns()

    for feat_set in embedding_fn_dict.keys():
        if feat_set == "windowed_mfccs":
            continue
        print(f"[INFO] Starting {feat_set}...")
        # setup wandb config
        run = wandb.init(
            config=arguments,
            project="multi-diagnosis",
            dir=arguments.default_root_dir,
            allow_val_change=True,
            reinit=True,
        )

        config = run.config
        run.name = f"baseline_{feat_set}"
        config.run_name = run.name
        run.log({"feat_set": feat_set})

        # Create dataloaders, model, and trainer
        train_loader, val_loader = create_dataloaders(
            config, embedding_fn_dict[feat_set]
        )
        model = BaselineClassifier(
            num_classes=4,
            feature_set=feat_set,
            learning_rate=config.learning_rate,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer = create_trainer(config)

        if config.auto_lr_find:
            lr_finder = trainer.tuner.lr_find(model)
            config.update(
                {"learning_rate": lr_finder.suggestion()}, allow_val_change=True
            )
            fig = lr_finder.plot(suggest=True)
            run.log({"lr_finder.plot": fig})
            run.log({"found_lr": lr_finder.suggestion()})

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Finish tracking run on wandb to start the next one
        run.finish()
