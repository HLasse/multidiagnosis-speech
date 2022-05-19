import os
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from wasabi import msg

from src.baseline_utils.baseline_pl_model import BaselineClassifier
from src.baseline_utils.dataloader import MultiDiagnosisDataset
from src.baseline_utils.embedding_fns import get_embedding_fns
from src.util import create_argparser


def create_dataloaders(
    train_data: pd.DataFrame, val_data: pd.DataFrame, config, embedding_fn
) -> Tuple[DataLoader, DataLoader]:
    training_data = MultiDiagnosisDataset(
        train_data["file"].tolist(),
        train_data["label_id"].tolist(),
        embedding_fn=embedding_fn,
    )

    validation_data = MultiDiagnosisDataset(
        val_data["file"].tolist(), val_data["label_id"].tolist(), embedding_fn=embedding_fn
    )

    train_loader = DataLoader(
        training_data, batch_size=config.batch_size, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        validation_data, batch_size=config.batch_size, num_workers=config.num_workers
    )

    return train_loader, val_loader


def create_trainer(config) -> pl.Trainer:
    wandb_cb = WandbLogger(name=config.run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config.default_root_dir, config.run_name),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
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
    train = load_dataset("data/audio_file_splits/windowed_splits/windowed_train_split.parquet")
    val = load_dataset("parquet", data_files={"val" : "data/audio_file_splits/windowed_splits/windowed_train_split.parquet"})

    train_files = "data/audio_file_splits/audio_train_split.csv"
    val_files = "data/audio_file_splits/audio_val_split.csv"

    train = pd.read_csv(train_files)
    train = train[train["duration"] >= 5]

    val = pd.read_csv(val_files)
    val = val[val["duration"] >= 5]

    mapping = {"TD": 0, "DEPR": 1, "ASD": 2, "SCHZ": 3}

    train["label_id"] = train.label.replace(mapping)
    val["label_id"] = val.label.replace(mapping)

    embedding_fn_dict = get_embedding_fns()
    #########################
    ##### Binary models #####
    #########################
    if arguments.train_binary_models:
        for diagnosis in ["ASD", "DEPR", "SCHZ"]:
            msg.divider(f"Training {diagnosis}")


            train_set = train[train["origin"] == diagnosis].reset_index()
            val_set = val[val["origin"] == diagnosis].reset_index()

            mapping = {diagnosis : 0, "TD" : 1}
            train_set["label_id"] = train_set["label"].replace(mapping)
            val_set["label_id"] = val_set["label"].replace(mapping)

            for feat_set in embedding_fn_dict.keys():
                if feat_set in ["windowed_mfccs"]:
                    continue
                msg.info(f"Starting {feat_set}...")
                # setup wandb config
                run = wandb.init(
                    config=arguments,
                    project="multi-diagnosis",
                    dir=arguments.default_root_dir,
                    allow_val_change=True,
                    reinit=True,
                )

                config = run.config
                run.name = f"baseline_{diagnosis}_{feat_set}"
                config.run_name = run.name
                run.log({"feat_set": feat_set})

                # Create dataloaders, model, and trainer
                train_loader, val_loader = create_dataloaders(
                    train_set, val_set, config, embedding_fn_dict[feat_set]
                )
                model = BaselineClassifier(
                    num_classes=2,
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

                trainer.fit(
                    model, train_dataloaders=train_loader, val_dataloaders=val_loader
                )

                # Finish tracking run on wandb to start the next one
                run.finish()


    #############################
    ##### Multiclass models #####
    #############################
    if arguments.train_multiclass_models:
        msg.divider("Training multiclass models")
        for feat_set in embedding_fn_dict.keys():
            if feat_set in ["windowed_mfccs"]:
                continue
            msg.info(f"Starting {feat_set}...")
            # setup wandb config
            run = wandb.init(
                config=arguments,
                project="multi-diagnosis",
                dir=arguments.default_root_dir,
                allow_val_change=True,
                reinit=True,
            )

            config = run.config
            run.name = f"baseline_multiclass_{feat_set}"
            config.run_name = run.name
            run.log({"feat_set": feat_set})

            # Create dataloaders, model, and trainer
            train_loader, val_loader = create_dataloaders(
                train, val, config, embedding_fn_dict[feat_set]
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
