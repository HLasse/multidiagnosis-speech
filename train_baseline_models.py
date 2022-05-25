import os
from functools import partial
from typing import Tuple, Union, List

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_audiomentations
import wandb
from datasets import Dataset, concatenate_datasets, load_dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from wasabi import msg

from src.baseline_utils.baseline_pl_model import BaselineClassifier
from src.baseline_utils.dataloader import MultiDiagnosisDataset
from src.baseline_utils.embedding_fns import get_embedding_fns
from src.util import create_argparser

from pathlib import Path

def create_dataloaders(
    train_filepaths: Union[pd.Series, List],
    train_labels: Union[pd.Series, List],
    val_filepaths: Union[pd.Series, List],
    val_labels: Union[pd.Series, List],
    config,
    embedding_fn,
    augment_fn=None,
) -> Tuple[DataLoader, DataLoader]:
    training_data = MultiDiagnosisDataset(
        train_filepaths, train_labels, embedding_fn=embedding_fn, augment_fn=augment_fn
    )

    validation_data = MultiDiagnosisDataset(
        val_filepaths, val_labels, embedding_fn=embedding_fn, augment_fn=augment_fn
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
        check_val_every_n_epoch=config.check_val_every_n_epoch,
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


def label2id(col, mapping):
    return {"label_id": mapping[col]}


if __name__ == "__main__":
    # Load hyperparam config
    yml_path = os.path.join(
        os.path.dirname(__file__), "configs", "baseline_configs", "default_config.yaml"
    )

    SPLIT_PATH = Path("data") / "audio_file_splits" / "windowed_splits"
    parser = create_argparser(yml_path)
    arguments = parser.parse_args()
    msg.info(f"ARGS: {' '.join(f'{k}={v}' for k, v in vars(arguments).items())}")
    # Load data files
    train = pd.read_csv(SPLIT_PATH / "windowed_train_split.csv")
    val = pd.read_csv(SPLIT_PATH / "windowed_validation_split.csv")
    # Shuffle dataset
    train = train.sample(frac=1).reset_index(drop=True)
    
    # Prepare augmentation function
    if arguments.augmentations:
        augmenter = torch_audiomentations.utils.config.from_yaml(
            arguments.augmentations
        )
        augment_fn = partial(augmenter, sample_rate=16_000)
    else:
        augment_fn = None

    # Subset data for faster debugging
    if arguments.debug:
        msg.info(f"Debug mode: using 1000 random samples")
        train = train.sample(200)
        val = val.sample(200)

    # Get functions for making embeddings
    embedding_fn_dict = get_embedding_fns()

    #########################
    ##### Binary models #####
    #########################
    if arguments.train_binary_models:
        for diagnosis in ["ASD", "DEPR", "SCHZ"]:
            ## Prepare data, subset and make label mapping
            msg.divider(f"Training {diagnosis}")

            msg.info("Subsetting...")
            train_set = train[train["origin"] == diagnosis]
            val_set = val[val["origin"] == diagnosis]

            mapping = {diagnosis: 0, "TD": 1}
            train_set["label_id"] = train_set["label"].replace(mapping)
            val_set["label_id"] = val_set["label"].replace(mapping)

            msg.info(f"Training on {len(train_set)} samples")            
            msg.info(f"Evaluating on {len(val_set)} samples")
            # Instantiate dataloader and trainer and train
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
                    train_set["filename"].tolist(),
                    train_set["label_id"].tolist(),
                    val_set["filename"].tolist(),
                    val_set["label_id"].tolist(),
                    config,
                    embedding_fn=embedding_fn_dict[feat_set],
                    augment_fn=augment_fn,
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
        mapping = {"TD": 0, "DEPR": 1, "ASD": 2, "SCHZ": 3}
        
        # map label to idx
        train["label_id"] = train["label"].replace(mapping)
        val["label_id"] = train["label"].replace(mapping)

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
                train["filename"].tolist(),
                train["label_id"].tolist(),
                val["filename"].tolist(),
                val["label_id"].tolist(),
                config, 
                embedding_fn=embedding_fn_dict[feat_set], 
                augment_fn=augment_fn
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

            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

            # Finish tracking run on wandb to start the next one
            run.finish()
