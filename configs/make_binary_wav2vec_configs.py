import json
from pathlib import Path

BASE_CONFIG = {
    "model_name": "",
    "train": "",
    "validation": "",
    "augmentations": "",
    "input_col": "file",
    "label_col": "label",
    "use_windowing": True,
    "window_length": 4,
    "stride_length": 1.0,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.1,
    "final_dropout": 0.1,
    "feat_proj_dropout": 0.2,
    "mask_time_prob": 0.05,
    "layerdrop": 0.1,
    "gradient_checkpointing": True,
    "ctc_loss_reduction": "sum",
    "freeze_encoder": True,
    "freeze_base_model": True,
    "output_dir": "",
    "run_name": "",
    "num_train_epochs": 20,
    "per_device_train_batch_size": 32,
    "learning_rate": 1e-5,
    "evaluation_strategy": "epoch",
    "group_by_length": True,
    "gradient_accumulation_steps": 2,
    "fp16": True,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "save_strategy": "epoch",
}


def make_config(
    model_name: str,
    train: str,
    validation: str,
    augmentations: str,
    output_dir: str,
    run_name: str,
):
    config = BASE_CONFIG.copy()
    config["model_name"] = model_name
    config["train"] = train
    config["validation"] = validation
    config["augmentations"] = augmentations
    config["output_dir"] = output_dir
    config["run_name"] = run_name
    return config


if __name__ == "__main__":
    CONFIG_DIR = Path("configs") / "wav2vec_configs"
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir()
    BASE_OUTPUT_DIR = Path("wav2vec_models")
    classes = ["multiclass", "ASD", "DEPR", "SCHZ"]

    augmentations = ["configs/augmentation_config.yml", ""]

    model_types = ["Alvenir/wav2vec-base-da", "facebook/wav2vec2-xls-r-300m"]

    for c in classes:

        if c != "multiclass":
            train = f"data/audio_file_splits/binary_splits/{c}_train_split.csv"
            validation = f"data/audio_file_splits/binary_splits/{c}_val_split.csv"
        else:
            train = "data/audio_file_splits/audio_train_split.csv"
            validation = "data/audio_file_splits/audio_val_split.csv"

        for aug in augmentations:
            if aug:
                aug_run_name = "aug"
            else:
                aug_run_name = "no-aug"
            for model in model_types:
                if "base-da" in model:
                    model_run_name = "alvenir"
                elif "xls-r" in model:
                    model_run_name = "xls-r"

                run_name = f"{model_run_name}_{c}_{aug_run_name}"
                output_dir = str(BASE_OUTPUT_DIR / run_name)

                config = make_config(
                    model, train, validation, aug, output_dir, run_name
                )

                file_name = CONFIG_DIR / f"{run_name}.json"
                with open(file_name, "w") as f:
                    json.dump(config, f)
