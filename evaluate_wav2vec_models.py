import os

from src.evaluation.model_evaluator import ModelEvaluator
from constants import MULTICLASS_ID2LABEL_MAPPING


from pathlib import Path

from wasabi import Printer

if __name__ == "__main__":

    msg = Printer(timestamp=True)

    MODEL_TYPE = "wav2vec"

    BASE_SPLIT_PATH = (
        Path("/work")
        / "wav2vec_finetune"
        / "data"
        / "audio_file_splits"
    )
    BASE_MODEL_PATH = Path("/work") / "wav2vec_finetune" / "wav2vec_models"

    for model_path in BASE_MODEL_PATH.iterdir():
        msg.divider(f"Working on {model_path}")

        model_name, diagnosis, augmentation = model_path.name.split("_")

        num_classes = 4 if diagnosis == "multiclass" else 2
        id2label = MULTICLASS_ID2LABEL_MAPPING if diagnosis == "multiclass" else {0: "TD", 1: diagnosis}

        for split in ["val", "test", "train"]:
            msg.info(f"Split: {split}")

            if diagnosis == "multiclass":
                data_path = BASE_SPLIT_PATH / f"audio_{split}_split.csv"
            else:
                data_path = BASE_SPLIT_PATH / "binary_splits" / f"{diagnosis}_{split}_split.csv"
            
            save_name = f"{model_path.name}_{split}.jsonl"
            if (Path("results") / save_name).exists():
                msg.info(f"{save_name} already exists. Skipping...")
                continue    

            evaluator = ModelEvaluator(
                model_type=MODEL_TYPE,
                model_path=model_path,
                data_path=data_path,
                num_classes=num_classes,
                id2label=id2label,
            )
            evaluator.evaluate_model()
            evaluator.save_to_json(save_name)

