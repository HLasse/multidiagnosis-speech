"""Evaluate model from cmd line.
Example:
python evaluate_cli.py --model_name baseline_aggregated_mfccs --feature_set aggregated_mfccs"""

import os
import fire
from src.model_evaluator import ModelEvaluator
from wasabi import msg


def evaluate(model_name, feature_set, split: "val"):

    split_path = f"data/audio_file_splits/audio_{split}_split.csv"

    msg.text(f"Evaluating {model_name} with {feature_set} features", spaced=True)

    model_ckpts = os.path.join(os.path.dirname(__file__), "baseline_models", model_name)
    latest_model = os.path.join(model_ckpts, os.listdir(model_ckpts)[-1])

    evaluator = ModelEvaluator(
        model_type="embedding_baseline",
        model_path=latest_model,
        feature_set=feature_set,
        data_path=split_path,
    )

    evaluator.evaluate_model()
    evaluator.print_performance()
    evaluator.print_performance_by_id()


if __name__ == "__main__":

    fire.Fire(evaluate)
