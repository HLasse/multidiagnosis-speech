import os

from src.model_evaluator import ModelEvaluator


if __name__ == "__main__":

    model_type = "embedding_baseline"
    models = [
        "baseline_aggregated_mfccs",
        #    "baseline_compare",
        #    "baseline_egemaps",
        #    "baseline_xvector",
    ]
    feature_sets = [
        "aggregated_mfccs",
        #    "compare",
        #    "egemaps",
        #    "xvector"
    ]
    save_names = [
        "agg_mfccs_eval.jsonl",
        #    "compare_eval.csv",
        #    "egemaps_eval.csv",
        #    "xvector_eval.csv",
    ]

    for model, feature_set, save_name in zip(models, feature_sets, save_names):
        print(f"[INFO] Evaluating {model}")

        model_ckpts = os.path.join(os.path.dirname(__file__), "baseline_models", model)
        latest_model = os.path.join(model_ckpts, os.listdir(model_ckpts)[-1])

        evaluator = ModelEvaluator(
            model_type=model_type, model_path=latest_model, feature_set=feature_set
        )

        evaluator.evaluate_model()
        evaluator.save_to_csv(save_name)
        evaluator.print_performance()
        evaluator.print_performance_by_id()
