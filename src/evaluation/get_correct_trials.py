from pathlib import Path

import pandas as pd
from psycopmlutils.model_performance import ModelPerformance

if __name__ == "__main__":
    models = ["baseline_multiclass_xvector", "baseline_multiclass_tfidf-100"]

    multiclass_mapping = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

    for model in models:
        p = Path("results") / "combined_results"
        p = p / f"{model}_test.jsonl"
        df = pd.read_json(p, orient="records", lines=True)
        df = ModelPerformance.performance_metrics_from_df(
            prediction_df=df,
            prediction_col_name="scores",
            label_col_name="label",
            to_wide=False,
            id2label=multiclass_mapping,
        )
        df.to_json(
            p.parent / f"trial_level_{model}.jsonl", orient="records", lines=True
        )
