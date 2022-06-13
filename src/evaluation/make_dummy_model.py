"""Make dummy model that always predicts TD as a baseline comparison"""
from psycopmlutils.model_comparison import performance_metrics_from_df

import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    base_path = Path("results") 

    id2label = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

    dfs = []
    for split in ["train", "val", "test"]:
        dummy_df = pd.read_json(base_path / f"alvenir_multiclass_aug_{split}.jsonl", orient="records", lines=True)
        all_td_scores = pd.Series([[1,0,0,0]] * dummy_df.shape[0], index=dummy_df.index)
        dummy_df["scores"] = all_td_scores
        dummy_df["model_name"] = "dummy_baseline"
        dummy_df["type"] = "baseline"

        perf = performance_metrics_from_df(dummy_df, id_col="id", id2label=id2label)
        dfs.append(perf)
    dfs = pd.concat(dfs)
    dfs.to_json("dummy_baseline.jsonl", orient="records", lines=True)
