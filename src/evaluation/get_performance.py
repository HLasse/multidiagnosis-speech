import pandas as pd
from pathlib import Path

from psycopmlutils.model_comparison import performance_metrics_from_folder


def add_target_class(filename: Path):
    df = pd.read_json(filename, orient="records", lines=True)

    target_class = filename.name.split("_")[1]
    is_baseline = 1 if filename.name.split("_")[0] == "baseline" else 0
    model_name = "_".join(filename.stem.split("_")[:-1])

    df["target_class"] = target_class
    df["is_baseline"] = is_baseline
    df["model_name"] = model_name

    df.to_json(filename, orient="records", lines=True)


if __name__ == "__main__":
    results_path = Path("results")

    for f in results_path.iterdir():
        add_target_class(f)

    metadata_cols = [
        "model_name",
        "split",
        "type",
        "binary",
        "target_class",
        "is_baseline",
    ]

    dfs = []
    # binary
    for diagnosis in ["DEPR", "ASD", "SCHZ"]:
        score_mapping = {0: "TD", 1: diagnosis}

        diag_df = performance_metrics_from_folder(
            path=results_path,
            pattern=f"*{diagnosis}*.jsonl",
            id_col="id",
            id2label=score_mapping,
            metadata_cols="all",
        )
        dfs.append(diag_df)

    # multiclass
    multiclass_mapping = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

    multiclass_df = performance_metrics_from_folder(
        path=results_path,
        pattern="*multiclass*.jsonl",
        id_col="id",
        id2label=multiclass_mapping,
        metadata_cols="all",
    )
    dfs.append(multiclass_df)

    dfs = pd.concat(dfs)
    dfs.to_json("audio_performance.jsonl", orient="records", lines=True)
