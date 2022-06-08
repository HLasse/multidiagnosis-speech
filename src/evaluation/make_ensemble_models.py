from ensembler import ensemble_models
from pathlib import Path

import pandas as pd

from psycopmlutils.model_comparison import performance_metrics_from_df

if __name__ == "__main__":
    res_path = Path("results") / "combined_results"

    ### Best performing text baseline paired with best performing audio baseline
    ### Best performing text transformer paired with best performing audio transformer
    ensemble_pairs = [
        # DEPR
        [
            "baseline_DEPR_egemaps_test.jsonl",
            "baseline_DEPR_token-length-std_test.jsonl",
        ],
        [
            "alvenir_DEPR_no-aug_test.jsonl",
            "DEPR_aelaectra-danish-electra-small-cased_lr-0.0001_wdecay-0.001_wsteps-100_nofreeze_batch-16_test.jsonl",
        ],
        # SCHZ
        ["baseline_SCHZ_xvector_test.jsonl", "baseline_SCHZ_tfidf-1000_test.jsonl"],
        [
            "alvenir_SCHZ_aug_test.jsonl",
            "SCHZ_ScandiBERT_lr-1e-05_wdecay-0.001_wsteps-100_nofreeze_batch-16_test.jsonl",
        ],
        # ASD
        [
            "baseline_ASD_xvector_test.jsonl",
            "baseline_ASD_dependency-distance-std_test.jsonl",
        ],
        [
            "xls-r_ASD_aug_test.jsonl",
            "ASD_paraphrase-multilingual-MiniLM-L12-v2_lr-1e-05_wdecay-0.001_wsteps-100_nofreeze_batch-16_test.jsonl",
        ],
        # Multiclass
        [
            "baseline_multiclass_aggregated_mfccs_test.jsonl",
            "baseline_multiclass_tfidf-100_test.jsonl",
        ],
        [
            "alvenir_multiclass_aug_test.jsonl",
            "multiclass_paraphrase-multilingual-MiniLM-L12-v2_lr-1e-05_wdecay-0.001_wsteps-100_nofreeze_batch-16_test.jsonl",
        ],
    ]

    performance = []
    for i, (pair, c) in enumerate(
        zip(
            ensemble_pairs,
            ["DEPR", "DEPR", "SCHZ", "SCHZ", "ASD", "ASD", "multiclass", "multiclass"],
        )
    ):
        # get file path
        p = [res_path / f for f in pair]
        # make ensemble model
        ensemble = ensemble_models(p)
        # add metadata
        model_type = "baseline" if i % 2 == 0 else "transformer"
        ensemble["model_name"] = f"{model_type}_{c}_ensemble"
        ensemble["target_class"] = c
        ensemble["is_baseline"] = 1 if i % 2 == 0 else 0

        # calculate performance
        if c != "multiclass":
            id2label = {0: "TD", 1: c}
        else:
            id2label = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

        perf = performance_metrics_from_df(ensemble, id_col="id", id2label=id2label)
        performance.append(perf)
    performance = pd.concat(performance)
    performance.to_json("ensemble_performance.jsonl", orient="records", lines=True)
