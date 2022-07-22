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
            "baseline_DEPR_aggregated_mfccs_",
            "baseline_DEPR_token-length-std_",
        ],
        [
            "alvenir_DEPR_aug_",
            "DEPR_aelaectra-danish-electra-small-cased_lr-0.0001_wdecay-0.001_wsteps-100_nofreeze_batch-16_",
        ],
        # SCHZ
        ["baseline_SCHZ_xvector_", "baseline_SCHZ_tfidf-1000_"],
        [
            "gjallarhorn_SCHZ_aug_",
            "SCHZ_ScandiBERT_lr-1e-05_wdecay-0.001_wsteps-100_nofreeze_batch-16_",
        ],
        # ASD
        [
            "baseline_ASD_compare_",
            "baseline_ASD_dependency-distance-std_",
        ],
        [
            "alvenir_ASD_aug_",
            "ASD_paraphrase-multilingual-MiniLM-L12-v2_lr-1e-05_wdecay-0.001_wsteps-100_nofreeze_batch-16_",
        ],
        # Multiclass
        [
            "baseline_multiclass_xvector_",
            "baseline_multiclass_tfidf-100_",
        ],
        [
            "alvenir_multiclass_aug_",
            "multiclass_paraphrase-multilingual-MiniLM-L12-v2_lr-1e-05_wdecay-0.001_wsteps-100_nofreeze_batch-16_",
        ],
    ]

    performance = []
    for split in ["train", "val", "test"]:

        for i, (pair, c) in enumerate(
            zip(
                ensemble_pairs,
                [
                    "DEPR",
                    "DEPR",
                    "SCHZ",
                    "SCHZ",
                    "ASD",
                    "ASD",
                    "multiclass",
                    "multiclass",
                ],
            )
        ):
            # get file path
            p = [res_path / (f + split + ".jsonl") for f in pair]
            # make ensemble model
            ensemble = ensemble_models(p, final_agg_fun="max")
            # add metadata
            model_type = "baseline" if i % 2 == 0 else "transformer"
            ensemble["model_name"] = f"{model_type}_{c}_ensemble"
            ensemble["target_class"] = c
            ensemble["is_baseline"] = 1 if i % 2 == 0 else 0
            ensemble["type"] = "ensemble"
            # save to file
            ensemble.to_json(res_path / f"ensemble_{c}_{model_type}_{split}.jsonl", orient="records", lines=True)
            # calculate performance
            if c != "multiclass":
                id2label = {0: "TD", 1: c}
            else:
                id2label = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

            perf = performance_metrics_from_df(ensemble, id_col="id", id2label=id2label)
            performance.append(perf)
    performance = pd.concat(performance)
    performance.to_json("ensemble_performance_max.jsonl", orient="records", lines=True)
