"""Combine predictions from multiple modalities by averaging"""
import pandas as pd

from typing import List


import numpy as np


def merge_predictions(df: pd.DataFrame, group_column: str, score_column: str):
    """Aggregates

    Args:
        df (pd.DataFrame): dataframe containing a `group_column`, `score_column` and optional metadata
        group_column (str): column to aggregate predictions by
        score_column (str): column that scores are stored in. Assumes the column to be a list/array type
    """

    def agg_fun(scores: List[float]):
        array = np.array(scores.tolist())
        return np.mean(array, axis=0)

    # Aggregate by trial id
    agg_scores = df.groupby(group_column).agg({score_column: agg_fun})

    # join to the old dataframe
    df = df.drop(score_column, axis=1)
    df = df.merge(agg_scores, on=group_column)

    # remove duplicates
    df = df.drop_duplicates(subset=group_column)
    return df


if __name__ == "__main__":

    df = pd.DataFrame(
        {
            "trial_id": ["xyz_1", "xyz_2", "xyz_1", "xyz_2"],
            "modality": ["text", "text", "medication", "medication"],
            "scores": [
                [0, 0.5, 0.25, 0.25],
                [0, 0.25, 0.5, 0.25],
                [0, 0, 0.5, 0.5],
                [0, 0, 1, 0],
            ],
        }
    )

    df = merge_predictions(df, "trial_id", "scores")

    assert len(df["trial_id"].unique()) == 2
    assert df["scores"].tolist == [[0.0, 0.25, 0.375, 0.375], 0.0, 0.125, 0.75, 0.125]
