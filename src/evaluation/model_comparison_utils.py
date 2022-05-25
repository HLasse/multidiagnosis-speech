from typing import List, Union
import pandas as pd

import numpy as np


def aggregate_predictions(df: pd.DataFrame, id_col: str):
    """Calculates the mean prediction by a grouping col (id_col).
    Assumes that df has the columns 'scores': List[float] and
    'label' : str

    Args:
        df (pd.DataFrame): Dataframe with 'scores', 'label' and id_col columns
        id_col (str): Column to group by
    """

    def mean_scores(x: pd.Series):
        gathered = np.stack(x)
        return gathered.mean(axis=0)

    def get_first_entry(x: pd.Series):
        return x.unique()[0]

    return df.groupby(id_col).agg({"scores": mean_scores, "label": get_first_entry})


def idx_to_class(idx: List[int], mapping: dict):
    return [mapping[id] for id in idx]


def get_metadata_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Extracts model metadata and generates a dataframe with same m

    Args:
        df (pd.DataFrame): Dataframe with predictions and metadata.
        cols (List[str]): Which columns contain metadata.
            The columns should only contain a single value.

    Raises:
        ValueError: If a metadata col contains more than a single unique value.

    Returns:
        pd.DataFrame: 1 row dataframe with metadata
    """

    metadata = {}
    all_columns = df.columns
    for col in cols:
        if col in all_columns:
            val = df[col].unique()
            if len(val) > 1:
                raise ValueError(
                    f"The column '{col}' contains more than one unique value."
                )
            metadata[col] = val[0]
    return pd.DataFrame.from_records([metadata])


def add_metadata_cols(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Adds 1 row dataframe with metadata to the long format performance dataframe

    Args:
        df (pd.DataFrame): Dataframe to add metadata to.
        metadata (pd.Dataframe): 1 row dataframe with metadata

    Returns:
        pd.DataFrame: Dataframe with added metadata
    """
    nrows = df.shape[0]

    meta_dict = {}
    for col in metadata.columns:
        meta_dict[col] = [metadata[col][0]] * nrows
    meta_df = pd.DataFrame.from_records(meta_dict)

    return df.reset_index(drop=True).join(meta_df)


def string_to_list(str_or_list: Union[List, str]):
    if isinstance(str_or_list, str):
        return [str_or_list]
    elif isinstance(str_or_list, list):
        return str_or_list
    else:
        raise ValueError(f"{str_or_list} is neither a string nor list")


def subset_df_from_dict(df: pd.DataFrame, subset_by: dict):
    for col, value in subset_by.items():
        df = df[df[col] == value]
    return df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
            ],
            "label": ["ASD", "ASD", "TD", "TD"],
        }
    )

    aggregate_predictions(df, "id")
