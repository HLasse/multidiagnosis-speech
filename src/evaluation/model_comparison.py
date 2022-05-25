""""Tools for model comparison"""

import pandas as pd

from typing import Union, List, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


import numpy as np

from model_comparison_utils import (
    add_metadata_cols,
    aggregate_predictions,
    idx_to_class,
    get_metadata_cols,
    add_metadata_cols,
)


class ModelComparison:
    def __init__(
        self,
        id_col: Optional[str] = None,
        score_mapping: Optional[dict] = None,
        metadata_cols: Optional[List[str]] = None,
    ):
        """Methods for loading and transforming dataframes with 1 row per prediction into aggregated results.
        Expects files/dataframes to have the following columns:
            label,scores,[id_col], [optional_grouping_columns]
        Where `label` is the true label for the row, `scores` is the list output
        of a softmax layer or a float. If data is grouped by an id, specifying
        an id_col will allow the class methods to also calculate performance by
        id.

        Args:
            id_col (Optional[str]): id column in case of multiple predictions.
            score_mapping (Optional[dict]): Mapping from scores index to group (should match label). E.g. if scores [0.3, 0.6, 0.1]
                score_mapping={0:"control", 1:"depression", 2:"schizophrenia}. Not needed for binary models.
            metadata_cols (Optional[List[str]], optional): Column(s) containing metadata to add to the performance dataframe.
                Each column should only contain 1 unique value. E.g. model_name, modality..

        Returns:
            pd.Dataframe: _description_
        """
        self.id_col = id_col
        self.score_mapping = score_mapping
        if isinstance(metadata_cols, str):
            metadata_cols = [metadata_cols]
        self.metadata_cols = metadata_cols

    def transform_data_from_file(self, path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if path.suffix != ".jsonl":
            raise ValueError(
                f"Only .jsonl files are supported for import, not {path.suffix}"
            )
        df = pd.read_json(path, orient="records", lines=True)
        return self.transform_data_from_dataframe(df)

    def transform_data_from_folder(
        self, path: Union[str, Path], pattern: str = "*.jsonl"
    ) -> pd.DataFrame:
        """Loads and transforms all files matching a pattern in a folder to the long result format.
        Only supports jsonl for now.

        Args:
            path (Union[str, Path]): Path to folder.
            pattern (str, optional): Pattern to match. Defaults to "*.jsonl".

        Returns:
            pd.Dataframe: Long format dataframe with aggreagted predictions.
        """
        path = Path(path)
        dfs = [self.transform_data_from_file(p) for p in path.glob(pattern)]
        return pd.concat(dfs)

    def transform_data_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe of individual predictions into long format with
        results and optionally adds metadata. Row-level performance is identified
        by the `level` column as 'overall'. If the class was instantiated with an id_col,
        id-level performance is added and identfied via the ´level´ column as 'id'.

        Args:
            df (pd.DataFrame): Dataframe with 1 row per prediction.

        Returns:
            pd.Dataframe: Long format dataframe with aggregated predictions.
        """
        performance = self._evaluate_single_model(df, aggregate_by_id=False)
        performance["level"] = "overall"

        if self.id_col:
            # Calculate performance id and add to the dataframe
            performance_by_id = self._evaluate_single_model(df, aggregate_by_id=True)
            performance_by_id["level"] = "id"
            performance = pd.concat([performance, performance_by_id])

        if self.metadata_cols:
            # Add metadata if specified
            metadata = get_metadata_cols(df, self.metadata_cols)
            performance = add_metadata_cols(performance, metadata)
        return performance

    def _evaluate_single_model(
        self, df: pd.DataFrame, aggregate_by_id: bool
    ) -> pd.DataFrame:
        """Transforms a dataframe of individual predictions into long format with columns
        ´class´, ´score_type`, and ´value´.

        Args:
            df (pd.DataFrame): Dataframe with one prediction per row
            aggregate_by_id (bool): Whether to calculate predictions on row level or aggregate by id

        Returns:
            pd.Dataframe: _description_
        """
        if aggregate_by_id:
            df = aggregate_predictions(df, self.id_col)

        # get predicted labels
        if df["scores"].dtype != "float":
            argmax_indices = df["scores"].apply(lambda x: np.argmax(x))
            predictions = idx_to_class(argmax_indices, self.score_mapping)
        else:
            predictions = np.round(df["scores"])
        return self.compute_metrics(df["label"], predictions)

    @staticmethod
    def compute_metrics(
        labels: Union[pd.Series, List],
        predicted: Union[pd.Series, List],
    ) -> pd.DataFrame:
        """Computes performance metrics for both binary and multiclass tasks

        Arguments:
            labels {Union[pd.Series, List]} -- true class
            predicted {Union[pd.Series, List]} -- predicted class

        Returns:
            pd.DataFrame -- Long format dataframe with performance metrics
        """
        classes = sorted(set(labels))
        performance = {}

        performance["acc-overall"] = accuracy_score(labels, predicted)
        performance["f1_macro-overall"] = f1_score(labels, predicted, average="macro")
        performance["f1_micro-overall"] = f1_score(labels, predicted, average="micro")
        performance["precision_macro-overall"] = precision_score(
            labels, predicted, average="macro"
        )
        performance["precision_micro-overall"] = precision_score(
            labels, predicted, average="micro"
        )
        performance["recall_macro-overall"] = recall_score(
            labels, predicted, average="macro"
        )
        performance["recall_micro-overall"] = recall_score(
            labels, predicted, average="micro"
        )

        # TODO: requires us to pass the predicted score (e.g. 0.65) and map to the correct class
        # How much do we really care about AUC after all..?
        # if len(classes) == 2:
        #     performance["roc_auc-overall"] = roc_auc_score(labels, predicted)

        # calculate scores by class
        f1_by_class = f1_score(labels, predicted, average=None)
        precision_by_class = precision_score(labels, predicted, average=None)
        recall_by_class = recall_score(labels, predicted, average=None)

        for i, c in enumerate(classes):
            performance[f"f1-{c}"] = f1_by_class[i]
            performance[f"precision-{c}"] = precision_by_class[i]
            performance[f"recall-{c}"] = recall_by_class[i]

        # to df
        performance = pd.DataFrame.from_records([performance])
        # convert to long format
        performance = pd.melt(performance)
        # split score and class into two columns
        performance[["score_type", "class"]] = performance["variable"].str.split(
            "-", 1, expand=True
        )
        # drop unused columns and rearrange
        performance = performance[["class", "score_type", "value"]]
        return performance

    def plot_f1(self):
        "lots of fun grouping options to handle"
        pass


if __name__ == "__main__":

    # example_data = "/Users/au554730/Desktop/Projects/psycop-ml-utils/tests/test_model_comparison/agg_mfccs_eval.jsonl"
    #    df = pd.read_csv(example_data)

    #   df = df[["label", "scores", "id"]]
    # scores_mapping = {0: "TD", 1: "DEPR", 2: "SCHZ", 3: "ASD"}

    multiclass_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "scores": [
                # id 1
                [0.8, 0.1, 0.05, 0.05],
                [0.4, 0.7, 0.1, 0.1],
                # id 2
                [0.1, 0.05, 0.8, 0.05],
                [0.1, 0.7, 0.1, 0.1],
                # id 3
                [0.1, 0.1, 0.7, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                # id 4
                [0.1, 0.1, 0.2, 0.6],
                [0.1, 0.2, 0.1, 0.6],
            ],
            "label": ["ASD", "ASD", "DEPR", "DEPR", "TD", "TD", "SCHZ", "SCHZ"],
            "model_name": ["test"] * 8,
        }
    )
    scores_mapping = {0: "ASD", 1: "DEPR", 2: "TD", 3: "SCHZ"}

    model_comparer = ModelComparison(
        score_mapping=scores_mapping, id_col="id", metadata_cols="model_name"
    )

    res = model_comparer.transform_data_from_dataframe(multiclass_df)
