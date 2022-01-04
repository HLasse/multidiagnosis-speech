"""Creates a subsample of the training files to test models on before fitting the whole dataset"""

import pandas as pd

from pathlib import Path

if __name__ == "__main__":

    split_dir = Path("data") / "audio_file_splits"

    df = pd.read_csv(split_dir / "audio_test_split.csv")
    df = df.sample(5)

    df.to_csv(split_dir / "dummy_train_set.csv", index=False)
