"""Generate train/test split"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_gender(filepath):
    """files are coded so first two characters decide gender.
    odd = man, even = woman
    code woman as 1, men as 0"""
    if int(filepath.split("/")[-1][0:2]) % 2 == 0:
        return "woman"
    return "man"


if __name__ == "__main__":
    # Only using the neutral data (no specific emotion)
    data = os.path.join(os.getcwd(), "data", "Neutre")

    filepaths = os.listdir(data)
    file_dict = [{"file": os.path.join(data, path)} for path in filepaths]

    df = pd.DataFrame(file_dict)
    df["label"] = df.file.apply(get_gender)
    df["name"] = df.file.apply(lambda x: x.split("/")[-1])

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    save_path = "preproc_data"
    train.to_csv(os.path.join(save_path, "train_data.csv"), sep="\t", index=False)
    test.to_csv(os.path.join(save_path, "test_data.csv"), sep="\t", index=False)




