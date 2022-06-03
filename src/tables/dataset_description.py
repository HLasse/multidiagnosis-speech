import pandas as pd
import numpy as np
from pathlib import Path

def add_hline(latex: str, index: int) -> str:
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines).replace('NaN', '')



def count_words(text):
    return len(text.split())

DATA_DIR = Path.cwd() / "data"
# DATA_DIR = Path("/work") / "projectFiles" / "wav2vec_finetune" / "data"


# Duration/number  of audio
audio_splits = [pd.read_csv(f) for f in (DATA_DIR / "audio_file_splits").glob("*split.csv")]
audio = pd.concat(audio_splits)

audio_count_unique = audio.groupby(["origin", "label"])["id"].describe()[["unique", "count"]].rename(columns={"count" : "N recordings", "unique" : "N participants"})

audio_duration = audio.groupby(["origin", "label"])["duration"].agg(["mean", "sum"]).assign(sum = lambda x: np.round(x["sum"] / 60 / 60, 1)).rename(columns={"mean" : "Mean duration (s)", "sum" : "Total duration (h)"})

audio_table = audio_count_unique.join(audio_duration)

audio_table.index.set_names(["Dataset", "Diagnosis"], inplace=True)

## Add totals
multi_index = pd.MultiIndex.from_tuples([("Total", "")], names=['Dataset','Diagnosis'])

totals = pd.DataFrame({
    "N participants" : audio_table["N participants"].sum(),
    "N recordings" : audio_table["N recordings"].sum(),
    "Mean duration (s)" : np.average(audio_table["Mean duration (s)"], weights=audio_table["N recordings"]),
    "Total duration (h)" : audio_table["Total duration (h)"].sum()},
    index=multi_index
)

audio_table = pd.concat([audio_table, totals])
audio_table["Mean duration (s)"] = audio_table["Mean duration (s)"].round(1)


print(add_hline(audio_table.to_latex(), 1))



# Number of participants with both audio and transcripts

# Lengths of transcripts
transcripts = pd.read_csv(DATA_DIR / "transcripts" / "full.csv") 



transcript_count_unique = transcripts.groupby(["Group", "Diagnosis"])["ID"].describe()[["unique", "count"]].rename(columns={"count" : "N transcripts", "unique" : "N participants"})
transcripts["n_words"] = transcripts["Transcript"].apply(count_words)
transcript_duration = transcripts.groupby(["Group", "Diagnosis"])["n_words"].agg(["mean", "sum"]).rename(columns={"mean" : "Mean number of words", "sum" : "Total number of words"})

transcript_table = transcript_count_unique.join(transcript_duration)
transcript_table.index.set_names(["Dataset", "Diagnosis"], inplace=True)

totals = pd.DataFrame({
    "N participants" : transcript_table["N participants"].sum(),
    "N transcripts" : transcript_table["N transcripts"].sum(),
    "Mean number of words" : np.average(transcript_table["Mean number of words"], weights=transcript_table["N transcripts"]),
    "Total number of words" : transcript_table["Total number of words"].sum()},
    index=multi_index
)

transcript_table = pd.concat([transcript_table, totals])
transcript_table["Mean duration (s)"] = transcript_table["Mean number of words"].round(1)


print(add_hline(transcript_table.to_latex(), 1))
