import pandas as pd
from datasets import TextDataset
from transformers import AutoTokenizer

def _get_data(df, ids):
    sub_df = df[df['Subject'].isin(ids)]
    lst = sub_df[['Transcript', 'Diagnosis']].to_records()
    return zip(*lst)

def make_dataset(checkpoint, split='train'):
    ''' Make dataset from transcripts and train / val ids 
    Args:
        checkpoint: model checkpoint
    '''
    DPATH = Path('data') / 'transcripts' / 'processed'
    ids = pd.read_csv(DPATH/f'{which}_ids.txt') 
    data = pd.read_csv(DPATH/'data.csv')
    txt, lab = _get_data(data, ids)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    enc = tokenizer(txt, truncation=True, padding=True)
    dataset = TextDataset(enc, lab)
    return train_dataset, val_dataset, test_dataset