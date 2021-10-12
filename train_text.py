import pandas as pd
from pathlib import Path
from transformers import Trainer, TrainingArguments
from datasets import TextDataset

# TO DO:
# Make splits
# Make category-label dict
# Make checkpoint -> model/tokenizer dict (pick DK/multilang models w/ Lasse)
# Add other arguments to trainer (see wav2vec script)
# Make compute metrics function
# Implement parser and cmdline args
# Make bash scripts

def _get_data(df, ids):
    sub_df = df[df['Subject'].isin(ids)]
    lst = sub_df[['Transcript', 'Diagnosis']].to_records()
    return zip(*lst)

def _compute_metrics():
    # TO DO
    pass

def _make_dataset(tokenizer, checkpoint):
    ''' Make a dataset '''
    DPATH = Path('data') / 'transcripts' / 'processed'

    train_ids = pd.read_csv(DPATH/'train_ids.txt') 
    train_ids = pd.read_csv(DPATH/'val_ids.txt') 
    data = pd.read_csv(DPATH/'data.csv')

    train_txt, train_lab = _get_data(data, train_ids)
    val_txt, val_lab = _get_data(data, train_ids)

    tokenizer = tokenizer.from_pretrained(checkpoint)
    train_enc = tokenizer(train_txt, truncation=True, padding=True)
    val_enc = tokenizer(val_txt, truncation=True, padding=True)

    train_dataset = TextDataset(train_enc, train_lab)
    val_dataset = TextDataset(val_enc, val_lab)
    return train_dataset, val_dataset


def _make_trainer(model, checkpoint, train_dataset, val_dataset):
    ''' Train model 
    Args:
        model: which model to train
        checkpoint: from which checkpoint
        train_dataset: training dataset
        val_dataset: validation dataset
    '''

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        # Check other
        )

    model = model.from_pretrained(checkpoint)
    trainer = Trainer(
        model=model,                         # the instantiated transformer
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
        ) 
    
    return trainer


if __name__=='__main__':
    # TO DO: get command line args
    train_ds, val_ds = _make_dataset()
    trainer = _make_trainer()
    trainer.train()
    # compute metrics
    
