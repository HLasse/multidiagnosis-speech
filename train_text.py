import pandas as pd
from pathlib import Path
from transformers import (Trainer, TrainingArguments,
                          AutoModelForSequenceClassification, 
                          AutoTokenizer)
from datasets import TextDataset
import argparse

# TO DO:
# Make splits [W/ Riccardo and Lasse]
# Add other models [w/ Riccardo and Lasse]
# Make category-label dict - Check multiclass
# Add other arguments to trainer and cmdline - See Wav2Vec
# Make compute metrics function - See Wav2Vec
# Set up WandB - See Wav2Vec
# Bash script [When everything else is ready]

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default=None,
                    help='Unique model name')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Model checkpoint to train from')
parser.add_argument('--epochs', type=int, default=3,
                    help='Training epochs')
parser.add_argument('--train-examples-per-device', 
                    type=int, default=16,
                    help='Training examples per device at training')
parser.add_argument('--eval-examples-per-device', 
                    type=int, default=64,
                    help='Training examples per device at evaluation')
parser.add_argument('--warmup-steps', 
                    type=int, default=500,
                    help='Number of warmup steps')
parser.add_argument('--weight-decay', 
                    type=float, default=0.001,
                    help='Weight decay')
parser.add_argument('--logging-steps', 
                    type=int, default=10,
                    help='Log every')

# Dataset creation
def _get_data(df, ids):
    sub_df = df[df['Subject'].isin(ids)]
    lst = sub_df[['Transcript', 'Diagnosis']].to_records()
    return zip(*lst)


def _make_dataset(checkpoint):
    ''' Make dataset from transcripts and train / val ids 
    Args:
        checkpoint: model checkpoint
    '''
    DPATH = Path('data') / 'transcripts' / 'processed'
    train_ids = pd.read_csv(DPATH/'train_ids.txt') 
    train_ids = pd.read_csv(DPATH/'val_ids.txt') 
    data = pd.read_csv(DPATH/'data.csv')
    train_txt, train_lab = _get_data(data, train_ids)
    val_txt, val_lab = _get_data(data, train_ids)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_enc = tokenizer(train_txt, truncation=True, padding=True)
    val_enc = tokenizer(val_txt, truncation=True, padding=True)

    train_dataset = TextDataset(train_enc, train_lab)
    val_dataset = TextDataset(val_enc, val_lab)
    return train_dataset, val_dataset


def _make_trainer(model_id,
                  checkpoint, 
                  train_dataset, val_dataset,
                  epochs, 
                  train_examples_per_device, 
                  eval_examples_per_device,
                  warmup_steps, 
                  weight_decay,
                  logging_steps):
    ''' Train model 
    Args:
        model_id: unique model name
        checkpoint: model checkpoint
        train_dataset: training dataset
        val_dataset: validation dataset
        epochs: training epochs
        train_examples_per_device: examples per device at training
        eval_examples_per_device: examples per device at eval
        warmup_steps: nr optimizer warmup steps
        weight_decay: weight decay for adam optimizer
        logging_steps: how often to log
    '''

    training_args = TrainingArguments(
        output_dir=f'./results/{model_id}',  # does it create it?        
        num_train_epochs=epochs,
        per_device_train_batch_size=train_examples_per_device,
        per_device_eval_batch_size=eval_examples_per_device,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f'./logs/{model_id}', # does it create it?
        logging_steps=logging_steps,
        # Add other and pass as cmdline args
        )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
        ) 
    
    return trainer


def _compute_metrics():
    # TO DO
    pass


if __name__=='__main__':
    args = parser.parse_args()
    train_ds, val_ds = _make_dataset(args.checkpoint)
    trainer = _make_trainer(args.model_id,
                            args.checkpoint,
                            train_ds, val_ds, 
                            args.epochs, 
                            args.train_examples_per_device, 
                            args.eval_examples_per_device,
                            args.warmup_steps,
                            args.weight_decay,
                            args.logging_steps)
    trainer.train()
    # Compute metrics
    
