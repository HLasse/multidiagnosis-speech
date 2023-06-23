from pathlib import Path
from transformers import (TrainingArguments, Trainer,
                          AutoModelForSequenceClassification)
from utils_text import make_dataset
import numpy as np
import json
import argparse
from datasets import load_metric

# UP NEXT
# Missing: put inputs and models to cuda if needed (or does the trainer handle that automatically?)
# Make splits [W/ Riccardo and Lasse]
# Add other models [w/ Riccardo and Lasse]
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
parser.add_argument('--gradient-accumulation-steps', 
                    type=int, default=1,
                    help='Steps for gradient accumulation')
parser.add_argument('--num-labels', 
                    type=int, default=4,
                    help='Steps for gradient accumulation')
parser.add_argument('--problem-type', 
                    type=str, default='multi_label_classification',
                    help='Is the problem single_label_classification ' 
                         'or multi_level_classification?')


# Which metrics to compute for evaluation
def compute_metrics(pred):
    prec, rec, f1 = (load_metric(m) 
                     for m in ('precision', 'recall', 'f1'))
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    precision = prec.compute(predictions=predictions, references=labels)["precision"]
    recall = rec.compute(predictions=predictions, references=labels)["recall"]
    f1_score = f1.compute(predictions=predictions, references=labels)["f1"]
    return {"precision": precision, "recall": recall, "f1": f1_score}


# Training module
def _make_trainer(model_id,
                  checkpoint, 
                  train_dataset, val_dataset,
                  epochs, 
                  train_examples_per_device, 
                  eval_examples_per_device,
                  warmup_steps, 
                  weight_decay,
                  logging_steps,
                  gradient_accumulation_steps,
                  num_labels,
                  problem_type):
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

    # Make directories
    logpath = Path('logs') / f'{model_id}'
    respath = Path('models') / f'{model_id}'
    logpath.mkdir(exist_ok=True, parents=True)
    respath.mkdir(exist_ok=True, parents=True)

    # Label mapping
    ldict = json.load(open('data/transcripts/labels.json')) # hard-coded right now...
    if 'multi' in problem_type: # seems like multiclass expects float labels?
        ldict = {lab:float(id) for lab,id in ldict.items()}
    rev_ldict = dict(zip(ldict.values(), ldict.keys()))

    # Set up trainer
    training_args = TrainingArguments(
        output_dir=str(logpath),      
        num_train_epochs=epochs,
        per_device_train_batch_size=train_examples_per_device,
        per_device_eval_batch_size=eval_examples_per_device,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=str(respath),
        logging_steps=logging_steps,
        evaluation_strategy='epoch',
        gradient_accumulation_steps=gradient_accumulation_steps,
        run_name=model_id,
        )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=num_labels, 
                                                               id_to_label=rev_ldict,
                                                               label_to_id=ldict,
                                                               problem_type=problem_type)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        ) 
    
    return trainer


# Exec
if __name__=='__main__':
    args = parser.parse_args()
    train_ds, val_ds = (make_dataset(args.checkpoint, s) 
                        for s in ['train', 'val'])
    trainer = _make_trainer(args.model_id,
                            args.checkpoint,
                            train_ds, 
                            val_ds, 
                            args.epochs, 
                            args.train_examples_per_device, 
                            args.eval_examples_per_device,
                            args.warmup_steps,
                            args.weight_decay,
                            args.logging_steps,
                            args.gradient_accumulation_steps,
                            args.num_labels)
    trainer.train()
    