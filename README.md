# wav2vec_finetune

Test finetuning of XLSR (multilingual wav2vec 2.0) for other speech classification tasks

- [X] Initial test: gender recognition on [this](https://zenodo.org/record/1219621#.YTcmxS2w0ws) dataset.
- [X] Finetune for autism detection
- [] Clean up directory
- [] Make training and evaluation scripts runnable with cmd line / shell scripts
- [] Add random noise on training samples
- [] Make baseline models
```
# make virtual env
pip install -r requirements.txt

mkdir data
mkdir preproc_data
mkdir model
cd data
wget https://zenodo.org/record/1219621/files/CaFE_48k.zip?download=1
unzip the file 

python preproc.py
python train.py
python evaluate.py
```

## Updates
- 11/9: success! Trained a sex classifier on a small dataset that performs soso. Everything seems to work though.

## TODO
- Chunk audio files - make predictions in batches of e.g. 5 seconds
- Set up benchmark models

## Resources: 
- https://github.com/pytorch/fairseq/blob/master/examples/xlmr/README.md
- https://arxiv.org/abs/2006.13979
- https://huggingface.co/transformers/training.html
- https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
- https://discuss.huggingface.co/t/german-asr-fine-tuning-wav2vec2/4558/5
- https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files
- https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/FINE_TUNE_XLSR_WAV2VEC2.md
- https://github.com/m3hrdadfi/soxan
- https://www.zhaw.ch/storage/engineering/institute-zentren/cai/BA21_Speech_Classification_Reiser_Fivian.pdf
- https://github.com/DReiser7/w2v_did
- https://github.com/ARBML/klaam
- https://github.com/talhanai/speech-nlp-datasets

## Notes:
- Look into SpecAugment for finetuning: https://arxiv.org/abs/1904.08779 (on by default)
- How to make the prediction? 
  - Better way than a small feedforward projection? (LSTM or something?)
