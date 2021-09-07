# wav2vec_finetune

```
mkdir data
mkdir json_files
cd data
wget https://zenodo.org/record/1219621/files/CaFE_48k.zip?download=1
unzip 
```



Test finetuning of XLSR (multilingual wav2vec 2.0) for other speech classification tasks
Initial test: gender recognition on [this](https://zenodo.org/record/1219621#.YTcmxS2w0ws) dataset.

Resources: 
- https://github.com/pytorch/fairseq/blob/master/examples/xlmr/README.md
- https://arxiv.org/abs/2006.13979
- https://huggingface.co/transformers/training.html
- https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
- https://discuss.huggingface.co/t/german-asr-fine-tuning-wav2vec2/4558/5
- https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files
- https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/FINE_TUNE_XLSR_WAV2VEC2.md

Notes:
- Look into SpecAugment for finetuning: https://arxiv.org/abs/1904.08779
- How to make the prediction? 
  - Feed sequence of length N to LSTM?
  - Single FF layer on top - one prediction for each timestep (how long is a timestep?)


TODO:

Preprocessing:
- Only include the neutral emotion (Neutre folder)
- Preprocess data to be correct input format (16khz)
- Format to huggingface dataset loader: {"file" : filepath_i, "label" : gender_i}
  
Training:
- Determine length of timestep
- Investigate layers
- Make loss function (binary cross-entropy?)
- Train!