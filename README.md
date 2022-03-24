# wav2vec_finetune

Test finetuning of XLSR (multilingual wav2vec 2.0) for classification of mental disorders from speech.


```
# make virtual env
pip install -r requirements.txt

mkdir data
# download and unzip data to 'data/multi_diagnosis'
wget ...
unzip ..
# make sure metadata file (CleanData3.csv) is in 'data/multidiagnosis.
# if only access to CleanData.csv, run preprocessing/merge_participant_metadata.py

# run preprocessing scripts
bash run_preprocessing.py

# train model from config file
python train_wav2vec path_to_config_file

# evaluate model from config file
python evaluate_wav2vec path_to_config_file

```

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
- https://github.com/agemagician/wav2vec2-sprint/blob/main/run_common_voice.py
- https://github.com/huggingface/transformers/tree/master/examples/research_projects/robust-speech-event
- https://github.com/talhanai/speech-nlp-datasets

## Audio augmentation
- Using torch-audiomentation
- IR taking from [MIT Acoustical Reverberation Scene Statistics Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html). Placed in 'augmentation_files/ir'. Specificy path in 'augmentation_config.yml'
- 
