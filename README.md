# Multidiagnosis-speech

Code for finetuning of feature-based baselines and Transformer models for classification of mental disorders from speech.

```
# make virtual env
pip install -r requirements.txt

mkdir data
# download and unzip data to 'data/multi_diagnosis'
wget ...
unzip ..
# make sure metadata file (CleanData4.csv) is in 'data/multidiagnosis.
# if only access to CleanData.csv, run preprocessing/merge_participant_metadata.py

# run preprocessing scripts
bash run_preprocessing.py

# train baseline models
python train_baseline_models.py

# train wav2vec models
bash train_wav2vec.sh

#evaluate models
bash evaluate_baselines.sh
python evaluate_wav2vec_models.py

```

## Audio augmentation
- Using torch-audiomentation
- IR taking from [MIT Acoustical Reverberation Scene Statistics Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html). Placed in 'augmentation_files/ir'. Specificy path in 'augmentation_config.yml'
