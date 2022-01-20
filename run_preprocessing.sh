echo "Running make_train_val_test_split.py"
python preprocessing/make_train_val_test_split.py
echo "Running preproc_audio.py"
python preprocessing/preproc_audio.py
echo "Running make_dummy_audio_files.py"
python preprocessing/make_dummy_audio_files.py
echo "Running make_binary_prediction_splits.py"
python preprocessing/make_binary_prediction_splits.py