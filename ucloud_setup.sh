sudo apt-get update --allow-releaseinfo-change
sudo apt-get install -y libsndfile1

sudo apt install -y tmux
echo "set -g mouse on" >> ~/.tmux.conf # enable scrolling

sudo apt-get install libsox-fmt-all -y
sudo apt-get install sox -y

pip install sox
pip install soxbindings

pip install -r requirements.txt

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html