sudo apt-get update --allow-releaseinfo-change
sudo apt-get install -y libsndfile1
sudo apt install libpython3.9-dev

sudo apt install -y tmux
echo "set -g mouse on" >> ~/.tmux.conf # enable scrolling

sudo apt-get install libsox-fmt-all -y
sudo apt-get install sox -y
sudo apt-get install python3.9-tk -y


pip install sox
pip install soxbindings

pip install pip --upgrade
pip install -r requirements.txt

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
git config --global user.email "lasseh0310@gmail.com"
git config --global user.name "Lasse Hansen"

# symbolic link from python3 to python
sudo ln -sf /usr/bin/python3 /usr/bin/python

# maybe: pip install -upgrade importlib-metadata if opensmile errors