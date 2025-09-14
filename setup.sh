wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n stress python=3.9
conda activate stress
bash install.sh
cd TrainModel
python3 train.py