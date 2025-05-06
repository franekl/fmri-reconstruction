#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

# Check for python3.11.9
if ! command -v python3.11.9 &> /dev/null
then
    echo "python3.11.9 could not be found. Please install Python3.11.9"
    exit 1
fi

pip install --upgrade pip

python3.11 -m venv fmri
source fmri/bin/activate

pip install numpy matplotlib==3.8.2 jupyter jupyterlab_nvdashboard jupyterlab tqdm scikit-image==0.22.0 accelerate==0.24.1 webdataset==0.2.73 pandas==2.2.0 einops ftfy regex kornia==0.7.1 h5py==3.10.0 open_clip_torch==2.24.0 torchvision==0.16.0 torch==2.1.0 transformers==4.37.2 xformers==0.0.22.post7 torchmetrics==1.3.0.post0 diffusers==0.23.0 deepspeed==0.13.1 wandb omegaconf==2.3.0 pytorch-lightning==2.0.1 sentence-transformers==2.5.1 evaluate==0.4.1 nltk==3.8.1 rouge_score==0.1.2 umap==0.1.1
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch

# run htop to check ram, gpu and cpu