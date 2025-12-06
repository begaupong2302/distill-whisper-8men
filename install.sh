#!/bin/bash
conda create --name diswhis2 python=3.11 -y
conda activate diswhis2 -y
conda install "ffmpeg" -c conda-forge -y
pip install accelerate==1.11.0 datasets==3.6.0 evaluate==0.4.6 jiwer==4.0.0 soundfile==0.13.1 transformers==4.57.1 einops==0.8.1 librosa==0.11.0
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchcodec==0.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
pip install flash-attn --no-build-isolation

python snap_download.py
create_student_model_v2.py
chmod +x script.sh
./script.sh
