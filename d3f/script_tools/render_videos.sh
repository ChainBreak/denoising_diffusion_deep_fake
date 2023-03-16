#!/usr/bin/bash
source $HOME/.bashrc
con
conda activate satfix


checkpoint=$1
videos=/home/thomas/datasets/denoising_diffusion_deep_fake
python3 put_video_through_fake_model.py $videos/tom_3.mp4 $checkpoint b 448 448
python3 put_video_through_fake_model.py $videos/georgia_3.mp4 $checkpoint a 448 448
