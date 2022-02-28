#!/bin/bash

mkdir plots
mkdir gifs
mkdir data

conda create -y -n owl python=3.6
source activate owl

pip install gym
pip install gym-minigrid

conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install -y tensorboard
conda install -y moviepy
conda install -y array2gif
conda install -y matplotlib
conda install -y pandas