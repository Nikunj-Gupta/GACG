#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n gacg python=3.11 -y
# conda activate gacg

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y 
pip install git+https://github.com/oxwhirl/smac.git
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html # Optional dependencies: 
pip install pymongo setproctitle sacred pyyaml tensorboard_logger matplotlib 
# pip install torch_geometric
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 