#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install torch torchvision
pip install transformers==4.41.2 tokenizers==0.19.1
pip install torchmetrics
pip install pandas
pip install numpy
pip install pyyaml
pip install tqdm
pip install box
pip install scikit-learn
pip install tensorboard
pip install python-box
