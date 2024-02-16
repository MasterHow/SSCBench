#!/bin/bash
source /opt/conda/bin/activate
conda create --name monoscene --clone occformer
conda activate monoscene
cd /workspace/mnt/storage/shihao/MyCode-02/SSCBench/method/MonoScene
pip install -r requirements.txt
conda install -c bioconda tbb=2020.2 -y
pip install torchmetrics==0.6.0
pip install setuptools==59.5.0
pip install -e ./
