# Disclaimer
Code in this is folder is originally from MonoScene's [official repository](https://github.com/astra-vision/MonoScene). We have made essential modifications the code to work with SSCBench dataset. If you are using the code in this folder, please cosider citing the following paper:
```
@inproceedings{cao2022monoscene,
    title={MonoScene: Monocular 3D Semantic Scene Completion}, 
    author={Anh-Quan Cao and Raoul de Charette},
    booktitle={CVPR},
    year={2022}
}
```
MonoScene's repository is realeased under the Apache 2.0 license. We adhere to the same license for the code in this folder. Please refer to the [LICENSE](./LICENSE) file for more details.

# Environment Setup
We refer users to the official repository's [instructions](https://github.com/astra-vision/MonoScene#installation) for environment setup. 

# Usage
TODO: add instructions for running the code.

## Installation



1. Create conda environment:

```
$ source /opt/conda/bin/activate
$ conda create -y -n monoscene python=3.7
$ conda activate monoscene
```
2. This code was implemented with python 3.7, pytorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/): 

```
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the additional dependencies:

```
$ cd MonoScene/
$ pip install -r requirements.txt
```

4. Install tbb:

```
$ conda install -c bioconda tbb=2020.2
```

5. Downgrade torchmetrics to 0.6.0
```
$ pip install torchmetrics==0.6.0
```

6. Finally, install MonoScene:

```
$ pip install -e ./
```