English | [中文](installation_guide_CN.md)

The following assumes that you have conda and walks you through creating the required conda environment. You can install conda from the following link (https://docs.anaconda.com/anaconda/install/). 
Please note that since RDkit is not guaranteed to run on the latest version of python, our suggestion is that you use python 3.6 or earlier. 

1. Create and activate a new python environment and install pip:
````
conda create --name p36_repurposing python=3.6 pip
````
````
conda activate p36_repurposing
````

2. Install torch using conda
````
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
````

3. Install using pip required packages. Note that versioned requirements are being used here but any recent version should work:
````
pip install -r requirements.txt --use-feature=2020-resolver
````

4. Install RDkit from the RDkit channel:
````
conda install -c rdkit rdkit
````


1. To install pytorch geometric with CUDA support, please visit https://data.pyg.org/whl/torch-1.6.0+cu101.html
Find the .whl files that match your environment (for example, Windows, Python 3.6, CUDA 10.1), such as:
```
torch_scatter-2.0.5+cu101-cp36-cp36m-win_amd64.whl
```

After downloading, install with:
````
pip install torch_scatter-2.0.5+cu101-cp36-cp36m-win_amd64.whl
pip install torch_sparse-0.6.9-cp36-cp36m-win_amd64.whl
pip install torch_cluster-1.5.9-cp36-cp36m-win_amd64.whl
pip install torch_spline_conv-1.2.1-cp36-cp36m-win_amd64.whl
pip install torch-geometric
````