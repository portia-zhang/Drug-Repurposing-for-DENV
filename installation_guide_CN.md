[English](installation_guide.md) | 中文

以下假设你已安装 conda，并将指导你创建所需的 conda 环境。你可以通过以下链接安装 conda（https://docs.anaconda.com/anaconda/install/）。
请注意，由于 RDkit 并不保证支持最新版本的 python，建议使用 python 3.6 或更早版本。

1. 创建并激活新的 python 环境并安装 pip：
````
conda create --name p36_repurposing python=3.6 pip
````
````
conda activate p36_repurposing
````

1. 使用 conda 安装 torch
````
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
````

1. 使用 pip 安装所需依赖包。注意这里使用了指定版本的 requirements，但任何较新版本通常也可用：
````
pip install -r requirements.txt --use-feature=2020-resolver
````

1. 从 RDkit 官方源安装 RDkit：
````
conda install -c rdkit rdkit
````

1. 如需安装支持 cuda 的 pytorch geometric，请访问 https://data.pyg.org/whl/torch-1.6.0+cu101.html
查找与你的环境（Windows, Python 3.6, CUDA 10.1）对应的 .whl 文件，比如：
```
torch_scatter-2.0.5+cu101-cp36-cp36m-win_amd64.whl
```

下载后安装：
````
pip install torch_scatter-2.0.5+cu101-cp36-cp36m-win_amd64.whl
pip install torch_sparse-0.6.9-cp36-cp36m-win_amd64.whl
pip install torch_cluster-1.5.9-cp36-cp36m-win_amd64.whl
pip install torch_spline_conv-1.2.1-cp36-cp36m-win_amd64.whl
pip install torch-geometric
````