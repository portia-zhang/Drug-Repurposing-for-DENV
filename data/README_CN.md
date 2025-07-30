# 基于序列的药物-病毒蛋白活性预测的多方法共识方案（DENV）

本文档详细介绍了用于训练集、测试集和 DENV 数据集的数据准备步骤。

# SMILES Autoencoder

1. 首先，我们收集了 556,134 个用于论文 [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111) 的化合物 SMILES 字符串，并与 MOSES 数据集 [Molecular Sets (MOSES): A BenchmarkingPlatform for Molecular Generation Models](https://github.com/molecularsets/moses) 中获得的 1,936,962 个 SMILES 字符串合并。该文件位于 `SMILES_Autoencoder/all_smiles.csv`。

2. 接下来执行以下操作：
    * 运行 `cd SMILES_Autoencoder`
    * 运行 `python cleanup_smiles.py all_smiles.csv all_smiles_revised.csv`，以过滤含有盐和去除立体化学信息的化合物。

3. 生成的输出文件 `all_smiles_revised.csv` 包含 2,459,695 个化合物。

4. 然后运行：`python prepare_smiles_autoencoder.py`，得到输出文件 `all_smiles_revised_final.csv`。我们根据分子长度进行筛选，最终保留 SMILES 序列长度在 [10,128] 之间的化合物，使得化学搜索空间既包含小分子也包含大分子配体，比 [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111) 使用的空间更大。

5. 输出文件 `all_smiles_revised_final.csv` 应包含 2,454,665 个化合物。

6. 训练 SMILES 自动编码器模型：
   `cd ../../scripts/`
   `python torchtext_smiles_autoencoder.py ../data/SMILES_Autoencoder/all_smiles_revised_final.csv`
   结果文件 `models/lstm_out/torchtext_checkpoint.pt` 会生成在 models 文件夹下。

# Protein Autoencoder

1. 我们从 Uniprot 下载了 2,684,774 条病毒蛋白序列，数据可在 (https://data.mendeley.com/datasets/8rrwnbcgmx) 下载，下载后应放在 `data/Protein_Autoencoder` 文件夹下。

2. 训练蛋白质自动编码器的步骤如下：

 * `python clean_proteins.py Full_Viral_proteins_with_Uniprot_IDs.csv clean_Uniprot_proteins.csv`
 
结果为 2,658,225 条长度不超过 2000 的病毒蛋白序列。

 * `python encode_proteins.py clean_Uniprot_proteins.csv encoded_Uniprot_proteins.csv`
 * `python train_protein_autoencoder.py encoded_Uniprot_proteins.csv ../../models/cnn_out/cnn_protein_autoencoder.h5`

文件 `cnn_protein_autoencoder.h5` 即为蛋白质自动编码器模型。

# 用于端到端深度学习模型的化合物-病毒活性数据

我们在 Pubmed (NCBI) 上检索生成了优质的 AID（Assay Id）列表：

1. 蛋白靶点 GI73745819 - SARS 蛋白酶 - 本文称为 `SARS_C3_Assays.txt`
2. 蛋白靶点 GI75593047 - HIV pol 多肽 - 本文称为 `HIV_Protease_Assays.txt`
3. NS3 - Hep3 蛋白酶 - 本文称为 `NS3_Protease_Assays.txt`
4. 3CL-Pro - Mers 蛋白酶 - 本文称为 `MERS_Protease_Assays.txt`

所有数据均位于 `Compound_Virus_Interactions/additional_data` 文件夹内。

我们通过 Uniprot 获得这些病毒的蛋白酶序列，保存在 `Compound_Virus_Interactions/ncbi_Filtered_Viral_Proteins.csv` 文件中。

接下来执行：
 * `cd Compound_Virus_Interactions`
 * `gunzip additional_data/ns3_assays.pkl.gz`
 * `python PreProcessing_More_Data.py ncbi_Filtered_Viral_Proteins.csv ncbi_Filtered_Compound_Viral_Proteins_Network.csv`

结果文件 `ncbi_Filtered_Compound_Viral_Proteins_Network.csv` 位于 `Compound_Virus_Interactions` 文件夹。

5. 我们还下载了 ChEMBL 上整理的化合物-病毒蛋白活性数据，文件名为 `Compound_Viral_protein_Networks.csv`。

6. 病毒蛋白序列从 Uniprot 下载，下载链接为 (https://drive.google.com/file/d/1nmqUZd5_RKxF_FJ9A_nkHIA9H0sevBMK/view?usp=sharing)，下载后应放在 `Compound_Virus_Interactions` 文件夹。

7. 运行：`python chembl_filter_compound_virus_interactions.py Compound_Viral_protein_Networks.csv Full_Viral_proteins_with_Uniprot_IDs.csv.gz chembl_Filtered_Compound_Viral_Proteins_Networks`，筛选 IC50、Ki 或 Kd 类型的活性数据，并去除长度不合规的化合物（x<10 或 x>128）。
   结果文件为 `chembl_Filtered_Compound_Viral_Proteins_Network.csv`，位于 `Compound_Virus_Interactions` 文件夹。

8. 合并并划分化合物-病毒蛋白活性数据为训练集和测试集：
   `cd ../../scripts/`
   `python train_valid_test_deep_learning.py chembl_Filtered_Compound_Viral_Proteins_Network.csv ncbi_Filtered_Compound_Viral_Proteins_Network.csv Train_Compound_Viral_interactions_for_Supervised_Learning.csv Test_Compound_Viral_interactions_for_Supervised_Learning.csv`

输出文件 `Train_Compound_Viral_interactions_for_Supervised_Learning.csv` 和 `Test_Compound_Viral_interactions_for_Supervised_Learning.csv` 位于 `data` 文件夹。

# 基于嵌入的传统监督学习模型的数据准备

使用 Teacher Forcing-LSTM SMILES 自动编码器生成化合物的嵌入表示，步骤如下：

1. 训练集：
   * 在仓库主目录下运行 `cd scripts`
   * 运行 `python ls_generator_smiles.py --input Train_Compound_Viral_interactions_for_Supervised_Learning.csv --output Train_Compound_LS.csv`
   结果文件 `Train_Compound_LS.csv` 位于 `data` 文件夹，行数与训练集一致，维度为 256。

2. 测试集：
   * 运行 `python ls_generator_smiles.py --input Test_Compound_Viral_interactions_for_Supervised_Learning.csv --output Test_Compound_LS.csv`
   结果文件 `Test_Compound_LS.csv` 位于 `data` 文件夹。

使用 Morgan 指纹生成化合物的向量表示，步骤如下：

1. 训练集：
   * 在仓库主目录下运行 `cd scripts`
   * 运行 `python ls_generator_morgan.py --input Train_Compound_Viral_interactions_for_Supervised_Learning.csv --output Train_Compound_MFP.csv`
   结果文件 `Train_Compound_MFP.csv` 位于 `data` 文件夹，行数与训练集一致，维度为 256。

2. 测试集：
   * 运行 `python ls_generator_morgan.py --input Test_Compound_Viral_interactions_for_Supervised_Learning.csv --output Test_Compound_MFP.csv`
   结果文件 `Test_Compound_MFP.csv` 位于 `data` 文件夹。

生成病毒蛋白的向量表示，步骤如下：

1. 针对训练/测试/DENV 集合，执行：
   * 运行 `cd scripts`
   * 运行 `python ../data/Protein_Autoencoder/clean_proteins.py ../data/<type>_Compound_Viral_interactions_for_Supervised_Learning.csv ../data/clean_<type>_proteins.csv`
   * 运行 `python ../data/Protein_Autoencoder/encode_proteins.py ../data/clean_<type>_proteins.csv ../data/encoded_<type>_proteins.csv`
   * 运行 `python ../data/Protein_Autoencoder/generate_PLS.py ../data/encoded_<type>_proteins.csv ../data/<type>_Protein_LS.csv ../models/cnn_out/cnn_protein_autoencoder.h5`
   * 运行 `rm ../clean_<type>_proteins.csv ../encoded_<type>_proteins.csv`
   其中 `<type>` 可为 `Train`、`Test` 或 `denv`。

生成包含 SMILES/Morgan 指纹和蛋白嵌入的训练集和测试集：

1. 运行 `cd scripts`
2. 运行 `python train_valid_test_supervised_learning_on_ls.py <type>_Compound_Viral_interactions_for_Supervised_Learning.csv <type>_Compound_LS.csv <type>_Compound_MFP.csv <type>_Protein_LS.csv <type>_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv <type>_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv`

其中 `<type>` 可为 `Train` 或 `Test`。
`<type>_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv` 表示包含 SMILES 嵌入+蛋白嵌入的文件。
`<type>_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv` 表示包含 Morgan 指纹+蛋白嵌入的文件。

# DENV 病毒蛋白的化合物活性预测

1. 我们从 [Drug Virus Info](http://drugvirus.info/)、[Barabasi 论文](https://arxiv.org/abs/2004.07229)、[Discovery of SARS-CoV-2 antiviral drugs through large-scale compound repurposing](https://www.nature.com/articles/s41586-020-2577-1) 下载了化合物列表，分别保存在 `data/DENV/compound_virus_info.lst`、`data/DENV/compounds_barabasi.list` 和 `data/DENV/compounds_12k.csv`。12k 化合物可在 [ReframeDB](https://reframedb.org/) 下载。作者未直接提供化合物名称，而是提供了 68 个不同的活性筛选实验，下载后共获得 2383 个化合物。

2. 合并所有化合物：

   `cd data/DENV`

   `python get_all_compounds_DENV.py compounds_barabasi.list compound_virus_info.list compounds_12k.csv all_verified_keys.list`
   
该脚本筛选 SMILES 长度在 10-128 且不含盐桥的唯一化合物，最终在 `all_verified_keys.list` 中得到 1482 个化合物。

1. 登革热病毒与宿主相互作用的数据来源于 DenvInd 数据库（ICGEB，印度），该数据库专注于收录经实验验证和计算预测的 [DENV-宿主分子互作](http://bioinfo.icgeb.res.in/denvind/)，最终合并到 `all_verified_keys.list` 中得到 1960 个化合物。

2. 准备 DENV 端到端深度学习测试集：

   `python denv_preprocessing.py denv.fasta all_verified_keys.list denv_Compound_Viral_interactions_for_Supervised_Learning.csv`
   
结果文件 `denv_Compound_Viral_interactions_for_Supervised_Learning.csv` 位于 `data` 文件夹。

其中 `pchembl_value` 字段使用 0.0 作为占位，以符合深度学习模型的输入格式。

5. 可通过以下脚本获得 CNN、LSTM、CNN+LSTM 和 GAT-CNN 模型在 DENV 病毒蛋白上的预测结果：
   * `cd scripts`
   * 运行：
     * `torchtext_cnn_supervised_learning.py`
     * `torchtext_lstm_supervised_learning.py`
     * `torchtext_cnn_lstm_supervised_learning.py`
     * `torchtext_gat_cnn_supervised_learning.py`
   这些脚本均位于 `scripts` 文件夹，更多运行细节见主目录下的 **README**。

6. 使用 SMILES 编码+蛋白编码和 Morgan 指纹+蛋白编码准备 DENV 测试集：
   * `cd scripts`
   * 运行 `python ls_generator_smiles.py --input denv_Compound_Viral_interactions_for_Supervised_Learning.csv --output denv_Compound_LS.csv`
   * 运行 `python ls_generator_morgan.py --input denv_Compound_Viral_interactions_for_Supervised_Learning.csv --output denv_Compound_MFP.csv`
   * 按前述方法生成蛋白嵌入，得到 `denv_Protein_LS.csv`
  
   * 运行 `python test_denv_supervised_learning_on_ls.py denv_Compound_Viral_interactions_for_Supervised_Learning.csv denv_Compound_LS.csv denv_Compound_MFP.csv denv_Protein_LS.csv denv_Compound_Viral_interactions_for_Supervised_Learning_with_LS_LS.csv denv_Compound_Viral_interactions_for_Supervised_Learning_with_MFP_LS.csv`
   
最后两个参数分别为 SMILES 嵌入+蛋白嵌入和 Morgan 指纹+蛋白嵌入的输出文件。

 * 现在可以运行主目录 **README** 中提到的 RF、SVM 和 XGBoost 等主流机器学习方法。 