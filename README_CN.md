# 基于嵌入的化合物-病毒蛋白活性预测建模框架

[English](README.md) | 中文

环境配置及所需全部依赖包的安装指南详见 [installation_guide](installation_guide_CN.md) 文件。

本仓库提供了用于预测化合物-病毒活性分数的脚本、数据、模型和结果。

关于如何获取预测模型的训练集和测试集的详细说明，请参见 data 文件夹。

我们以 DENV 病毒为用例，基于三种主要蛋白酶提供了化合物的排序列表：

a) NS1 蛋白
b) NS3 蛋白
c) NS5 蛋白  

本项目包含七种机器学习模型。三种传统机器学习模型如下：

a) 随机森林 - `scripts/supervised_rf_on_ls_protein_compound.py`  
b) 支持向量机（SVM） - `scripts/supervised_svm_on_ls_protein_compound.py`  
c) XGBoost - `scripts/supervised_xgb_on_ls_protein_compound.py`

运行方法：

 * `cd scripts`

 * `python supervised_<method>_on_ls_protein_compound.py Train_Compound_Viral_interactions_for_Supervised_Learning_with_<compound_features>_LS.csv <type>_Compound_Viral_interactions_for_Supervised_Learning_with_<compound_features>_LS.csv <method>_<compound_features>_Compound_LS_Protein_supervised_<type>_predictions.csv`

其中 `<method>` 可为 `rf`、`svm`、`xgb`，`<compound_features>` 可为 `MFP` 或 `LS`，`<type>` 可为 `Test` 或 `denv`。

训练和测试（`Test` 或 `denv`）所需的文件可通过按照 data 文件夹下 **README** 的说明生成。

输出：  

a) 随机森林 - `results/RF_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`、`results/RF_LS_Compound_LS_Protein_supervised_Test_predictions.csv`、`results/RF_supervised_denv_predictions.csv`  

b) SVM - `results/SVM_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`、`results/SVM_LS_Compound_LS_Protein_supervised_test_predictions.csv`、`results/SVM_supervised_denv_predictions.csv`  

c) XGB - `results/XGB_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`、`results/XGB_LS_Compound_LS_Protein_supervised_Test_predictions.csv`、`results/XGB_supervised_denv_predictions.csv`   


四种端到端深度学习模型：  

a) CNN - `scripts/torchtext_cnn_supervised_learning.py`  
b) LSTM - `scripts/torchtext_lstm_supervised_learning.py`  
c) CNN-LSTM - `scripts/torchtext_cnn_lstm_supervised_learning.py`  
d) GAT-CNN  - `scripts/torchtext_gat_cnn_supervised_learning.py`

测试模式下运行：  
1. `data/Test_Compound_Viral_interactions_for_Supervised_Learning.csv`  
2. `data/denv_Compound_Viral_interactions_for_Supervised_Learning.csv`

运行方法：

 * `cd scripts`

 * `python torchtext_<method>_supervised_learning.py Train_Compound_Viral_interactions_for_Supervised_Learning.csv <type>_Compound_Viral_interactions_for_Supervised_Learning.csv <method>_supervised_<type>_predictions.csv`

其中 `<method>` 可为 `cnn`、`lstm`、`cnn_lstm`、`gat_cnn`，`<type>` 可为 `Test` 或  `denv`。

输出文件及位置：

a) CNN - `results/cnn_supervised_Test_predictions.csv`、`results/cnn_supervised_denv_predictions.csv`  
b) LSTM - `results/lstm_supervised_Test_predictions.csv`、`results/lstm_supervised_denv_predictions.csv`  
c) CNN-LSTM - `results/cnn_lstm_supervised_Test_predictions.csv`、`results/cnn_lstm_supervised_denv_predictions.csv`
d) GAT-CNN - `results/gat_cnn_supervised_Test_predictions.csv`、`results/gat_cnn_supervised_denv_predictions.csv`


获取 DENV 病毒蛋白的化合物排序列表：   
a) 运行 `denv_postprocessing.py`

输出：  
a) NS1-Pro - `results/NS1_Pro_Top_Ranked_Compounds.csv`  
b) NS3-Pro - `results/NS3_Pro_Top_Ranked_Compounds.csv`  
c) NS5-Pro  - `results/NS5_Pro_Top_Ranked_Compounds.csv`  