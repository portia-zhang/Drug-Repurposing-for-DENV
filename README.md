# A Modelling Framework for Embedding-based Predictions for Compound-Viral Protein Activity

English | [中文](README_CN.md)

The guidelines to setup the environment and all the packages required to run our framework is available in [installation_guide](installation_guide.md) file

Here we provide the scripts, data, model and results for predicting compound-viral activity scores using our pipeline.

Details about how to obtain the training and test set for the predictive models is provided in the data folder.

We consider the DENV virus as a use-case and provide ranked list of compounds based on the 3 main proteases:

a) NS1 protein 
b) NS3 protein  
c) NS5 protein  

This package contains seven individual machine learning models. The three traditional machine learning models are as follows:

a) Random Forests - `scripts/supervised_rf_on_ls_protein_compound.py`  
b) SVM - `scripts/supervised_svm_on_ls_protein_compound.py`  
c) XGBoost - `scripts/supervised_xgb_on_ls_protein_compound.py`

How to run:

 * `cd scripts`

 * `python supervised_<method>_on_ls_protein_compound.py Train_Compound_Viral_interactions_for_Supervised_Learning_with_<compound_features>_LS.csv <type>_Compound_Viral_interactions_for_Supervised_Learning_with_<compound_features>_LS.csv <method>_<compound_features>_Compound_LS_Protein_supervised_<type>_predictions.csv`

Here `<method>` can be either `rf`, `svm`, `xgb`, `<compound_features>` can be either `MFP` or `LS` and `<type>` can either `Test` or `denv`.

The files for training and testing (`Test` or `denv`) are produced in `data` folder by following the instructions in the **README** available in the `data` folder.

Outputs:  

a) RF - `results/RF_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/RF_LS_Compound_LS_Protein_supervised_Test_predictions.csv` and `results/RF_supervised_denv_predictions.csv`  

b) SVM - `results/SVM_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/SVM_LS_Compound_LS_Protein_supervised_test_predictions.csv` and `results/SVM_supervised_denv_predictions.csv`  
c) XGB - `results/XGB_MFP_Compound_LS_Protein_supervised_Test_predictions.csv`, `results/XGB_LS_Compound_LS_Protein_supervised_Test_predictions.csv` and `results/XGB_supervised_denv_predictions.csv`   


The four end-to-end deep learning models:  

a) CNN - `scripts/torchtext_cnn_supervised_learning.py`  
b) LSTM - `scripts/torchtext_lstm_supervised_learning.py`  
c) CNN-LSTM - `scripts/torchtext_cnn_lstm_supervised_learning.py`  
d) GAT-CNN  - `scripts/torchtext_gat_cnn_supervised_learning.py`

Runs on test mode:  
1. `data/Test_Compound_Viral_interactions_for_Supervised_Learning.csv`  
2. `data/denv_Compound_Viral_interactions_for_Supervised_Learning.csv`

How to run:

 * `cd scripts`

 * `python torchtext_<method>_supervised_learning.py Train_Compound_Viral_interactions_for_Supervised_Learning.csv <type>_Compound_Viral_interactions_for_Supervised_Learning.csv <method>_supervised_<type>_predictions.csv`

Here `<method>` can be either `cnn`, `lstm`, `cnn_lstm`, `gat_cnn` and `<type>` can be either `Test` or  `denv`.

Ouput Files with location:

a) CNN - `results/cnn_supervised_Test_predictions.csv` and `results/cnn_supervised_denv_predictions.csv`  
b) LSTM - `results/lstm_supervised_Test_predictions.csv` and `results/lstm_supervised_denv_predictions.csv`  
c) CNN-LSTM - `results/cnn_lstm_supervised_Test_predictions.csv` and `results/cnn_lstm_supervised_denv_predictions.csv`
d) GAT-CNN - `results/gat_cnn_supervised_Test_predictions.csv` and `results/gat_cnn_supervised_denv_predictions.csv`


To get a ranked list of compounds for DENV viral proteins:   
a) Run `denv_postprocessing.py`

Outputs:  
a) NS1-Pro - `results/NS1_Pro_Top_Ranked_Compounds.csv`  
b) NS3-Pro - `results/NS3_Pro_Top_Ranked_Compounds.csv`  
c) NS5-Pro  - `results/NS5_Pro_Top_Ranked_Compounds.csv`  
