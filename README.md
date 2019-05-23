### Conditional-generation-of-molecules-by-disentanglement

1. Download the preprocessed data from the following repository and save it under the folder name data/data_100 for QM9 or data/data_278 for ZINC

## Install the dependencies:
The current code depends on pytorch 0.5.0a0. All the dependency is discribed in requirement.txt file. You can also use the Dockerfile to set all the dependency.

## Training 
#1. You can download from public webpage ZINC and QM9 dataset
#2. Use make_dataset_paralle.py file to generate one hot representation of your molecule
#. Run the supervised VAE, then regressor, Then you are ready to train the main model. The well trained supervised VAE and regressor models are also provided. 
There are two kind of models: main_1.py corresponding to the one where we update both the propery estimator and encoder decoder parameters, while main_3.py corresponding to the model we pretrain property estimator and keep it fixed during main model training phase.
To train CGD_GRU_1 model for QM9, run:
```
cd zinc/zinc_1_gru/main_1.py
```
