### Conditional-generation-of-molecules-by-disentanglement

## Preparing the data
You can either pre-process the raw smile string data to generate one hot encoding of molecules and their masks, or you can directly download the preprocessed data:
1. To preprocess the raw smile string data:
   -
   - ZINC: download the smile string from the [link](https://www.dropbox.com/sh/621ufmvqgg5h2d8/AAC5y8QTKdtEdBa4HX0jX8fwa/data/zinc?dl=0&preview=250k_rndm_zinc_drugs_clean.smi&subfolder_nav_tracking=1)

1. Download the preprocessed data from the following repository and save it under the folder name data/data_100 for QM9 or data/data_278 for ZINC

## Install the dependencies:
The current code depends on pytorch 0.5.0a0. All the dependency is discribed in requirement.txt file. You can also use the Dockerfile to set up all the dependency or directly pull the ready image from the following cite https://cloud.docker.com/u/2012913/repository/docker/2012913/sdvae_5_8

## Training 

There are two kind of models: main_1.py corresponding to the one where we update both the propery estimator and encoder decoder parameters, while main_3.py corresponding to the model we pretrain property estimator and keep it fixed during main model training phase.
To train CGD_GRU_1 model for QM9, run:
```
cd zinc/zinc_1_gru/main_1.py
```
