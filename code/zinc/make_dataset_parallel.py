# this file is used to generate one hot encoding of the smile and masking from zinc data file 
# it also do train test sepratrion and save the data in ./data/zinc_clean_smi_test.h5, zinc_clean_smi_train.h5 as well as their labels.

from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import Descriptors, QED


from past.builtins import range
import os
import sys
import numpy as np
import math
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import h5py
from cmd_args import cmd_args


sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
import sascorer


from mol_tree import AnnotatedTree2MolTree
from mol_util import DECISION_DIM
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks
from tree_walker import OnehotBuilder 


## specify the reading and writing directory for the data
#cmd_args.data_save_dir = "./data/data_" + str(cmd_args.max_decode_steps)
#cmd_args.smiles_file = './data/250k_rndm_zinc_drugs_clean.smi'


if not os.path.exists(cmd_args.data_save_dir):
    os.makedirs(cmd_args.data_save_dir)

def process_chunk(smiles_list):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    for smiles in smiles_list:
        ts = parser.parse(smiles, grammar)
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2MolTree(ts[0])
        cfg_tree_list.append(n)

    walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()
    onehot, masks = batch_make_att_masks(cfg_tree_list, tree_decoder, walker, dtype=np.byte)

    return (onehot, masks)

def run_job(L):
    chunk_size = 5000
    
    
    list_binary = Parallel(n_jobs=cmd_args.data_gen_threads, verbose=50)(
        delayed(process_chunk)(L[start: start + chunk_size])
        for start in range(0, len(L), chunk_size)
    )

    
    #process_chunk(L[start: start + chunk_size] for start in range(0, len(L), chunk_size))
   
    
    all_onehot = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)
    all_masks = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)

    for start, b_pair in zip( range(0, len(L), chunk_size), list_binary ):
        all_onehot[start: start + chunk_size, :, :] = b_pair[0]
        all_masks[start: start + chunk_size, :, :] = b_pair[1]

    f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])
    out_file = '%s/%s-%d.h5' % (cmd_args.data_save_dir, f_smiles, cmd_args.skip_deter)
    h5f = h5py.File(out_file, 'w')
    h5f.create_dataset('x', data=all_onehot)
    h5f.create_dataset('masks', data=all_masks)
    h5f.close()
    
    
## generate labels
smiles_list = []
outfile = open("./data/zinc.csv", "w")

with open(cmd_args.smiles_file, 'r') as f:
    for row in tqdm(f):
        smiles = row.strip()
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
        # filter out invalid
        if smiles is None: 
            continue
        # sanity check on rdkit processing
        mol = Chem.MolFromSmiles(smiles)
        logP = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
        qed = QED.qed(mol)
        smiles_list.append(smiles)
        print(','.join([str(a) for a in ( smiles, qed, sa_score, logP)]), file=outfile)
        
outfile.close()


cmd_args.data = './data/zinc.csv'
## load the zinc.csv dataset
import pandas as pd 
data = pd.read_csv(cmd_args.data, header=None)
data = np.array(data)
zinc_clean_smi = data[:,0]
zinc_clean_label= data[:,1:]
## normalizer of  sa_score and logp
sascore_normalizer=  (np.absolute(zinc_clean_label[:,1])).max()
logp_normalizer = (np.absolute(zinc_clean_label[:,2])).max()

#binarize the smile strings
run_job(zinc_clean_smi)

# load the binarized data
h5f = h5py.File(cmd_args.data_save_dir + '/250k_rndm_zinc_drugs_clean-0.h5', 'r')
binary = h5f['x'][:]
masks = h5f['masks'][:]


# shuffle
num = np.arange(binary.shape[0])
random.shuffle(num)
binary_shuffle = binary[num,:,:]
masks_shuffle = masks[num,:,:]
y_shuffle = zinc_clean_label[num,:]
smile_shuffle = zinc_clean_smi[num]

# train test seperation
n_test = 5000
train_x = binary_shuffle[n_test:, :,:]
train_x_mask = masks_shuffle[n_test:, :,:]
train_y = y_shuffle[n_test:, :]
train_smiles = smile_shuffle[n_test:]

test_x = binary_shuffle[:n_test, :,:]
test_x_mask = masks_shuffle[:n_test, :,:]
test_y = y_shuffle[:n_test,:]
test_smiles = smile_shuffle[:n_test]

#normalize
train_normalized_y = np.copy(train_y)
train_normalized_y[:, 1]= train_y[:,1] / sascore_normalizer
train_normalized_y[:,2] = train_y[:,2] / logp_normalizer


# save the training data
train = h5py.File(cmd_args.data_save_dir  + '/zinc_clean_smi_train.h5', 'w')
train.create_dataset('x_train', data=train_x)
train.create_dataset('masks_train', data=train_x_mask)
train.close()
np.save(cmd_args.data_save_dir  + '/zinc_clean_smi_train_y.npy', train_y)
np.save(cmd_args.data_save_dir  + '/zinc_normalized_train_y.npy', train_normalized_y)
np.save(cmd_args.data_save_dir  + '/zinc_clean_smi_train_smile.npy', train_smiles)

# save the ttest data
test = h5py.File(cmd_args.data_save_dir + '/zinc_clean_smi_test.h5', 'w')
test.create_dataset('x_test', data=test_x)
test.create_dataset('masks_test', data=test_x_mask)
test.close()
np.save(cmd_args.data_save_dir  + '/zinc_clean_smi_test_y.npy', test_y)
np.save(cmd_args.data_save_dir + '/zinc_clean_smi_test_smile.npy', test_smiles)

np.save(cmd_args.data_save_dir + '/zinc_sascore_normalizer.npy', sascore_normalizer)
np.save(cmd_args.data_save_dir + '/zinc_logp_normalizer.npy', logp_normalizer)