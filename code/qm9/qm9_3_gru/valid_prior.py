## reconstruction demo 
## encoder only take x as input
## spcify the trained model in the cmd_args.saved_model
from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from joblib import Parallel, delayed
import future

from cmd_args import cmd_args
sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks
from tree_walker import OnehotBuilder, ConditionalDecoder 
sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
from reconstruct_util import cal_accuracy

cmd_args.training_data_dir = '../data/data_100'
#cmd_args.regressor_saved_dir = './regressor/regressor_pretrained_pretrained_vae_output'
cmd_args.saved_model = './model_2/epoch-best.model'
grammar = parser.Grammar(cmd_args.grammar_file)


test_smiles = np.load(cmd_args.training_data_dir + '/QM9_clean_smi_test_smile.npy')
test_labels = np.load(cmd_args.training_data_dir + '/QM9_normalized_test_y.npy')
logP_normalizer = np.load(cmd_args.training_data_dir + '/QM9_logp_normalizer.npy')
cmd_args.mode ='cpu'
label_index = 2

#sys.path.append('%s/../' % os.path.dirname(os.path.realpath('__file__')))
from model import CNNEncoder, MolVAE
vae = MolVAE()
if cmd_args.mode == 'gpu':
    vae = vae.cuda()

device = torch.device('cpu')
assert cmd_args.saved_model is not None
if cmd_args.saved_model is not None and cmd_args.saved_model != '':
        if os.path.isfile(cmd_args.saved_model):
            print('loading model from %s' % cmd_args.saved_model)
            vae.load_state_dict(torch.load(cmd_args.saved_model, map_location=device))

## valid prior
from AttMolProxy import AttMolProxy
from reconstruct_util import cal_valid_prior
model = AttMolProxy()

# 0. Constants
nb_latent_point = 1000
chunk_size = 100
sample_times = 1
sigma = 0.035
seed = cmd_args.seed
np.random.seed(seed)
model = AttMolProxy()
decoded_prior = cal_valid_prior(model, cmd_args.latent_dim,test_labels[0:nb_latent_point,label_index], nb_latent_point, sample_times, chunk_size, sigma)

print('validity:', decoded_prior[0])

num_unique = len(np.unique(np.array(decoded_prior[3]), return_index=True)[1])
print('number of unique smiles string:', num_unique)
print('uniquness:', num_unique / decoded_prior[1])


train_smiles = np.load(cmd_args.training_data_dir + '/QM9_clean_smi_train_smile.npy')
common = list(set(train_smiles.tolist()).intersection(decoded_prior[3]))
print('number of common smiles strings with training data:', len(common))
novelty = (decoded_prior[1] - len(common)) / decoded_prior[1]
print('novelty:', novelty)

