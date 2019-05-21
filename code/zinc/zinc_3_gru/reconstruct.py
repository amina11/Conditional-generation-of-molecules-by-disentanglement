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


## import helper functions
from cmd_args import cmd_args
sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks
from tree_walker import OnehotBuilder, ConditionalDecoder 
#sys.path.append('%s/../' % os.path.dirname(os.path.realpath('__file__')))
from reconstruct_util import cal_accuracy


## load the data 
cmd_args.training_data_dir = '../data/data_278'
test_smiles = np.load(cmd_args.training_data_dir + '/zinc_clean_smi_test_smile.npy')
test_unnormalized_labels = np.load(cmd_args.training_data_dir + '/zinc_clean_smi_test_y.npy')
logP_normalizer = np.load(cmd_args.training_data_dir + '/zinc_logp_normalizer.npy')
test_labels = np.copy(test_unnormalized_labels)
#test_labels[:, 1]= test_unnormalized_labels[:,1] / Sa_score_normalizer
test_labels[:,2] = test_unnormalized_labels[:,2] / logP_normalizer





cmd_args.saved_model = './model_1/epoch-best.model'
grammar = parser.Grammar(cmd_args.grammar_file)
cmd_args.mode ='cpu'


num_test = 1000
label_index = 2
small_testx = test_smiles[:num_test]
small_testy = (test_labels[:num_test,label_index]).astype(float)
encode_times = 1
decode_times = 1
chunk_size = 100
decode_result_save_file =  cmd_args.saved_model + '-reconstruct_decode_result.csv'
accuracy_save_file =  cmd_args.saved_model + '-reconstruct_accuracy.txt'            



## load the trained model
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

##############################################################
sys.path.append('%s/../' % os.path.dirname(os.path.realpath('__file__')))
from reconstruct_util import reconstruct, cal_accuracy
from AttMolProxy import AttMolProxy
from reconstruct_util import save_decode_result
model = AttMolProxy()
decode_result = reconstruct(model, small_testx, small_testy,chunk_size, encode_times, decode_times)
accuracy, junk = cal_accuracy(decode_result,small_testx)
print('accuracy:', accuracy, 'junk:', junk)

save_result = True
if save_result:
    with open(accuracy_save_file, 'w') as fout:
        print('accuracy:', accuracy, 'junk:', junk, file=fout)
        
    save_decode_result(decode_result,small_testx, decode_result_save_file)
#################################################

