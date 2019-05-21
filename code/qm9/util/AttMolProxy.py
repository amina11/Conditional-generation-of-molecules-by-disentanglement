## for reconstruction
from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

import torch
from torch.autograd import Variable
from tqdm import tqdm
from joblib import Parallel, delayed

from cmd_args import cmd_args
from model import CNNEncoder, MolVAE
from reconstruct_util import parse


sys.path.append('%s/../util' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks
from tree_walker import OnehotBuilder, ConditionalDecoder
import cfg_parser as parser



class AttMolProxy(object):
    def __init__(self, *args, **kwargs):
        if cmd_args.ae_type == 'vae':
            self.ae = MolVAE()
        else:
            raise Exception('unknown ae type %s' % cmd_args.ae_type)
            
        if cmd_args.mode == 'gpu':
            self.ae = self.ae.cuda()

        assert cmd_args.saved_model is not None
        
        if cmd_args.mode == 'cpu':
            self.ae.load_state_dict(torch.load(cmd_args.saved_model, map_location=lambda storage, loc: storage))
        else:
            self.ae.load_state_dict(torch.load(cmd_args.saved_model))

        self.onehot_walker = OnehotBuilder()
        self.tree_decoder = create_tree_decoder()
        self.grammar = parser.Grammar(cmd_args.grammar_file)
        

    def encode(self, chunk, use_random):
        '''
        Args:
            chunk: a list of `n` strings, each being a SMILES.
        Returns:
            A numpy array of dtype np.float32, of shape (n, latent_dim)
            Note: Each row should be the *mean* of the latent space distrubtion rather than a sampled point from that distribution.
            (It can be anythin as long as it fits what self.decode expects)
        '''
        '''
        if cmd_args.mode == 'cpu':
            y_inputs = Variable(torch.from_numpy(labels))
        else:
            y_inputs = Variable(torch.from_numpy(labels).cuda())
            
         '''
        
        #y_inputs = labels
            
        if type(chunk[0]) is str:
            cfg_tree_list = parse(chunk, self.grammar)
        else:
            cfg_tree_list = chunk
            
        onehot, _ = batch_make_att_masks(cfg_tree_list, self.tree_decoder, self.onehot_walker, dtype=np.float32)

        x_inputs = np.transpose(onehot, [0, 2, 1])
        

        if use_random:
            self.ae.train()
        else:
            self.ae.eval()
            
        z_mean, _ = self.ae.encoder(x_inputs)

        return z_mean.data.cpu().numpy()
    

    def pred_raw_logits(self, chunk, labels, n_steps=None):
        '''
        Args:
            chunk: A numpy array of dtype np.float32, of shape (n, latent_dim)
        Return:
            numpy array of MAXLEN x batch_size x DECISION_DIM
        '''
        if cmd_args.mode == 'cpu':
            z = Variable(torch.from_numpy(chunk))
            y_inputs = Variable(torch.from_numpy(labels))
        else:
            z = Variable(torch.from_numpy(chunk).cuda())
            y_inputs = Variable(torch.from_numpy(labels).cuda())

        raw_logits = self.ae.state_decoder(z,y_inputs)

        raw_logits = raw_logits.data.cpu().numpy()

        return raw_logits

    def decode(self, chunk, labels, use_random):
        '''
        Args:
            chunk: A numpy array of dtype np.float32, of shape (n, latent_dim)
        Return:
            a list of `n` strings, each being a SMILES.
        '''
        raw_logits = self.pred_raw_logits(chunk, labels)

        result_list = []
        for i in range(raw_logits.shape[1]):
            pred_logits = raw_logits[:, i, :]

            walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

            new_t = Node('smiles')
            try:
                self.tree_decoder.decode(new_t, walker)
                sampled = get_smiles_from_tree(new_t)
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    print('Warning, decoder failed with', ex)
                # failed. output a random junk.
                import random, string
                #sampled = 'JUNK' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))
                sampled = 'JUNK'
            result_list.append(sampled)
        
        return result_list
    
    
   
