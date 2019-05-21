## main model is defined here
import h5py
import numpy as np
import argparse
import random
import sys, os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from cmd_args import cmd_args
sys.path.append('%s/../../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser

sys.path.append('%s/../../util/' % os.path.dirname(os.path.realpath('__file__')))
from mol_util import rule_ranges, terminal_idxes, DECISION_DIM
from train_util import PerpCalculator
from train_util import raw_logit_to_smile_labels, run_job

class MolVAE(nn.Module):
    def __init__(self):
        super(MolVAE, self).__init__()
        self.latent_dim = cmd_args.latent_dim
        self.encoder = CNNEncoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.latent_dim)
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.latent_dim)
        self.perp_calc = PerpCalculator()

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = mu.data.new(mu.size()).normal_(0, cmd_args.eps_std)            
            if cmd_args.mode == 'gpu':
                eps = eps.cuda()
            eps = Variable(eps)
            
            return mu + eps * torch.exp(logvar * 0.5)            
        else:
            return mu

    def forward(self, x_inputs, y_inputs, true_binary, rule_masks, t_y):        
        z_mean, z_log_var = self.encoder(x_inputs)
        z = self.reparameterize(z_mean, z_log_var)
        raw_logits = self.state_decoder(z)   
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        
        return perplexity, cmd_args.kl_coeff * torch.mean(kl_loss)


# encoder and decoder
sys.path.append('%s/../../util/' % os.path.dirname(os.path.realpath('__file__')))
from pytorch_initializer import weights_init
from mol_util import DECISION_DIM
#q(z|x)
class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1d(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)
        weights_init(self)

    def forward(self, x_cpu):
        if cmd_args.mode == 'cpu':
            batch_input = Variable(torch.from_numpy(x_cpu))
        else:
            batch_input = Variable(torch.from_numpy(x_cpu).cuda())

        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)        
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        # h3 = torch.transpose(h3, 1, 2).contiguous()
        flatten = h3.view(x_cpu.shape[0], -1)
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)
        
        return (z_mean, z_log_var)    
    
    
#decoder    
class StateDecoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(StateDecoder, self).__init__()
        self.latent_dim = latent_dim 
        self.max_len = max_len

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if cmd_args.rnn_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, 501, 3)
        elif cmd_args.rnn_type == 'sru':
            self.gru = SRU(self.latent_dim, 501, 3)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(501, DECISION_DIM)
        weights_init(self)
        
    def forward(self, z):
       
       

        assert len(z.size()) == 2 # assert the input is a matrix
        
        h = self.z_to_latent(z)
        h = F.relu(h)

        rep_h = h.expand(self.max_len, z.size()[0], z.size()[1]) # repeat along time steps

        out, _ = self.gru(rep_h) # run multi-layer gru

        logits = self.decoded_logits(out)

        return logits
    
    
    
