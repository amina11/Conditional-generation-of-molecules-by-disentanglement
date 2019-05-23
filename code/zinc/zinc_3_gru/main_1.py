import h5py
import pdb
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
cmd_args.save_dir = './model_1'
cmd_args.RL_param = 50
cmd_args.regressor_saved_dir = './regressor/regressor_pretrained/'
cmd_args.vanilla_vae_save_dir = './supervised_vae/vanilla_supervised_vae'
seed = 19260817
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if not os.path.exists(cmd_args.save_dir):
    os.makedirs(cmd_args.save_dir)


sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
from train_util import Prepare_data, get_batch_input_vae, get_batch_input_regressor


getting_data = Prepare_data(cmd_args)
train_binary_x, train_masks, valid_binary_x, valid_masks, train_y, valid_y = getting_data.load_data()
print('num_train: %d\tnum_valid: %d' % (train_y.shape[0], valid_binary_x.shape[0]))
train_inputs, train_binary, train_masks = getting_data.pre_process(train_binary_x, train_masks, train_y)
valid_inputs, valid_binary, valid_masks = getting_data.pre_process(valid_binary_x, valid_masks, valid_y)

from model import Regressor, MolVAE, train_regressor, epoch_train
regressor = Regressor(cmd_args.max_decode_steps, cmd_args.latent_dim)
regressor.cuda()
#define vae model
ae = MolVAE()
if cmd_args.mode == 'gpu':
    ae = ae.cuda()
assert cmd_args.encoder_type == 'cnn'


# load the pretrained vae model
#pretrained_model = cmd_args.vanilla_vae_save_dir + '/epoch-40.model'
pretrained_model = cmd_args.save_dir + '/epoch-best.model'
if pretrained_model  is not None and pretrained_model != '':
        if os.path.isfile(pretrained_model):
            print('loading model from %s' % pretrained_model)
            ae.load_state_dict(torch.load(pretrained_model))


# load the pretrained regressor
#cmd_args.regressor_saved_model = cmd_args.regressor_saved_dir + '/epoch-best.model'
cmd_args.regressor_saved_model = cmd_args.save_dir + '/regressor-epoch-best.model'
if cmd_args.regressor_saved_model is not None and cmd_args.regressor_saved_model != '':
        if os.path.isfile(cmd_args.regressor_saved_model):
            print('loading model from %s' % cmd_args.regressor_saved_model)
            regressor.load_state_dict(torch.load(cmd_args.regressor_saved_model))


#optimizer
optimizer_encoder = optim.Adam(ae.encoder.parameters(), lr=cmd_args.learning_rate)
optimizer_decoder = optim.Adam(ae.state_decoder.parameters(), lr = cmd_args.learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer_decoder, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.001)
optimizer_regressor = optim.Adam(regressor.parameters(), lr=cmd_args.learning_rate)
#train
sample_idxes = list(range(train_binary_x.shape[0]))
best_valid_loss = None
kl = []
prep = []
original_reward = []
permuted_reward = []



import time
for epoch in range(cmd_args.num_epochs):
    start = time.time()
    random.shuffle(sample_idxes)
    ## update the vae:
    for epoch_vae in range(5):
        ae, vae_loss = epoch_train('train',epoch_vae, ae, regressor, sample_idxes, train_inputs, train_binary, train_masks, train_y,cmd_args, optimizer_encoder, optimizer_decoder)
        print('>>>>average \033[92mtraining\033[0m of epoch %d: vae_loss loss %.5f regularizer_loss %.5f original_reward %.5f permuted_reward %.5f prep %.5f kl %.5f' % (epoch_vae, vae_loss[0], vae_loss[1], vae_loss[2], vae_loss[3], vae_loss[4], vae_loss[5]))
        kl.append(vae_loss[5])
        prep.append(vae_loss[4])
        permuted_reward.append(vae_loss[3])
        original_reward.append(vae_loss[2])

    ## update the reggressor:
    for epoch_regressor in range(2):
        regressor, regressor_loss, reg_1= train_regressor('train', epoch_regressor, optimizer_regressor, ae, regressor,sample_idxes,  train_inputs, train_y)
        print('>>>>average regressor \033[92mtraining\033[0m of epoch %d: loss %.5f, reg1 %.5f' % (epoch_regressor, regressor_loss, reg_1))




    if epoch % 1 == 0:
        _, valid_loss = epoch_train('valid', epoch,  ae, regressor, list(range(valid_inputs.shape[0])), valid_inputs, valid_binary, valid_masks, valid_y,cmd_args)
        print('>>>>average \033[92mtraining\033[0m of epoch %d: vae_loss loss %.5f regularizer_loss %.5f original_reward %.5f permuted_reward %.5f prep %.5f kl %.5f' % (epoch_vae, valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5]))
        valid_loss = valid_loss[2] + valid_loss[3]+ valid_loss[4]
        lr_scheduler.step(valid_loss)
        torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-%d.model' % epoch)

        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('saving to best model since this is the best valid loss so far.----')
            torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-best.model')
            torch.save(regressor.state_dict(), cmd_args.save_dir + '/regressor-epoch-best.model')

            np.save(cmd_args.save_dir + '/kl.npy', kl)
            np.save(cmd_args.save_dir + '/prep.npy', prep)
            np.save(cmd_args.save_dir +'/permuted_reward.npy', permuted_reward)
            np.save(cmd_args.save_dir + '/original_reward.npy', original_reward)

    print('time per epoch:', time.time()- start)
np.save(cmd_args.save_dir + '/kl.npy', kl)
np.save(cmd_args.save_dir + '/prep.npy', prep)
np.save(cmd_args.save_dir +'/permuted_reward.npy', permuted_reward)
np.save(cmd_args.save_dir + '/original_reward.npy', original_reward)



