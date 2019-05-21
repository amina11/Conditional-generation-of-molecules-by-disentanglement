# 2019.1.16
# encoder take only x as input
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
sys.path.append('%s/../../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
seed = 19260817


# train vae model for one epoch
sys.path.append('%s/../' % os.path.dirname(os.path.realpath('__file__')))
from model import MolVAE
sys.path.append('%s/../../util/' % os.path.dirname(os.path.realpath('__file__')))
from pytorch_initializer import weights_init
from train_util import get_batch_input_vae, Prepare_data


#q(z|x)
def epoch_train(phase, epoch, ae, sample_idxes, data_binary, data_masks, data_property, cmd_args, optimizer_vae=None):
    total_loss = []
    pbar = tqdm(range(0, (len(sample_idxes) + (cmd_args.batch_size - 1) * (optimizer_vae is None)) // cmd_args.batch_size), unit='batch')

    if phase == 'train' and optimizer_vae is not None:
        ae.train()
    else:
        ae.eval()

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * cmd_args.batch_size : (pos + 1) * cmd_args.batch_size]
        x_inputs, y_inputs,v_tb, v_ms, t_y = get_batch_input_vae(selected_idx, data_binary, data_masks, data_property)  # no grad for evaluate mode.
        loss_list = ae.forward(x_inputs, y_inputs,v_tb,v_ms, t_y)
        loss_vae = loss_list[0] + loss_list[1]

        perp = loss_list[0].data.cpu().numpy()[0] # reconstruction loss
        kl = loss_list[1].data.cpu().numpy()


        minibatch_vae_loss = loss_vae.data.cpu().numpy()
        pbar.set_description('At epoch: %d  %s vae loss: %0.5f perp: %0.5f kl: %0.5f' % (epoch, phase, minibatch_vae_loss, perp, kl))


        if optimizer_vae is not None:
            assert len(selected_idx) == cmd_args.batch_size
            optimizer_vae.zero_grad()
            loss_vae.backward(retain_graph=True)
            optimizer_vae.step()


        total_loss.append(np.array([minibatch_vae_loss, perp, kl]) * len(selected_idx))

        n_samples += len(selected_idx)

    if optimizer_vae is None:
        assert n_samples == len(sample_idxes)

    total_loss = np.array(total_loss)

    avg_loss = np.sum(total_loss, 0) / n_samples
    return ae, avg_loss



def main():

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not os.path.exists(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)


    sys.path.append('%s/util/' % os.path.dirname(os.path.realpath('__file__')))
    getting_data = Prepare_data(cmd_args)
    train_binary_x, train_masks, valid_binary_x, valid_masks, train_y, valid_y = getting_data.load_data()
    print('num_train: %d\tnum_valid: %d' % (train_y.shape[0], valid_binary_x.shape[0]))

    ae = MolVAE()
    if cmd_args.mode == 'gpu':
        ae = ae.cuda()

    kl = []
    prep = []

    cmd_args.saved_model = cmd_args.save_dir + '/epoch-best.model'
    if cmd_args.saved_model  is not None and cmd_args.saved_model != '':
        if os.path.isfile(cmd_args.saved_model):
            print('loading model from %s' % cmd_args.saved_model)
            ae.load_state_dict(torch.load(cmd_args.saved_model))



    assert cmd_args.encoder_type == 'cnn'
    optimizer_vae = optim.Adam(ae.parameters(), lr=cmd_args.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer_vae, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.001)


    sample_idxes = list(range(train_binary_x.shape[0]))
    best_valid_loss = None

    for epoch in range(347, cmd_args.num_epochs):
        random.shuffle(sample_idxes)

        ## update the vae

        ae, vae_loss = epoch_train('train',epoch, ae, sample_idxes, train_binary_x, train_masks, train_y,cmd_args, optimizer_vae)
        print('>>>>average \033[92mtraining\033[0m of epoch %d: loss %.5f perp %.5f kl %.5f' % (epoch, vae_loss[0], vae_loss[1], vae_loss[2]))
        kl.append(vae_loss[2])
        prep. append(vae_loss[1])

        if epoch % 1 == 0:
            _, valid_loss = epoch_train('valid', epoch,  ae, list(range(valid_binary_x.shape[0])), valid_binary_x, valid_masks, valid_y,cmd_args)
            print('>>>>average \033[93mvalid\033[0m of epoch %d: loss %.5f perp %.5f kl %.5f' % (epoch, valid_loss[0], valid_loss[1], valid_loss[2]))
            valid_loss = valid_loss[0]
            lr_scheduler.step(valid_loss)
            torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-%d.model' % epoch)

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print('saving to best model since this is the best valid loss so far.----')
                torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-best.model')
                np.save(cmd_args.save_dir + '/kl.npy', kl)
                np.save(cmd_args.save_dir + '/prep.npy', prep)

    np.save(cmd_args.save_dir + '/kl.npy', kl)
    np.save(cmd_args.save_dir + '/prep.npy', prep)

if __name__ == '__main__':
   main()

