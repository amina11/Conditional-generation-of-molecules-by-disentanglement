
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

## reconstruction utility functions
sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
from cmd_args import cmd_args
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks
from tree_walker import OnehotBuilder, ConditionalDecoder
import cfg_parser as parser

def parse_single(smiles, grammar):
    ts = parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1
    n = AnnotatedTree2MolTree(ts[0])
    return n

def parse(chunk, grammar):
    return [parse_single(smiles, grammar) for smiles in chunk]


import glob
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import sascorer
def get_label_from_string(string):
    y = []
    index = []
    for i in range(len(string)):
        if isinstance(string[i], list):
            smiles_string = string[i][0]
        else:
            smiles_string = string[i]
        
        
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            continue
        smiles =  Chem.MolToSmiles(mol)
        if smiles is None: 
            continue
        mol = Chem.MolFromSmiles(smiles)  
        logP = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
        qed = QED.qed(mol) 
        y.append([qed, sa_score, logP]) 
        index.append(i)
    return y, index


def decode_chunk(raw_logits, use_random, decode_times):
    tree_decoder = create_tree_decoder()    
    chunk_result = [[] for _ in range(raw_logits.shape[1])]
        
    for i in tqdm(range(raw_logits.shape[1])):
        pred_logits = raw_logits[:, i, :]
        walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

        for _decode in range(decode_times):
            new_t = Node('smiles')
            try:
                tree_decoder.decode(new_t, walker)
                sampled = get_smiles_from_tree(new_t)
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    print('Warning, decoder failed with', ex)
                # failed. output a random junk.
                import random, string
                sampled = 'JUNK' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))

            chunk_result[i].append(sampled)
    return chunk_result

def batch_decode(raw_logits, use_random, decode_times):
    size = int((raw_logits.shape[1] + 7) / 8)

    logit_lists = []
    for i in range(0, raw_logits.shape[1], size):
        if i + size < raw_logits.shape[1]:
            logit_lists.append(raw_logits[:, i : i + size, :])
        else:
            logit_lists.append(raw_logits[:, i : , :])

    result_list = Parallel(n_jobs=-1)(delayed(decode_chunk)(logit_lists[i], use_random, decode_times) for i in range(len(logit_lists)))
    return [_1 for _0 in result_list for _1 in _0]       


def reconstruct_single(model, smiles, labels, encode_times, decode_times):
    print('a chunk starts...')
    decode_result = []

    chunk = smiles
    chunk_result = [[] for _ in range(len(chunk))]
    for _encode in range(encode_times):
        z1 = model.encode(chunk, use_random=True)
        this_encode = []
        encode_id, encode_total = _encode + 1, encode_times
        for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
            ):
            _result = model.decode(z1, labels, use_random=True)
            print(_result)
            for index, s in enumerate(_result):
                chunk_result[index].append(s)

    decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)
    return decode_result






def reconstruct(model, smiles, labels, chunk_size, encode_times, decode_times):
    chunk_result = Parallel(n_jobs=1)(
        delayed(reconstruct_single)(model, smiles[chunk_start: chunk_start + chunk_size], labels[chunk_start: chunk_start + chunk_size], encode_times, decode_times)
        for chunk_start in range(0, len(smiles), chunk_size)
    )

    decode_result = [_1 for _0 in chunk_result for _1 in _0]
    assert len(decode_result) == len(smiles)
    return decode_result

def save_decode_result(decode_result, smiles, filename):
    with open(filename, 'w') as fout:
        for s, cand in zip(smiles, decode_result):
            print(','.join([s] + cand), file=fout)
            
def cal_accuracy(decode_result,smiles):
    accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    junk = [sum([1 for c in cand if c.startswith('JUNK')]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    return (sum(accuracy) * 1.0 / len(accuracy)), (sum(junk) * 1.0 / len(accuracy))
            




def cal_valid_prior(model, latent_dim, labels, nb_latent_point, sample_times, chunk_size, sigma):
    import rdkit
    from rdkit import Chem
    whole_valid, whole_total = 0, 0
    valid_smile = []
    pbar = tqdm(list(range(0, nb_latent_point, chunk_size)), desc='decoding')
    for start in pbar:
        end = min(start + chunk_size, nb_latent_point)
        latent_point = np.random.normal(0, sigma, size=(end - start, latent_dim))
        latent_point = latent_point.astype(np.float32)
        #y = np.tile(labels, (nb_latent_point,1)) 
        y = labels[:end-start].astype(np.float32)
        raw_logits = model.pred_raw_logits(latent_point, y)
        decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)
        for i in range(end - start):
            for j in range(sample_times):
                s = decoded_array[i][j]
                if not s.startswith('JUNK') and Chem.MolFromSmiles(s) is not None:
                    whole_valid += 1
                    valid_smile.append(s)
                whole_total += 1
        pbar.set_description('valid : total = %d : %d = %.5f' % (whole_valid, whole_total, whole_valid * 1.0 / whole_total))
    return 1.0 * whole_valid / whole_total, whole_valid, whole_total, valid_smile


# this function will take a target label, sample 50 times z fromt the prior, decoede it, check the molecules fall in range
# of $differences_range from the center(target label) , return the molecules and their labels in that region as well as all 
# valid molecules
def near_by_moles(model, desired_label, differences_range, center, logP_normalizer):
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw
    sigma = 0.05
    nb_latent_point = 50
    latent_point = np.random.normal(0, sigma, size=(nb_latent_point, cmd_args.latent_dim))
    latent_point = latent_point.astype(np.float32)
    fixed_labels = (np.tile(desired_label, (nb_latent_point, 1))).astype(float)
    fixed_y_decoded_smiles = model.decode(latent_point,  5 *fixed_labels, use_random=True)
    mol_list = []
    index_valid = []
    for i in range(nb_latent_point):
        s_m = Chem.MolFromSmiles(fixed_y_decoded_smiles[i])

        if s_m is None:
            continue
        m_s = Chem.MolToSmiles(s_m)

        if m_s is None:
            continue
        mol_list.append(s_m)
        index_valid.append(i)

    img_1 = Draw.MolsToGridImage(mol_list, molsPerRow=7,subImgSize=(100, 100), useSVG=True)
    b = np.array(fixed_y_decoded_smiles)
    valid_decoded_smile = b[index_valid]
    print(len(valid_decoded_smile))
    valid_y = np.array(get_label_from_string(np.array(fixed_y_decoded_smiles)[index_valid])[0])[:,2]
    target = desired_label* np.array([logP_normalizer]).astype(np.float32)
    c = np.abs(valid_y - center) / np.abs(center)
    y_near_molecule = valid_y[c<differences_range] 
    near_molecule = valid_decoded_smile[c<differences_range] 
    
    try:
        mol_list_2 = []
        for i in range(near_molecule.shape[0]):
            s_m_2 = Chem.MolFromSmiles(near_molecule[i])
    
            if s_m_2 is None:
                continue
            m_s_2 = Chem.MolToSmiles(s_m_2)
    
            if m_s_2 is None:
                continue
            mol_list_2.append(s_m_2)
        print('there is', len(mol_list_2), 'molecules are in this range')
        print('desired labels is:', target)
        print('the labels of the near by molecules are :', y_near_molecule)
        img_2 = Draw.MolsToGridImage(mol_list_2, molsPerRow=7,subImgSize=(100, 100), useSVG=True)
    except near_molecule.shape[0] == 0:
        print('in this range no valid molecules are generated')
    return img_1, img_2, valid_decoded_smile, valid_y, near_molecule, y_near_molecule

# for given set of labels, for each, sample 50 z, return labels for corresponding to the vaid mols
# get mean label of generated mold for given label as input
def iny_outy(target_labels, sigma, model, nb_latent_point):
    input_label = []
    output_label =[]
    mean_output_label = []
   

    for i in range(target_labels.shape[0]):
        labels = np.tile(target_labels[i], (nb_latent_point, 1))
        latent_point = np.random.normal(0, sigma, size=(nb_latent_point, cmd_args.latent_dim))
        z = latent_point.astype(np.float32)
        decoded_string = model.decode(z, labels.astype(float), use_random=True)
        y,i = get_label_from_string(decoded_string)
        mean_output_label.append(np.median(np.array(y)[:,2]))
        input_label=np.append(input_label, np.array(labels[np.array(i)]))
        output_label=np.append(output_label, np.array(y)[:,2])
    return input_label, output_label, mean_output_label


# for given set of labels, for each, sample 50 z, return labels for corresponding to the vaid mols
# get mean label of generated mold for given label as input
def iny_outy_posterior(target_labels, smiles, model):
    input_label = []
    output_label =[]
    mean_output_label = []
    for i in range(target_labels.shape[0]):
        labels = np.tile(target_labels[i], (50, 1))
        x = smiles
        z = model.encode(x, use_random=True)
        decoded_string = model.decode(z, labels.astype(float), use_random=True)
        y,index = get_label_from_string(decoded_string)
        mean_output_label.append(np.mean(np.array(y)[:,2]))
        input_label=np.append(input_label, np.array(labels[np.array(index)]))
        output_label=np.append(output_label, np.array(y)[:,2])
    return input_label, output_label, mean_output_label



## this function gives error, run parse_many instead!
'''
def parse(chunk, grammar):
    size = 2
    result_list = Parallel(n_jobs=-1)(delayed(parse_many)(chunk[i: i + size], grammar) for i in range(0, len(chunk), size))
    return [_1 for _0 in result_list for _1 in _0]
'''    


'''
def reconstruct_single(model, smiles, labels, encode_times, decode_times):
    print('a chunk starts...')
    decode_result = []

    chunk = smiles
    chunk_result = [[] for _ in range(len(chunk))]
    for _encode in range(encode_times):
        z1 = model.encode(chunk, labels, use_random=True)
        this_encode = []
        encode_id, encode_total = _encode + 1, encode_times
        for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
            ):
            _result = model.decode(z1, labels, use_random=True)
            print(_result)
            for index, s in enumerate(_result):
                chunk_result[index].append(s)

    decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)
    return decode_result


def reconstruct(model, smiles, labels, chunk_size, encode_times, decode_times):
    chunk_result = Parallel(n_jobs=1)(
        delayed(reconstruct_single)(model, smiles[chunk_start: chunk_start + chunk_size], labels[chunk_start: chunk_start + chunk_size], encode_times, decode_times)
        for chunk_start in range(0, len(smiles), chunk_size)
    )

    decode_result = [_1 for _0 in chunk_result for _1 in _0]
    assert len(decode_result) == len(smiles)
    return decode_result    
'''

'''
from reconstruct_util import decode_chunk
from collections import Counter
import rdkit
from rdkit import Chem
def cal_valid_prior(model, latent_dim, labels, sigma, nb_latent_point):
    whole_valid, whole_total = 0, 0
    latent_point = np.random.normal(0, sigma, size=(nb_latent_point, latent_dim))
    #latent_point = np.tile(latent_point, (nb_latent_point, 1))
    y = labels.astype(np.float32)
    latent_point = latent_point.astype(np.float32)
    raw_logits = model.pred_raw_logits(latent_point, y)
    decoded_array = decode_chunk(raw_logits, True, decode_times=sample_times)

    decode_list = []
    for i in range(nb_latent_point):
        c = Counter()
        for j in range(sample_times):
            c[decoded_array[i][j]] += 1
        decoded = c.most_common(1)[0][0]
        if decoded.startswith('JUNK'):
            continue
            
        mol = Chem.MolFromSmiles(decoded)
        if mol is None:
            continue
            
        m = Chem.MolToSmiles(mol)
        if m is None:
            print(i)
            continue
            
        decode_list.append(decoded)
        

    valid_prior_save_file =  cmd_args.saved_model + '-sampled_prior.txt'
    with open(valid_prior_save_file, 'w') as fout:
        for row in decode_list:
            fout.write('%s\n' % row)
    return decode_list      
'''
