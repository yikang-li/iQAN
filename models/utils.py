# Common functions shared by different modules
import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import pdb


__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def select_optimizer(optimizer_name, params, lr=0.1, weight_decay=0.0):
    return __optimizers[optimizer_name](params, lr=lr, weight_decay=weight_decay)


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    if param_group[key] != setting[key]:
                        print('OPTIMIZER modified- setting [\'%s\']' % key)
                        param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def translate_tokens(token_seq, wid_to_word, end_id = 0):
    sentence = []
    for wid in token_seq:
        if wid == end_id:
            break
        sentence.append(wid_to_word[wid])
    return sentence


def calculate_bleu_score(generated_question_tensor, reference_question_tensor, wid_to_word):
    batch_size = generated_question_tensor.size(0)
    bleu_score = 0.
    for j in range(batch_size):
            sampled_aqa = []
            new_question = translate_tokens(generated_question_tensor[j].tolist(), wid_to_word)
            ref_question = translate_tokens(reference_question_tensor[j][1:].tolist(), wid_to_word)
            bleu_score += float(sentence_bleu([' '.join(ref_question)], ' '.join(new_question), 
                                    smoothing_function=SmoothingFunction().method4))
    return bleu_score / batch_size