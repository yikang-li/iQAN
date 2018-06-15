import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.convnets import ResNet


import copy
import pdb

import vqa.models as models_vqa
import vqg.models as models_vqg

from vqg.models.vec2seq import process_lengths_sort, process_lengths


class VQA_Conv_Abstract(nn.Module):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQA_Conv_Abstract, self).__init__()
        self.shared_conv_layer = None
        self.model_vqa = getattr(models_vqa, opt['vqa']['arch'])(
            opt['vqa'], vocab_words, vocab_answers)

    def forward(self, input_v, input_q, target_a):
        input_v = self.shared_conv_layer(input_v)
        answers = self.model_vqa(input_v, input_q)
        return answers


class VQA_ResNet(VQA_Conv_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQA_ResNet, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = ResNet(opt['arch_resnet'], opt['pooling'])
    

class VQA_Conv(VQA_Conv_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQA_Conv, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = ResNet(opt['arch_resnet']).layer4



def prepare_input(input_visual, mode):
    if mode == 'noatt' and input_visual.dim() == 4:
        return F.avg_pool2d(input_visual, input_visual.size(2)).squeeze()
    else:
        return input_visual

