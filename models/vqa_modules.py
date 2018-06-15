import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from vqa.lib import utils
from vqa.models import seq2vec
from vqa.models import fusion

class VQA_Abstract(nn.Module):

    def __init__(self, opt={}, vocab_answers=[]):
        super(VQA_Abstract, self).__init__()
        self.opt = opt
        self.num_classes = len(vocab_answers)
        # VQA Modules
        self.linear_q_fusion = None
        self.linear_classif = None

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def forward(self, x_v, x_q):
        # Process question
        x_q = F.dropout(x_q,
                        p=self.opt['vqa']['fusion']['dropout_q'],
                        training=self.training)
        x_q = self.linear_q_fusion(x_q)
        if 'activation_q' in self.opt['vqa']['fusion']:
            x_q = getattr(F, self.opt['vqa']['fusion']['activation_q'])(x_q)
        # Second multimodal fusion
        x = self._fusion_classif(x_v, x_q)
        if 'activation' in self.opt['vqa']['classif']:
            x = getattr(F, self.opt['vqa']['classif']['activation'])(x)
        x = F.dropout(x,
                      p=self.opt['vqa']['classif']['dropout'],
                      training=self.training)
        x = self.linear_classif(x)
        return x


class MLB(VQA_Abstract):

    def __init__(self, opt={}, vocab_answers=[]):
        # TODO: deep copy ?
        super(MLB, self).__init__(opt, vocab_answers)
        # Modules for classification
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['vqa']['fusion']['dim_h']
                                         * self.opt['attention']['nb_glimpses'])
        self.linear_classif = nn.Linear(self.opt['vqa']['fusion']['dim_h']
                                        * self.opt['attention']['nb_glimpses'],
                                        self.num_classes)

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.mul(x_v, x_q)
        return x_mm


class Mutan(VQA_Abstract):

    def __init__(self, opt={}, vocab_answers=[]):
        # TODO: deep copy ?
        super(Mutan, self).__init__(opt, vocab_answers)
        # Modules for classification
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['vqa']['fusion']['dim_hq'])
        self.linear_classif = nn.Linear(self.opt['vqa']['fusion']['dim_mm'],
                                        self.num_classes)
        self.fusion_classif = fusion.MutanFusion(self.opt['vqa']['fusion'],
                                                 visual_embedding=False,
                                                 question_embedding=False)

    def _fusion_classif(self, x_v, x_q):
        return self.fusion_classif(x_v, x_q)
