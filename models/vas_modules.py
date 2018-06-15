import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from . import attention_modules


__DEBUG__ = False

# visual answers sampling
class VAS_Abstract(nn.Module):

    def __init__(self, opt={}, vocab_answers=[]):
        super(VAS_Abstract, self).__init__()
        self.opt = opt
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        # Modules for Question Generation
        self.answer_sampler = nn.Linear(opt['vas']['dim_h'], self.num_classes)
        self.is_testing = False

class noAtt(VAS_Abstract):
    def __init__(self, opt={}, vocab_answers=[]):
        super(noAtt, self).__init__(opt, vocab_answers)
        self.feature_transform = nn.Linear(opt['dim_v'], opt['vas']['dim_h'])
        self.opt = opt

    def forward(self, input_v):
        batch_size=input_v.size(0)
        if input_v.dim() > 2:
            input_v = F.avg_pool2d(input_v, (input_v.size(2), input_v.size(3)), ).view(batch_size, -1)
        if 'dropout' in self.opt.keys():
            input_v = F.dropout(input_v, self.opt['vas']['dropout'])
        x = self.feature_transform(input_v)
        x = getattr(F, self.opt['vas']['activation'])(x)
        if 'dropout' in self.opt.keys():
            x = F.dropout(x, self.opt['vas']['dropout'])
        x = self.answer_sampler(x)
        return x

class att(VAS_Abstract):
    def __init__(self, opt={}, vocab_answers=[]):
        super(att, self).__init__(opt, vocab_answers)
        self.attention = getattr(attention_modules, opt['attention']['arch'])(opt)
        #  self.conv_v_att = nn.Conv2d(opt['dim_v'], self.opt['att_dim_v'], 1, 1)
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'], self.opt['vas']['dim_h'])
            for i in range(self.opt['attention']['nb_glimpses'])])
        # Modules for Question Generation
        self.answer_sampler = nn.Linear(opt['vas']['dim_h'] * self.opt['attention']['nb_glimpses'], 
            self.num_classes)

    def _fusion_glimpses(self, list_v_att,):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=self.opt['vas']['dropout'],
                            training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            x_v = getattr(F, self.opt['vas']['activation'])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        return x_v


    def forward(self, input_v):
        v_att_list = self.attention(input_v)
        v_att = self._fusion_glimpses(v_att_list)
        if 'dropout' in self.opt.keys():
            v_att = F.dropout(v_att, self.opt['vas']['dropout'])
        y = self.answer_sampler(v_att)
        return y


