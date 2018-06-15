import torch
import torch.nn as nn
import torch.nn.functional as F


import copy
import pdb

from vqa.lib import utils
from vqa.models import seq2vec
from vqa.models import fusion

class Single_Attention_Module(nn.Module):
    def __init__(self, opt={}):
        self.opt = opt
        super(Single_Attention_Module, self).__init__()
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_h'], 1, 1)
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_h'],
                                  self.opt['attention']['nb_glimpses'], 1, 1)

    def forward(self, input_v):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)
        x_v = input_v
        x_v = F.dropout(x_v,
                        p=self.opt['attention']['dropout'],
                        training=self.training)
        x_v = self.conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = getattr(F, self.opt['attention']['activation'])(x_v)
        # Process attention vectors
        x_att = F.dropout2d(x_v,
                          p=self.opt['attention']['dropout'],
                          training=self.training)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.opt['attention']['nb_glimpses'],
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1,2)

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1).squeeze(1)
            list_v_att.append(x_v_att)

        return list_v_att

class Abstract_Attention_Module(nn.Module):
    def __init__(self, opt={}, use_linear=True):
        # Modules for attention
        self.opt = opt
        super(Abstract_Attention_Module, self).__init__()
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_v'], 1, 1)
        self.use_linear = use_linear
        if use_linear:
            self.linear_q_att = nn.Linear(self.opt['dim_q'],
                                        self.opt['attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'],
                                  self.opt['attention']['nb_glimpses'], 1, 1)


    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    # attention to several glimpes
    def forward(self, input_v, x_q):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)
        x_v = input_v
        x_v = F.dropout(x_v,
                        p=self.opt['attention']['dropout_v'],
                        training=self.training)
        x_v = self.conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = getattr(F, self.opt['attention']['activation_v'])(x_v)
        x_v = x_v.view(batch_size,
                       self.opt['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1,2)

        # Process question before fusion

        if self.use_linear:
            x_q = F.dropout(x_q, p=self.opt['attention']['dropout_q'],
                             training=self.training)
            x_q = self.linear_q_att(x_q)
            if 'activation_q' in self.opt['attention']:
                x_q = getattr(F, self.opt['attention']['activation_q'])(x_q)



        x_q = x_q.view(batch_size,
                       1,
                       self.opt['attention']['dim_q'])
        x_q = x_q.expand(batch_size,
                         width * height,
                         self.opt['attention']['dim_q'])

        # First multimodal fusion
        x_att = self._fusion_att(x_v, x_q)

        if 'activation_mm' in self.opt['attention']:
            x_att = getattr(F, self.opt['attention']['activation_mm'])(x_att)

        # Process attention vectors
        x_att = F.dropout(x_att,
                          p=self.opt['attention']['dropout_mm'],
                          training=self.training)
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width,
                           height,
                           self.opt['attention']['dim_mm']) 
        x_att = x_att.transpose(2,3).transpose(1,2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.opt['attention']['nb_glimpses'],
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1,2)

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
            list_v_att.append(x_v_att)

        return list_v_att


class MLB(Abstract_Attention_Module):
    def __init__(self, opt={}, use_linear=True):
        # TODO: deep copy ?
        opt['attention']['dim_v']  = opt['attention']['dim_h']
        opt['attention']['dim_q']  = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(MLB, self).__init__(opt, use_linear)
        

    def _fusion_att(self, x_v, x_q):
        x_att = torch.mul(x_v, x_q)
        return x_att

class Mutan(Abstract_Attention_Module):
    def __init__(self, opt={}, use_linear=True):
        # TODO: deep copy ?
        opt['attention']['dim_v'] = opt['attention']['dim_hv']
        opt['attention']['dim_q'] = opt['attention']['dim_hq']
        super(Mutan, self).__init__(opt, use_linear)
        # Modules for classification
        self.fusion_att = fusion.MutanFusion2d(self.opt['attention'],
                                               visual_embedding=False,
                                               question_embedding=False)

    def _fusion_att(self, x_v, x_q):
        return self.fusion_att(x_v, x_q)
