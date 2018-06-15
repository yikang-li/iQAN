import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import pdb

from vqa.models import fusion
from vqg.models import vec2seq
from .beam_search import CaptionGenerator

from ..lib.utils import weights_normal_init

__DEBUG__ = False

class VQG_Abstract(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Abstract, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        # Modules for attention
        self.embedding_answers = nn.Embedding(self.num_classes, self.opt['dim_a'])
        # Modules for Question Generation
        self.question_generation = getattr(vec2seq, opt['vec2seq']['arch'])(vocab_words, opt['vec2seq'])
        self.is_testing = False


# we directly remove the attention module, and concatenate the visual/answer feature for generation
class VQG_Merge_Abstract(VQG_Abstract):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Merge_Abstract, self).__init__(opt, vocab_words, vocab_answers)


    def prepare_feature(self, input_v, input_a):
        raise NotImplementedError

    def forward(self, input_v, input_a, input_q = None):
        x_a_vec = self.embedding_answers(input_a.view(1, -1)).squeeze()
        x_v_vec, x_a_vec = self.prepare_feature(input_v, x_a_vec)
        if self.is_testing:
            x = self.question_generation.generate(x_v_vec, x_a_vec)
        else:
            x = self.question_generation(x_v_vec, x_a_vec, input_q)
        return x

class VQG_Merge_noAtt_baseline(VQG_Merge_Abstract):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Merge_noAtt_baseline, self).__init__(opt, vocab_words, vocab_answers)
        self.img_transform = nn.Linear(opt['dim_v'], opt['vec2seq']['dim_v'])
        self.answer_transform = nn.Linear(opt['dim_a'], opt['vec2seq']['dim_a'])

    def prepare_feature(self, input_v, input_a):
        v_feat = self.img_transform(input_v)
        a_feat = self.answer_transform(input_a)
        return v_feat, a_feat

class VQG_Att_Merge_Abstract(VQG_Merge_Abstract):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Att_Merge_Abstract, self).__init__(opt, vocab_words, vocab_answers)
        self.img_transform = nn.Linear(opt['dim_v'], opt['vec2seq']['dim_v'])
        self.answer_transform = nn.Linear(opt['dim_a'], opt['vec2seq']['dim_a'])
        # Modules for attention
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_v'], 1, 1)
        self.linear_a_att = nn.Linear(self.opt['dim_a'], self.opt['attention']['dim_a'])
        self.embedding_answers = nn.Embedding(self.num_classes, self.opt['dim_a'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'],1, 1, 1)

    def _fusion_att(self, x_v, x_a):
        raise NotImplementedError

    def _attention(self, x_v, x_a):
        batch_size = x_v.size(0)
        width = x_v.size(2)
        height = x_v.size(3)

        x_v_orgin = x_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v_orgin = x_v_orgin.transpose(1,2)
        # Process visual before fusion
        x_v = F.relu(self.conv_v_att(x_v))
        x_v = x_v.view(batch_size, self.opt['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1,2)

        x_a = F.relu(self.linear_a_att(x_a))
        x_a = x_a.view(batch_size, 1,
                       self.opt['attention']['dim_a'])
        x_a = x_a.expand(batch_size,
                         width * height,
                         self.opt['attention']['dim_a'])

        # First multimodal fusion
        x_att = F.relu(self._fusion_att(x_v, x_a))
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width,
                           height,
                           self.opt['attention']['dim_mm']).transpose(2,3).transpose(1,2)
        x_att = self.conv_att(x_att).view(batch_size, width * height)
        x_att = F.softmax(x_att)
        x_att = x_att.view(batch_size,width * height,1)
        x_att = x_att.expand(batch_size, width * height, self.opt['dim_v'])
        # Apply attention vectors to input_v
        x_v_att = torch.mul(x_att, x_v_orgin)
        x_v_att = x_v_att.sum(1)
        x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
        return x_v_att

    def prepare_feature(self, input_v, input_a):
        v_feat = self.img_transform(self._attention(input_v, input_a))
        a_feat = self.answer_transform(input_a)
        return v_feat, a_feat

class VQG_Att_Merge_Mul(VQG_Att_Merge_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_v']  = opt['attention']['dim_h']
        opt['attention']['dim_a']  = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(VQG_Att_Merge_Mul, self).__init__(opt, vocab_words, vocab_answers)

    def _fusion_att(self, x_v, x_a):
        x_att = torch.mul(x_v, x_a)
        return x_att

class VQG_Att_Merge_Mutan(VQG_Att_Merge_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_v'] = opt['attention']['dim_hv']
        opt['attention']['dim_q'] = opt['attention']['dim_hq']
        opt['attention']['dim_a'] = opt['attention']['dim_q']
        super(VQG_Att_Merge_Mutan, self).__init__(opt, vocab_words, vocab_answers)
        # Modules for classification
        self.fusion_att = fusion.MutanFusion2d(self.opt['attention'],
                                               visual_embedding=True,
                                               question_embedding=False) # actually, the answer embedding

    def _fusion_att(self, x_v, x_a):
        return self.fusion_att(x_v, x_a)


class VQG_Att_Merge_Concat(VQG_Att_Merge_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_mm'] = opt['attention']['dim_v'] + opt['attention']['dim_a']
        super(VQG_Att_Merge_Concat, self).__init__(opt, vocab_words, vocab_answers)

    def _fusion_att(self, x_v, x_a):
        x_att = torch.cat([x_v, x_a], 2)
        return x_att


class VQG_Fusion_Abstract(VQG_Abstract):
    '''
    This scheme use the [fusion + generation] pipeline
    '''
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Fusion_Abstract, self).__init__(opt, vocab_words, vocab_answers)
        self.linear_va_transform = None
        self.concept_classifier = None
        self.concept_num = opt.get('concept_num', 0)

    def _fusion_feat(self, x_v, x_a):
        raise NotImplementedError

    def _generate_qestion(self, x_va, input_q = None):

        # transform the fused feature to word_embedding domain (length = opts['vec2seq']['dim_embedding'])
        x_va = getattr(F, self.opt['vec2seq']['activation'])(self.linear_va_transform(x_va))
        if self.is_testing:
            if x_va.size(0) == 1:
                x = self.question_generation.beam_search(x_va)
            else:
                x = self.question_generation.generate(x_va)

            return x
        else:
            x = self.question_generation(x_va, input_q)

        if self.concept_num > 0:
            concept_pred = self.concept_classifier(x_va)
            return x, concept_pred
        else:
            return x

    def forward(self, input_v, input_a, input_q = None):
        x_a_vec = self.embedding_answers(input_a.view(1, -1)).squeeze()
        x_va = self._fusion_feat(input_v, x_a_vec)
        x = self._generate_qestion(x_va, input_q)
        return x


class VQG_Att_Fusion_Abstract(VQG_Fusion_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Att_Fusion_Abstract, self).__init__(opt, vocab_words, vocab_answers)

        self.linear_va_transform = nn.Linear(
                        self.opt['dim_v'] + self.opt['dim_a'], 
                        self.opt['vec2seq']['dim_embedding'])
        if self.concept_num > 0:
            self.concept_classifier = nn.Linear(
                        self.opt['vec2seq']['dim_embedding'], self.concept_num)
        # Modules for attention
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_v'], 1, 1)
        self.linear_a_att = nn.Linear(self.opt['dim_a'], self.opt['attention']['dim_a'])
        self.embedding_answers = nn.Embedding(self.num_classes, self.opt['dim_a'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'],1, 1, 1)

    def _fusion_att(self, x_v, x_a):
        raise NotImplementedError

    def _attention(self, input_v, x_a):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)
        x_v = input_v
        x_v = self.conv_v_att(x_v)
        x_v = getattr(F, self.opt['attention']['activation'])(x_v)
        x_v = x_v.view(batch_size,
                       self.opt['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1,2)

        x_a = self.linear_a_att(x_a)
        x_a = getattr(F, self.opt['attention']['activation'])(x_a)
        x_a = x_a.view(batch_size, 1,
                       self.opt['attention']['dim_a'])
        x_a = x_a.expand(batch_size,
                         width * height,
                         self.opt['attention']['dim_a'])

        # First multimodal fusion
        x_att = self._fusion_att(x_v, x_a)
        x_att = getattr(F, self.opt['attention']['activation'])(x_att)
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width,
                           height,
                           self.opt['attention']['dim_mm']).transpose(2,3).transpose(1,2)
        x_att = self.conv_att(x_att).view(batch_size, width * height)
        x_att = F.softmax(x_att)
        x_att = x_att.view(batch_size,width * height,1)
        x_att = x_att.expand(batch_size, width * height, self.opt['dim_v'])
        # Apply attention vectors to input_v
        x_v_orgin = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v_orgin = x_v_orgin.transpose(1,2)
        x_v_att = torch.mul(x_att, x_v_orgin)
        x_v_att = x_v_att.sum(1)
        x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
        x_v_att = getattr(F, self.opt['attention']['activation'])(x_v_att)
        return x_v_att

    def _fusion_feat(self, x_v, x_a):
        x_att = self._attention(x_v, x_a)
        x_va = torch.cat([x_att, x_a], 1)
        return x_va


class VQG_Att_Fusion_Mul(VQG_Att_Fusion_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_v']  = opt['attention']['dim_h']
        opt['attention']['dim_a']  = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(VQG_Att_Fusion_Mul, self).__init__(opt, vocab_words, vocab_answers)

    def _fusion_att(self, x_v, x_a):
        x_att = torch.mul(x_v, x_a)
        return x_att

    


class VQG_Att_Fusion_Concat(VQG_Att_Fusion_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_mm'] = opt['attention']['dim_v'] + opt['attention']['dim_a']
        super(VQG_Att_Fusion_Concat, self).__init__(opt, vocab_words, vocab_answers)
        

    def _fusion_att(self, x_v, x_a):
        x_att = torch.cat([x_v, x_a], 2)
        return x_att

class VQG_Att_Fusion_Mutan(VQG_Att_Fusion_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_v'] = opt['attention']['dim_hv']
        opt['attention']['dim_q'] = opt['attention']['dim_hq']
        opt['attention']['dim_a'] = opt['attention']['dim_q']
        super(VQG_Att_Fusion_Mutan, self).__init__(opt, vocab_words, vocab_answers)
        # Modules for classification
        self.fusion_att = fusion.MutanFusion2d(self.opt['attention'],
                                               visual_embedding=True,
                                               question_embedding=False) # actually, the answer embedding

    def _fusion_att(self, x_v, x_a):
        return self.fusion_att(x_v, x_a)
