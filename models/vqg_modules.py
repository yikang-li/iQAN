import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import pdb

from vqa.models import fusion
from models import vec2seq
from models.beam_search import CaptionGenerator

__DEBUG__ = False

class VQG_Abstract(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQG_Abstract, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)

        # Modules for Question Generation
        self.question_generation = getattr(vec2seq, opt['vqg']['vec2seq']['arch'])(vocab_words, opt['vqg']['vec2seq'])
        self.is_testing = False


# we directly remove the attention module, and concatenate the visual/answer feature for generation
class Merge(VQG_Abstract):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(Merge, self).__init__(opt, vocab_words, vocab_answers)
        if self.opt['vqa']['arch'] == 'Mutan':
            dim_h = int(self.opt['vqa']['fusion']['dim_hv']
                          / opt['attention']['nb_glimpses']) * opt['attention']['nb_glimpses']
        else: # MLB
            dim_h = self.opt['vqa']['fusion']['dim_h'] * opt['attention']['nb_glimpses']

        self.img_transform = nn.Linear(dim_h, opt['vqg']['vec2seq']['dim_v'])
        self.answer_transform = nn.Linear(opt['dim_a'], opt['vqg']['vec2seq']['dim_a'])

    def prepare_feature(self, input_v, input_a):
        input_v = F.dropout(input_v,
                      p=self.opt['vqg']['dropout_v'],
                      training=self.training)
        v_feat = self.img_transform(input_v)
        if 'activation_v' in self.opt['vqg']:
            v_feat = getattr(F, self.opt['vqg']['activation_v'])(v_feat)
        v_feat = F.dropout(v_feat,
                      p=self.opt['vqg']['dropout_v'],
                      training=self.training)

        input_a = F.dropout(input_a,
                      p=self.opt['vqg']['dropout_a'],
                      training=self.training)
        a_feat = self.answer_transform(input_a)
        if 'activation_a' in self.opt['vqg']:
            a_feat = getattr(F, self.opt['vqg']['activation_a'])(a_feat)
        a_feat = F.dropout(a_feat,
                      p=self.opt['vqg']['dropout_a'],
                      training=self.training)
        return v_feat, a_feat

    def forward(self, input_v, x_a_vec, input_q = None):
        x_v_vec, x_a_vec = self.prepare_feature(input_v, x_a_vec)
        if self.is_testing:
            x = self.question_generation.generate(x_v_vec, x_a_vec)
        else:
            x = self.question_generation(x_v_vec, x_a_vec, input_q)
        return x


class Fusion(VQG_Abstract):
    '''
    This scheme use the [fusion + generation] pipeline
    '''
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(Fusion, self).__init__(opt, vocab_words, vocab_answers)
        self.concept_num = opt.get('concept_num', 0)
        if self.opt['vqa']['arch'] == 'Mutan':
            dim_h = int(self.opt['vqa']['fusion']['dim_hv']
                          / opt['attention']['nb_glimpses']) * opt['attention']['nb_glimpses']
        else: # MLB
            dim_h = self.opt['vqa']['fusion']['dim_h'] * opt['attention']['nb_glimpses']
        self.linear_va_transform = nn.Linear(
                        dim_h + self.opt['dim_a'], 
                        self.opt['vqg']['vec2seq']['dim_embedding'])
        if self.concept_num > 0:
            self.concept_classifier = nn.Linear(
                        self.opt['vqg']['vec2seq']['dim_embedding'], self.concept_num)

    def _fusion_feat(self, x_v, x_a):
        x_va = torch.cat([x_v, x_a], 1)
        return x_va

    def _generate_qestion(self, x_va, input_q = None):

        # transform the fused feature to word_embedding domain (length = opts['vec2seq']['dim_embedding'])
        x_va = F.dropout(x_va,
                      p=self.opt['vqg']['vec2seq']['dropout'],
                      training=self.training)
        x_va = getattr(F, self.opt['vqg']['vec2seq']['activation'])(self.linear_va_transform(x_va))
        x_va = F.dropout(x_va,
                      p=self.opt['vqg']['vec2seq']['dropout'],
                      training=self.training)
        if self.is_testing:
            if x_va.size(0) == 1:
                x = self.question_generation.beam_search(x_va, beam_size=3,)
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

    def forward(self, x_v_vec, x_a_vec, input_q = None):
        x_va = self._fusion_feat(x_v_vec, x_a_vec)
        x = self._generate_qestion(x_va, input_q)
        return x


class VQA_Dual(nn.Module):

    def __init__(self, vqa_module, seq2vec, opt={}, vocab_words=[], vocab_answers=[]):
        super(VQA_Dual, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        self.fusion_classif = fusion.MutanFusion(vqa_module.opt['vqa']['fusion'],
                                                 visual_embedding=False,
                                                 question_embedding=False)
        # self.fusion_classif = vqa_module.fusion_classif
        self.is_testing = False

        self.linear_va_transform = nn.Linear(
                        vqa_module.linear_classif.in_features, 
                        self.opt['vqg']['vec2seq']['dim_embedding'])

        self.linear_a_fusion = nn.Linear(vqa_module.linear_classif.in_features, vqa_module.linear_q_fusion.out_features)
        # Modules for Question Generation
        self.question_generation = getattr(vec2seq, opt['vqg']['vec2seq']['arch'])(vocab_words, opt['vqg']['vec2seq'])
        self.is_testing = False

        # share parameters for embeddings
        if self.opt['vqg']['vec2seq'].get('share_with_seq2vec', False):
            print('Sharing parameters between [seq2vec] and [vec2seq]')
            self.question_generation.embedder = seq2vec.embedding
            if self.opt['seq2vec']['arch'] == 'lstm':
                self.question_generation.rnn = seq2vec.rnn
            # self.question_generation.classifier.weight = seq2vec.embedding.weight

    def _generate_qestion(self, x_va, input_q = None):

        # transform the fused feature to word_embedding domain (length = opts['vec2seq']['dim_embedding'])
        x_va = F.dropout(x_va,
                      p=self.opt['vqg']['vec2seq']['dropout'],
                      training=self.training)
        x_va = getattr(F, self.opt['vqg']['vec2seq']['activation'])(self.linear_va_transform(x_va))
        x_va = F.dropout(x_va,
                      p=self.opt['vqg']['vec2seq']['dropout'],
                      training=self.training)
        if self.is_testing:
            if x_va.size(0) == 1:
                x = self.question_generation.beam_search(x_va, beam_size=3,)
            else:
                x = self.question_generation.generate(x_va)

            return x
        else:
            x = self.question_generation(x_va, input_q)
            return x


    def forward(self, x_v, x_a, input_q = None):
        x_a = self.linear_a_fusion(x_a)
        if 'activation' in self.opt['vqg']:
            x_a = getattr(F, self.opt['vqg']['activation'])(x_a)
        # Second multimodal fusion
        x = self.fusion_classif(x_v, x_a)
        if 'activation' in self.opt['vqg']:
            x = getattr(F, self.opt['vqg']['activation'])(x)
        x = F.dropout(x,
                      p=self.opt['vqg']['dropout'],
                      training=self.training)
        x = self._generate_qestion(x, input_q)
        return x
