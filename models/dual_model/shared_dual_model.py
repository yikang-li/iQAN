import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.convnets import ResNet

import copy
import pdb

from vqa.lib import utils
# modules for VQA
from vqa.models import seq2vec
import models.vqa_modules as vqa_modules
# modules for VQG
import models.vqg_modules as vqg_modules
# Attention Module
import models.attention_modules as attention_modules

from models.vec2seq import process_lengths_sort, process_lengths

class Dual_Learning_Model_Abstract(nn.Module):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(Dual_Learning_Model_Abstract, self).__init__()
        self.opt = opt
        self.attention = getattr(attention_modules, opt['attention']['arch'])(opt)
        self.vocab_answers = vocab_answers
        self.vocab_words = vocab_words
        self.num_classes = len(self.vocab_answers)

        # VQA Modules
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        self.vqa_module = getattr(vqa_modules, self.opt['vqa']['arch'])(self.opt, self.vocab_answers)
        if self.opt['vqg']['arch'] == 'VQA_Dual':
            self.vqg_module = vqg_modules.VQA_Dual(self.vqa_module, self.seq2vec, self.opt, self.vocab_words, self. vocab_answers)
        else:
            self.vqg_module = getattr(vqg_modules, self.opt['vqg']['arch'])(self.opt, self.vocab_words, self. vocab_answers)

        self.answer_embeddings =  nn.Embedding(self.vqa_module.linear_classif.out_features, self.vqa_module.linear_classif.in_features) 
        self.answer_embeddings.weight = self.vqa_module.linear_classif.weight

        self.shared_conv_layer = None
        self.is_testing = False
        self.sample_num = 5
        self.use_same_attention = opt['attention'].get('use_same_attention', True)
        

        # To fuse different glimpses
        if self.opt['vqa']['arch'] == 'Mutan':
            dim_h = int(self.opt['vqa']['fusion']['dim_hv'] / opt['attention']['nb_glimpses'])
        else: # MLB
            dim_h = self.opt['vqa']['fusion']['dim_h']

        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'], dim_h)
            for i in range(self.opt['attention']['nb_glimpses'])])

    def _fusion_glimpses(self, list_v_att,):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=self.opt['vqa']['fusion']['dropout_v'],
                            training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            if 'activation_v' in self.opt['vqa']['fusion']:
                x_v = getattr(F, self.opt['vqa']['fusion']['activation_v'])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        return x_v


    def forward(self, input_v, input_q, target_a=None):
        input_v = self.shared_conv_layer(input_v)
        x_q_vec = self.seq2vec(input_q)
        att_v_list = self.attention(input_v, x_q_vec)
        att_v = self._fusion_glimpses(att_v_list)
        # VQA
        answers = self.vqa_module(att_v, x_q_vec)

        if self.is_testing: # Start testing Mode
            g_answers_score, g_answers = torch.topk(F.softmax(answers), self.sample_num)
            generated_q = []
            # answered_a = []
            for j in range(self.sample_num):
                x_a_vec = self.answer_embeddings(g_answers[:, j].contiguous().view(1, -1)).squeeze(0)
                generated_q.append(self.vqg_module(att_v, x_a_vec))
            return answers, g_answers, g_answers_score, generated_q
        else:
            # Sample Answers
            selected_answer = F.softmax(answers).multinomial().squeeze(dim=-1)
            # VQG
            x_a_vec = self.answer_embeddings(target_a.view(1, -1)).squeeze(0)
            questions = self.vqg_module(att_v, x_a_vec, input_q[:, :-1])
            # Hack for the compatability of reinforce() and DataParallel()
            lengths = process_lengths(input_q) 
            loss_q = [F.cross_entropy(questions[i, :lengths[i]], input_q[i, 1:(lengths[i]+1)]) for i in range(questions.size(0))]
            loss_q = torch.cat(loss_q, 0)
            if self.training and self.use_reinforce:
                reward = loss_q.mean() - loss_q #/ (loss_q.std() + float(np.finfo(np.float32).eps))
                selected_answer.reinforce(reward.data.view(selected_answer.size()))
                
            return answers, selected_answer, questions, loss_q
    
    def set_testing(self, is_testing=True, sample_num=5):
        self.is_testing = is_testing
        self.vqg_module.is_testing = is_testing
        self.sample_num = sample_num


class DL_ResNet(Dual_Learning_Model_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(DL_ResNet, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = ResNet(opt['arch_resnet'], opt['pooling'])
    

class DL_Conv(Dual_Learning_Model_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(DL_Conv, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = ResNet(opt['arch_resnet']).layer4

class DL_FC(Dual_Learning_Model_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(DL_FC, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = lambda x: x

