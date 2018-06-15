import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from models.convnets import ResNet

import copy
import pdb

from vqa.lib import utils
# modules for VQA
from vqa.models import seq2vec
from vqa.models import fusion
from models import vec2seq
from models.beam_search import CaptionGenerator
import models.vqa_modules as vqa_modules
# modules for VQG
import models.vqg_modules as vqg_modules
# Attention Module
import models.attention_modules as attention_modules

from models.vec2seq import process_lengths_sort, process_lengths


class Dual_Model_Baseline(nn.Module):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(Dual_Model_Baseline, self).__init__()
        self.opt = opt
        # To fuse different glimpses
        self.vocab_answers = vocab_answers
        self.vocab_words = vocab_words
        self.num_classes = len(self.vocab_answers)

        # VQA Modules
        self.seq2vec = nn.Embedding(len(self.vocab_words), 1024)
        # Modules for classification
        self.linear_classif = nn.Linear(1024,
                                        self.num_classes)
        self.fusion_classif = lambda x,y: torch.cat([x, y], 1)

        # share W and E
        self.answer_embeddings =  nn.Embedding(self.num_classes, 1024) 
        # VQG modules
        self.linear_va_transform = nn.Linear(3072, 1024)
        self.linear_a_transform = nn.Linear(3072, 1024)
        # Modules for Question Generation
        self.question_generation = getattr(vec2seq, opt['vqg']['vec2seq']['arch'])(vocab_words, opt['vqg']['vec2seq'])

        self.is_testing = False
        self.sample_num = 5

    def set_share_parameters(self):
        self.answer_embeddings.weight = self.linear_classif.weight
        self.seq2vec.weight = self.question_generation.embedder.weight
        print('[answer] and [word] embeddings are shared')
        

        
    def forward(self, input_v, input_q=None, target_a=None):
        if self.is_testing: # Start testing Mode
            return self._testing(input_v, input_q, target_a)
        else:
            return self._train(input_v, input_q, target_a)

    def _BOW(self, input_q):
        lengths = process_lengths(input_q)
        q_embeddings = self.seq2vec(input_q)
        output_q_embeddings = []
        for i, length in enumerate(lengths):
            output_q_embeddings.append(q_embeddings[i][:length].sum(0))
        output_q_embeddings = torch.stack(output_q_embeddings, 0)
        return output_q_embeddings




    def _vqa_forward(self, input_v, input_q):
        x_q_embedding = self._BOW(input_q)
        x_q = F.dropout(x_q_embedding,
                        p=self.opt['vqa']['dropout'],
                        training=self.training)
        # Second multimodal fusion
        x_vq = self.fusion_classif(input_v, x_q)
        x_a_pred = getattr(F, self.opt['vqa']['activation'])(self.linear_a_transform(x_vq))
        x_a = F.dropout(x_a_pred,
                      p=self.opt['vqa']['dropout'],
                      training=self.training)
        answers = self.linear_classif(x_a)
        return answers, x_q_embedding, x_a_pred

    def _vqg_forward(self, input_v, target_a, input_q=None):
        x_a_embedding = self.answer_embeddings(target_a.view(1, -1)).squeeze(0)
        # Second multimodal fusion
        x_va = self.fusion_classif(input_v, x_a_embedding)
        x_va = F.dropout(x_va,
                      p=self.opt['vqg']['dropout'],
                      training=self.training)
        x_q_pred = getattr(F, self.opt['vqg']['activation'])(self.linear_va_transform(x_va))
        q_embedding = F.dropout(x_q_pred,
                      p=self.opt['vqg']['dropout'],
                      training=self.training)
        questions = self._generate_qestion(q_embedding, input_q, )
        return questions, x_q_pred, x_a_embedding


    def _train(self, input_v, input_q, target_a):
        ##### VQA forward #######
        answers, x_q, x_a_pred = self._vqa_forward(input_v, input_q)

        ##### VQG forward #######
        questions, x_q_pred, x_a = self._vqg_forward(input_v, target_a, input_q)
            
        return answers, questions, F.smooth_l1_loss(x_a_pred, x_a) + F.smooth_l1_loss(x_q_pred, x_q)

    def _testing(self, input_v, input_q, input_a=None):

        ##### VQA forward #######
        answers = self._vqa_forward(input_v, input_q)[0]
        g_answers_score, g_answers = torch.max(F.softmax(answers), dim=1)
        if input_a is not None:
            generated_q = self._vqg_forward(input_v, input_a)[0]
            g_answers = input_a
        else:
            generated_q = self._vqg_forward(input_v, g_answers)[0]
            # generated_q = []
            # for j in range(self.sample_num):
            #     generated_q.append(self._vqg_forward(input_v, g_answers[:, j].contiguous())[0])
        return answers, g_answers, g_answers_score, generated_q
    
    def set_testing(self, is_testing=True, sample_num=5):
        self.is_testing = is_testing
        self.sample_num = sample_num

    def _generate_qestion(self, x_va, input_q = None):

        # transform the fused feature to word_embedding domain (length = opts['vec2seq']['dim_embedding'])
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
