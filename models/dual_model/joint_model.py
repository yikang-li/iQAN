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

class QAG(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(QAG, self).__init__()
        self.model_vqa = getattr(models_vqa, opt['vqa']['arch'])(
            opt['vqa'], vocab_words, vocab_answers)
        self.model_vqg = getattr(models_vqg, opt['vqg']['arch'])(
            opt['vqg'], vocab_words, vocab_answers)
        self.mode_vqa = opt['vqa']['mode']
        self.mode_vqg = opt['vqg']['mode']
        self.use_reinforce = True
    

    # def forward(self, input_v, input_q):
    #     answers = self.model_vqa(prepare_input(input_v, self.mode_vqa), input_q)
    #     selected_answer = answers.multinomial()
    #     questions = self.model_vqg(prepare_input(input_v, self.mode_vqg), selected_answer.squeeze(1), input_q)
    #     return answers, (selected_answer), questions

    def forward(self, input_v, input_q, target_a):
        answers = self.model_vqa(prepare_input(input_v, self.mode_vqa), input_q)
        selected_answer = F.softmax(answers).multinomial()
        questions = self.model_vqg(prepare_input(input_v, self.mode_vqg), selected_answer.squeeze(1), input_q[:, :-1])
        # Hack for the compatability of reinforce() and DataParallel()
        lengths = process_lengths(input_q) 
        loss_q = [F.cross_entropy(questions[i, :lengths[i]], input_q[i, 1:(lengths[i]+1)]) for i in range(questions.size(0))]
        loss_q = torch.cat(loss_q, 0)
        if self.training:
            if self.use_reinforce:
                reward = -0.1 * (loss_q - loss_q.mean()) #/ (loss_q.std() + float(np.finfo(np.float32).eps))
                selected_answer.reinforce(reward.data.view(selected_answer.size()))
            else:
                selected_answer.reinforce(0.)

        return answers, selected_answer, questions, loss_q


class QAG_Conv_Abstract(nn.Module):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(QAG_Conv_Abstract, self).__init__()
        self.shared_conv_layer = None
        self.model_vqa = getattr(models_vqa, opt['vqa']['arch'])(
            opt['vqa'], vocab_words, vocab_answers)
        self.model_vqg = getattr(models_vqg, opt['vqg']['arch'])(
            opt['vqg'], vocab_words, vocab_answers)
        self.mode_vqa = opt['vqa']['mode']
        self.mode_vqg = opt['vqg']['mode']
        self.use_reinforce = True

    def forward(self, input_v, input_q, target_a):
        input_v = self.shared_conv_layer(input_v)
        answers = self.model_vqa(prepare_input(input_v, self.mode_vqa), input_q)
        selected_answer = F.softmax(answers).multinomial()
        questions = self.model_vqg(prepare_input(input_v, self.mode_vqg), selected_answer.squeeze(1), input_q[:, :-1])
        # Hack for the compatability of reinforce() and DataParallel()
        lengths = process_lengths(input_q) 
        loss_q = [F.cross_entropy(questions[i, :lengths[i]], input_q[i, 1:(lengths[i]+1)]) for i in range(questions.size(0))]
        loss_q = torch.cat(loss_q, 0)
        if self.training:
            if self.use_reinforce:
                reward = -0.1 * (loss_q - loss_q.mean()) #/ (loss_q.std() + float(np.finfo(np.float32).eps))
                selected_answer.reinforce(reward.data.view(selected_answer.size()))
            else:
                selected_answer.reinforce(0.)

        return answers, selected_answer, questions, loss_q


class QAG_ResNet(QAG_Conv_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(QAG_ResNet, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = ResNet(opt['arch_resnet'], opt['pooling'])
    

class QAG_Conv(QAG_Conv_Abstract):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(QAG_Conv, self).__init__(opt, vocab_words, vocab_answers)
        self.shared_conv_layer = ResNet(opt['arch_resnet']).layer4



def prepare_input(input_visual, mode):
    if mode == 'noatt' and input_visual.dim() == 4:
        return F.avg_pool2d(input_visual, input_visual.size(2)).squeeze()
    else:
        return input_visual

