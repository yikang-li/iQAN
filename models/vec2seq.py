import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .beam_search import CaptionGenerator

__DEBUG__ = False

def process_lengths(input):
    # the input sequence should be in [batch x word]
    max_length = input.size(1) - 1 # remove START
    lengths = list(max_length - input.data.eq(0).sum(1).squeeze() if isinstance(input, Variable) else input.eq(0).sum(1).squeeze())
    lengths = [min(max_length, length + 1) for length in lengths] # add additional word for EOS
    return lengths

def process_lengths_sort(input, include_inv = False, cuda=True):
    # the input sequence should be in [batch x word]
    max_length = input.size(1) - 1 # remove additional START
    lengths = list(max_length - input.eq(0).sum(1).squeeze())
    lengths = [(i, lengths[i]) for i in range(len(lengths))]
    lengths.sort(key=lambda p:p[1], reverse=True)
    feat_id = [lengths[i][0] for i in range(len(lengths))]
    lengths = [min(max_length, lengths[i][1] + 1) for i in range(len(lengths))] # add additional word for EOS
    if include_inv:
        inv_id = torch.LongTensor(len(lengths))
        for i, i_id in enumerate(feat_id):
            inv_id[i_id] = i
        if cuda:
            return torch.LongTensor(feat_id).cuda(), lengths, torch.LongTensor(inv_id).cuda()
        else:
            return torch.LongTensor(feat_id), lengths, torch.LongTensor(inv_id)
    else:
        if cuda:
            return torch.LongTensor(feat_id).cuda(), lengths
        else:
            return torch.LongTensor(feat_id), lengths
    
class Abstract_Gen_Model(nn.Module):
    def __init__(self, vocab, opt):
        super(Abstract_Gen_Model, self).__init__()
        self.vocab = vocab
        self.start = vocab.index('START') if 'START' in vocab else None
        self.end = vocab.index('EOS')
        self.unk = vocab.index('UNK')
        self.classifier = nn.Linear(opt['dim_h'], len(self.vocab),  bias=False)
        self.embedder = nn.Embedding(len(self.vocab), opt['dim_embedding'])
        self.opt = opt
        if opt['share_weight']:
            assert opt['dim_embedding'] == opt['dim_h'], 'If share_weight is set, dim_embedding == dim_h required!'
            self.embedder.weight = self.classifier.weight # make sure the embeddings are from the final 
        # initilization
        torch.nn.init.uniform(self.embedder.weight, -0.25, 0.25)


class SAMI(Abstract_Gen_Model): # Single Answer and Multiple Image
    def __init__(self, vocab, opt):
        super(SAMI, self).__init__(vocab, opt)
        self.rnn = nn.LSTM(opt['dim_embedding'] + opt['dim_v'], opt['dim_h'], num_layers=opt['nb_layers'], batch_first=True)


    def forward(self, v_feat, a_feat, questions):
        '''
        The answer embdding is fed at the first step, and the image
        embedding is fed into the model concatenated with word embedding 
        at every step.
        '''
        # prepare the data
        batch_size = questions.size(0)
        max_length = questions.size(1)
        new_ids, lengths, inv_ids = process_lengths_sort(questions.cpu().data, include_inv=True)
        new_ids = Variable(new_ids).detach()
        inv_ids = Variable(inv_ids).detach()


        padding_size = questions.size(1) - lengths[0]
        questions = questions.index_select(0, new_ids)
        v_feat = v_feat.index_select(0, new_ids)
        a_feat = a_feat.index_select(0, new_ids)
        embeddings = self.embedder(questions)
        v_feat = v_feat.unsqueeze(1).expand(batch_size, max_length, self.opt['dim_v'])
        embeddings = torch.cat([embeddings, v_feat], 2) # each time step, input image and word embedding
        a_feat = a_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(a_feat)
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True) # add additional image feature
        feats, _ = self.rnn(packed_embeddings, hidden_feat)
        if __DEBUG__:
            print("[Generation Module] RNN feature.std(): "),
            print(feats.std())
        pred = self.classifier(feats[0])
        pred = pad_packed_sequence([pred, feats[1]], batch_first=True)
        pred = pred[0].index_select(0, inv_ids)
        if padding_size > 0:
            pred = torch.cat([pred, Variable(torch.zeros(batch_size, padding_size, pred.size(2)).type_as(pred.data)).detach()], 1)
        return pred

    def generate(self, v_feat, a_feat):
        batch_size = v_feat.size(0)
        max_length = self.opt['nseq'] if 'nseq' in self.opt else 20
        #x = Variable(torch.ones(1, batch_size,).type(torch.LongTensor) * self.start, volatile=True).cuda() # <start>
        output = Variable(torch.zeros(max_length, batch_size).type(torch.LongTensor)).cuda()
        scores = torch.zeros(batch_size)
        flag = torch.ones(batch_size)
        input_x = a_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(input_x) # initialize the LSTM
        x = Variable(torch.ones(batch_size, 1, ).type(torch.LongTensor) * self.start, requires_grad=False).cuda() # <start>
        v_feat = v_feat.unsqueeze(1)
        input_x = torch.cat([self.embedder(x), v_feat], 2)
        for i in range(max_length):
            output_feature, hidden_feat = self.rnn(input_x, hidden_feat)
            output_t = self.classifier(output_feature.view(batch_size, output_feature.size(2)))
            output_t = F.log_softmax(output_t)
            logprob, x = output_t.max(1)
            output[i] = x
            scores += logprob.cpu().data * flag
            flag[x.cpu().eq(self.end).data] = 0
            if flag.sum() == 0:
                break
            input_x = torch.cat([self.embedder(x.view(-1, 1)), v_feat], 2)
        return output.transpose(0, 1)

class Baseline(Abstract_Gen_Model):
    def __init__(self, vocab, opt):
        super(Baseline, self).__init__(vocab, opt)
        self.rnn = nn.LSTM(opt['dim_embedding'], opt['dim_h'], num_layers=opt['nb_layers'], batch_first=True)


    def forward(self, va_feat, questions):
        # image feature tranform
        batch_size=va_feat.size(0)
        new_ids, lengths, inv_ids = process_lengths_sort(questions.cpu().data, include_inv=True)
        new_ids = Variable(new_ids).detach()
        inv_ids = Variable(inv_ids).detach()
        # manually set the first length to MAX_LENGTH
        padding_size = questions.size(1) - lengths[0]
        questions = questions.index_select(0, new_ids)
        if __DEBUG__:
            print("[Generation Module] input feat.std(): "),
            print(va_feat.std())
        embeddings = self.embedder(questions)
        va_feat = va_feat.index_select(0, new_ids).unsqueeze(1)
        _, hidden_feat = self.rnn(va_feat)
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True) # add additional image feature
        feats, _ = self.rnn(packed_embeddings, hidden_feat)
        if __DEBUG__:
            print("[Generation Module] RNN feature.std(): "),
            print(feats.std())
        pred = self.classifier(feats[0])
        pred = pad_packed_sequence([pred, feats[1]], batch_first=True)
        pred = pred[0].index_select(0, inv_ids)
        if padding_size > 0: # to make sure the sizes of different patches are matchable
            pred = torch.cat([pred, Variable(torch.zeros(batch_size, padding_size, pred.size(2)).type_as(pred.data)).detach()], 1)
        return pred



    def generate(self, va_feat):
        batch_size = va_feat.size(0)
        max_length = self.opt['nseq'] if 'nseq' in self.opt else 20
        #x = Variable(torch.ones(1, batch_size,).type(torch.LongTensor) * self.start, volatile=True).cuda() # <start>
        output = Variable(torch.zeros(max_length, batch_size).type(torch.LongTensor)).cuda()
        scores = torch.zeros(batch_size)
        flag = torch.ones(batch_size)
        input_x = va_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(input_x) # initialize the LSTM
        x = Variable(torch.ones(batch_size, 1, ).type(torch.LongTensor) * self.start, requires_grad=False).cuda() # <start>
        input_x = self.embedder(x)
        for i in range(max_length):
            output_feature, hidden_feat = self.rnn(input_x, hidden_feat)
            output_t = self.classifier(output_feature.view(batch_size, output_feature.size(2)))
            output_t = F.log_softmax(output_t)
            logprob, x = output_t.max(1)
            output[i] = x
            scores += logprob.cpu().data * flag
            flag[x.cpu().eq(self.end).data] = 0
            if flag.sum() == 0:
                break
            input_x = self.embedder(x.view(-1, 1))
        return output.transpose(0, 1)

    def beam_search(self, va_feat, beam_size=3, max_caption_length = 20, length_normalization_factor = 0.0, include_unknown = False):
        batch_size = va_feat.size(0)
        assert batch_size == 1, 'Currently, the beam search only support batch_size == 1'
        input_x = va_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(input_x) # initialize the LSTM
        x = Variable(torch.ones(batch_size, 1, ).type(torch.LongTensor) * self.start, volatile=True).cuda() # <start>
        input_x = self.embedder(x)
        cap_gen = CaptionGenerator(embedder=self.embedder,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.end,
                                   include_unknown = include_unknown,
                                   unk_id = self.unk,
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor, 
                                   batch_first=True)
        sentences, score = cap_gen.beam_search(input_x, hidden_feat)
        sentences = [' '.join([self.vocab[idx] for idx in sent])
                     for sent in sentences]
        return sentences
