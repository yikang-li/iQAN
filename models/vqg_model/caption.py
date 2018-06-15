import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from .beam_search import CaptionGenerator
from torchvision import transforms

normalize_values = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}

class CaptionModel(nn.Module):

    def __init__(self, vocab, embedding_size=256, rnn_size=256, num_layers=2,
                 share_embedding_weights=False, use_parallel = False):
        super(CaptionModel, self).__init__()
        self.vocab = vocab
        self.fc = nn.Linear(2048, embedding_size)
        self.use_parallel = use_parallel
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers=num_layers)
        self.classifier = nn.Linear(rnn_size, len(vocab))
        self.embedder = nn.Embedding(len(self.vocab), embedding_size)
        if share_embedding_weights:
            self.embedder.weight = self.classifier.weight

    def forward(self, imgs, captions, lengths):
        embeddings = self.embedder(captions)
        img_feats = self.fc(imgs).unsqueeze(0)
        embeddings = torch.cat([img_feats, embeddings], 0)
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        feats, state = self.rnn(packed_embeddings)
        pred = self.classifier(feats[0])
        return pred

    def generate(self, img, scale_size=256, crop_size=224,
                 eos_token='EOS', unk_token = 'UNK', beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0, include_unknown = False):

        cap_gen = CaptionGenerator(embedder=self.embedder,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.vocab.index(eos_token),
                                   include_unknown = include_unknown,
                                   unk_id = self.vocab.index(unk_token),
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        img_feats = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img_feats)
        sentences = [' '.join([self.vocab[idx] for idx in sent])
                     for sent in sentences]
        return sentences

    def save_checkpoint(self, filename):
        torch.save({'embedder_dict': self.embedder.state_dict(),
                    'rnn_dict': self.rnn.state_dict(),
                    'cnn_dict': self.cnn.state_dict(),
                    'classifier_dict': self.classifier.state_dict(),
                    'vocab': self.vocab,
                    'model': self},
                   filename)

    def load_checkpoint(self, filename):
        cpnt = torch.load(filename)
        if 'cnn_dict' in cpnt:
            self.cnn.load_state_dict(cpnt['cnn_dict'])
        self.embedder.load_state_dict(cpnt['embedder_dict'])
        self.rnn.load_state_dict(cpnt['rnn_dict'])

        self.classifier.load_state_dict(cpnt['classifier_dict'])

    def finetune_cnn(self, allow=True):

        if self.use_parallel:
            for p in self.cnn.module.parameters():
                p.requires_grad = allow
            for p in self.cnn.module.fc.parameters():
                p.requires_grad = True
        else:
            for p in self.cnn.parameters():
                p.requires_grad = allow
            for p in self.cnn.fc.parameters():
                p.requires_grad = True
