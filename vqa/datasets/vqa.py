import os
import pickle
import torch
import torch.utils.data as data
import copy
import numpy as np
import pdb

from ..lib import utils
from torch.utils.data import DataLoader
from .utils import AbstractVQADataset
from .vqa_interim import vqa_interim
from .vqa2_interim import vqa_interim as vqa2_interim
from .vqa_processed import vqa_processed
from .clevr_interim import clevr_interim
from .clevr_processed import clevr_processed
from . import coco
from . import vgenome
from . import clevr


class AbstractVQA(AbstractVQADataset):

    def __init__(self, data_split, opt, dataset_img=None):
        super(AbstractVQA, self).__init__(data_split, opt, dataset_img)

        if 'test' in self.data_split: # means self.data_split is 'val' or 'test'
            self.opt['samplingans'] = False
        assert 'samplingans' in self.opt, \
               "opt['vqa'] does not have 'samplingans' "\
               "entry. Set it to True or False."
  
        if self.data_split == 'test':
            path_testdevset = os.path.join(self.subdir_processed, 'testdevset.pickle')
            with open(path_testdevset, 'rb') as handle:
                self.testdevset_vqa = pickle.load(handle)
            self.is_qid_testdev = {}
            for i in range(len(self.testdevset_vqa)):
                qid = self.testdevset_vqa[i]['question_id']
                self.is_qid_testdev[qid] = True

    def _raw(self):
        raise NotImplementedError

    def _interim(self, select_questions=False):
        raise NotImplementedError

    def _processed(self):
        raise NotImplementedError

    def __getitem__(self, index):
        item = {}
        # TODO: better handle cascade of dict items
        item_vqa = self.dataset[index]

        # Process Visual (image or features)
        if self.dataset_img is not None:
            item_img = self.dataset_img.get_by_name(item_vqa['image_name'])
            item['visual'] = item_img['visual']
            item['image'] = item_vqa['image_name'] # Yikang added for tracing the image path
        
        # Process Question (word token)
        item['question_id'] = item_vqa['question_id']
        # Add additional <START> Token if set
        question = [self.word_to_wid['START']] + item_vqa['question_wids'] if self.opt['add_start'] else item_vqa['question_wids']
        item['question'] = torch.LongTensor(question) 

        if self.data_split == 'test':
            if item['question_id'] in self.is_qid_testdev:
                item['is_testdev'] = True
            else:
                item['is_testdev'] = False
        else:
        ## Process Answer if exists
            if 'all_ans'in self.opt.keys() and self.opt['all_ans']:
                item['answer_all'] = torch.zeros(len(self.aid_to_ans)) 
                for t_aid in item_vqa['answers_aid']:
                    item['answer_all'][t_aid] = 1 # use several-hots vectors to indicate which answers is sampled
            if self.opt['samplingans']:
                proba = item_vqa['answers_count']
                proba = proba / np.sum(proba)
                item['answer'] = int(np.random.choice(item_vqa['answers_aid'], p=proba))
            else:
                item['answer'] = item_vqa['answer_aid']

            if 'sample_concept' in self.opt.keys() and self.opt['sample_concept']:
                item['concept'] = torch.zeros(len(self.cid_to_concept))
                for t_aid in item_vqa['concepts_cid']:
                    item['concept'][t_aid] = 1 # use several-hots vectors to indicate which answers is sampled

        return item

    def __len__(self):
        if self.data_split == 'train':
            if self.opt.get('partial', -1) > 0:
                return int(len(self.dataset) * min(self.opt['partial'], 1.))
        return len(self.dataset)

    def num_classes(self):
        return len(self.aid_to_ans)

    def vocab_words(self):
        return list(self.wid_to_word.values())

    def vocab_answers(self):
        return self.aid_to_ans

    def data_loader(self, batch_size=10, num_workers=4, shuffle=False):
        return DataLoader(self,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False)
 
    def split_name(self, testdev=False):
        if testdev:
            return 'test-dev2015'
        if self.data_split in ['train', 'val']:
            return self.data_split+'2014'
        elif self.data_split == 'test':
            return self.data_split+'2015'
        elif self.data_split == 'testdev':
            return 'test-dev2015'
        else:
            assert False, 'Wrong data_split: {}'.format(self.data_split)
 
    #  def subdir_processed(self):
    #      subdir = 'nans,' + str(self.opt['nans']) \
    #            + '_maxlength,' + str(self.opt['maxlength']) \
    #            + '_minwcount,' + str(self.opt['minwcount']) \
    #            + '_nlp,' + self.opt['nlp'] \
    #            + '_pad,' + self.opt['pad'] \
    #            + '_trainsplit,' + self.opt['trainsplit']
    #      subdir = os.path.join(self.dir_processed, subdir)
    #      return subdir


class VQA(AbstractVQA):

    def __init__(self, data_split, opt, dataset_img=None):
        super(VQA, self).__init__(data_split, opt, dataset_img)

    def _raw(self):
        dir_zip = os.path.join(self.dir_raw, 'zip')
        dir_ann = os.path.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_zip)
        os.system('mkdir -p '+dir_ann)
        os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip -P '+dir_zip)
        os.system('unzip '+os.path.join(dir_zip, 'Questions_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'Questions_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'Questions_Test_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'Annotations_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'Annotations_Val_mscoco.zip')+' -d '+dir_ann)

    def _interim(self, select_questions=False):
        vqa_interim(self.opt['dir'], select_questions=select_questions)

    def _processed(self):
        vqa_processed(self.opt)


class VQA2(AbstractVQA):

    def __init__(self, data_split, opt, dataset_img=None):
        super(VQA2, self).__init__(data_split, opt, dataset_img)

    def _raw(self):
        dir_zip = os.path.join(self.dir_raw, 'zip')
        dir_ann = os.path.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_zip)
        os.system('mkdir -p '+dir_ann)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+dir_zip)
        os.system('unzip '+os.path.join(dir_zip, 'v2_Questions_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'v2_Questions_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'v2_Questions_Test_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'v2_Annotations_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+os.path.join(dir_zip, 'v2_Annotations_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('mv '+os.path.join(dir_ann, 'v2_mscoco_train2014_annotations.json')+' '
                       +os.path.join(dir_ann, 'mscoco_train2014_annotations.json'))
        os.system('mv '+os.path.join(dir_ann, 'v2_mscoco_val2014_annotations.json')+' '
                       +os.path.join(dir_ann, 'mscoco_val2014_annotations.json'))
        os.system('mv '+os.path.join(dir_ann, 'v2_OpenEnded_mscoco_train2014_questions.json')+' '
                       +os.path.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json'))
        os.system('mv '+os.path.join(dir_ann, 'v2_OpenEnded_mscoco_val2014_questions.json')+' '
                       +os.path.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json'))
        os.system('mv '+os.path.join(dir_ann, 'v2_OpenEnded_mscoco_test2015_questions.json')+' '
                       +os.path.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json'))
        os.system('mv '+os.path.join(dir_ann, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')+' '
                       +os.path.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json'))

    def _interim(self, select_questions=False):
        vqa2_interim(self.opt['dir'], select_questions=select_questions)

    def _processed(self):
        vqa_processed(self.opt)


class VQAVisualGenome(data.Dataset):

    def __init__(self, dataset_vqa, dataset_vgenome):
        self.dataset_vqa = dataset_vqa
        self.dataset_vgenome = dataset_vgenome
        self._filter_dataset_vgenome()

    def _filter_dataset_vgenome(self):
        print('-> Filtering dataset vgenome')
        data_vg = self.dataset_vgenome.dataset
        ans_to_aid = self.dataset_vqa.ans_to_aid
        word_to_wid = self.dataset_vqa.word_to_wid
        data_vg_new = []
        not_in = 0
        for i in range(len(data_vg)):
            if data_vg[i]['answer'] not in ans_to_aid:
                not_in += 1
            else:
                data_vg[i]['answer_aid'] = ans_to_aid[data_vg[i]['answer']]
                for j in range(data_vg[i]['seq_length']):
                    word = data_vg[i]['question_words_UNK'][j]
                    if word in word_to_wid:
                        wid = word_to_wid[word]
                    else:
                        wid = word_to_wid['UNK']
                    data_vg[i]['question_wids'][j] = wid
                data_vg_new.append(data_vg[i])
        print('-> {} / {} items removed'.format(not_in, len(data_vg)))
        self.dataset_vgenome.dataset = data_vg_new
        print('-> {} items left in visual genome'.format(len(self.dataset_vgenome)))
        print('-> {} items total in vqa+vg'.format(len(self)))
                

    def __getitem__(self, index):
        if index < len(self.dataset_vqa):
            item = self.dataset_vqa[index]
            #print('vqa')
        else:
            item = self.dataset_vgenome[index - len(self.dataset_vqa)]
            #print('vg')
        #import ipdb; ipdb.set_trace()
        return item

    def __len__(self):
        return len(self.dataset_vqa) + len(self.dataset_vgenome)

    def num_classes(self):
        return self.dataset_vqa.num_classes()

    def vocab_words(self):
        return self.dataset_vqa.vocab_words()

    def vocab_answers(self):
        return self.dataset_vqa.vocab_answers()

    def data_loader(self, batch_size=10, num_workers=4, shuffle=False):
        return DataLoader(self,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=False)

    def split_name(self, testdev=False):
        return self.dataset_vqa.split_name(testdev=testdev)


class CLEVR(AbstractVQADataset):

    def __init__(self, data_split, opt, dataset_img=None):
        super(CLEVR, self).__init__(data_split, opt, dataset_img)

        if 'test' in self.data_split: # means self.data_split is 'val' or 'test'
            raise NotImplementedError

    def _raw(self):
        raise NotImplementedError

    def _interim(self, select_questions=False):
        clevr_interim(self.opt['dir'], select_questions=select_questions)

    def _processed(self):
        clevr_processed(self.opt)

    def __getitem__(self, index):
        item = {}
        # TODO: better handle cascade of dict items
        item_vqa = self.dataset[index]

        # Process Visual (image or features)
        if self.dataset_img is not None:
            item_img = self.dataset_img.get_by_name(item_vqa['image_name'])
            item['visual'] = item_img['visual']
            item['image'] = item_vqa['image_name'] # Yikang added for tracing the image path
        
        # Process Question (word token)
        item['question_id'] = item_vqa['question_id']
        # Add additional <START> Token if set
        question = [self.word_to_wid['START']] + item_vqa['question_wids'] if self.opt['add_start'] else item_vqa['question_wids']
        item['question'] = torch.LongTensor(question) 

        if self.data_split == 'test':
            raise NotImplementedError
        else:
        ## Process Answer if exists
            item['answer'] = item_vqa['answer_aid']

            # if 'sample_concept' in self.opt.keys() and self.opt['sample_concept']:
            #     item['concept'] = torch.zeros(len(self.cid_to_concept))
            #     for t_aid in item_vqa['concepts_cid']:
            #         item['concept'][t_aid] = 1 # use several-hots vectors to indicate which answers is sampled

        return item

    def __len__(self):
        if self.data_split == 'train':
            if self.opt.get('partial', -1) > 0:
                return int(len(self.dataset)) * self.opt['partial']
        return len(self.dataset)

    def num_classes(self):
        return len(self.aid_to_ans)

    def vocab_words(self):
        return list(self.wid_to_word.values())

    def vocab_answers(self):
        return self.aid_to_ans

    def data_loader(self, batch_size=10, num_workers=4, shuffle=False):
        return DataLoader(self,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False)
 
    def split_name(self, testdev=False):
        if testdev:
            return 'test-dev'
        if self.data_split in ['train', 'val']:
            return self.data_split
        elif self.data_split == 'test':
            return self.data_split
        elif self.data_split == 'testdev':
            return 'test-dev'
        else:
            assert False, 'Wrong data_split: {}'.format(self.data_split)


def factory(data_split, opt, opt_coco=None, opt_vgenome=None, opt_clevr=None):
    dataset_img = None
    if opt_clevr is not None:
        dataset_img = clevr.factory(data_split, opt_clevr)
        dataset_clevr = CLEVR(data_split, opt, dataset_img)
        return dataset_clevr

    if opt_coco is not None:
        dataset_img = coco.factory(data_split, opt_coco)

    if opt['dataset'] == 'VQA' and '2' not in opt['dir']: # sanity check
        dataset_vqa = VQA(data_split, opt, dataset_img)
    elif opt['dataset'] == 'VQA2' and '2' in opt['dir']: # sanity check
        dataset_vqa = VQA2(data_split, opt, dataset_img)
    else:
        raise ValueError

    if opt_vgenome is not None:
        dataset_vgenome = vgenome.factory(opt_vgenome, vqa=True)
        return VQAVisualGenome(dataset_vqa, dataset_vgenome)
    else:
        return dataset_vqa

