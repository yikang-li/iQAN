import time
import torch
import pdb
from torch.autograd import Variable
import vqa.lib.utils as utils
from vqg.models.vec2seq import process_lengths_sort
from torch.nn.utils.rnn import pack_padded_sequence
from .utils import clip_gradient
import numpy as np


def train(loader, model, criterion, optimizer, logger, epoch, print_freq=10):
    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        target_question = sample['question']
        # To arrange the length of mini-batch by the descending order of question length
        new_ids, lengths = process_lengths_sort(target_question) 
        target_question = Variable(target_question.index_select(0, new_ids).transpose(0, 1).cuda())
        input_visual = Variable(sample['visual'].index_select(0, new_ids).cuda())
        
        # compute output
        output = model(input_visual, target_question, lengths)
        target_question = pack_padded_sequence(target_question, lengths)[0] # add additional image feature
        #  pdb.set_trace()
        loss = criterion(output, target_question)
        meters['loss'].update(loss.data[0], n=batch_size)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model, 5.)
        optimizer.step()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   loss=meters['loss']))

    logger.log_meters('train', n=epoch)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(loader, model, criterion, logger, epoch=0, print_freq=10):
    results = []
    model.eval()
    # switch to evaluate mode
    meters = logger.reset_meters('val')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        target_question = sample['question']
        # To arrange the length of mini-batch by the descending order of question length
        new_ids, lengths = process_lengths_sort(target_question) 
        target_question = Variable(target_question.index_select(0, new_ids).transpose(0, 1).cuda())
        input_visual = Variable(sample['visual'].index_select(0, new_ids).cuda())
        
        # compute output
        output = model(input_visual, target_question[:-1], lengths)
        target_question = pack_padded_sequence(target_question, lengths)[0]
        loss = criterion(output, target_question)
        meters['loss'].update(loss.data[0], n=batch_size)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], loss=meters['loss']))

    print(' * loss {loss.avg:.3f}'
          .format(loss=meters['loss']))

    logger.log_meters('val', n=epoch)
    model.train()
    return meters['loss'].avg


# to generate single image result with beam search
def generate(resized_img, cnn_model, vqg_model, ):
    raise NotImplementedError


def evaluate(loader, model, logger, print_freq=10):

    model.eval()
    meters = logger.reset_meters('test')
    results = []
    model.is_testing = True
    end = time.time()
    img_num = len(loader)
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        input_visual = Variable(sample['visual'].cuda(), volatile=True )

        # compute output
        output = model.generate(input_visual, beam_size=3, include_unknown=False)
        results.append({'gt_question': [loader.dataset.wid_to_word[w_id] for w_id in sample['question'][0]], #sample['question'][j].numpy(),
                        'image': sample['image'][0], 
                        'generated_question': output, 
                        'answer': loader.dataset.aid_to_ans[sample['answer'][0]]})
        print('generated questions:', output)
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()
        if i > 100:
            break

    model.is_testing = False
    return results
