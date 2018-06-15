import time
import torch
import pdb
from torch.autograd import Variable
import vqa.lib.utils as utils
from vqg.models.vec2seq import process_lengths_sort
from torch.nn.utils.rnn import pack_padded_sequence
from .utils import clip_gradient
from nltk.translate.bleu_score import modified_precision as bleu_score
from .engine import evaluate, generate, translate_tokens


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
        new_ids = Variable(new_ids).detach()
        target_question = Variable(target_question.cuda(async=True))
        input_visual = Variable(sample['visual'].cuda())
        input_answer = Variable(sample['answer'].cuda())
        target_concept = Variable(sample['concept'].cuda(async=True))
        
        # compute output
        output_question, output_concept = model(input_visual, input_answer, target_question)
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
        output_question = pack_padded_sequence(output_question.index_select(0, new_ids), lengths, batch_first=True)[0] 
        loss = criterion([output_question, output_concept], [target_question, target_concept])
        meters['loss'].update(loss.data[0], n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
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

    # switch to evaluate mode
    meters = logger.reset_meters('val')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        target_question = sample['question']
        # To arrange the length of mini-batch by the descending order of question length
        new_ids, lengths = process_lengths_sort(target_question) 
        new_ids = Variable(new_ids).detach()
        target_question = Variable(target_question.cuda(async=True))
        input_visual = Variable(sample['visual'].cuda())
        input_answer = Variable(sample['answer'].cuda())
        target_concept = Variable(sample['concept'].cuda(async=True))
        
        # compute output
        output_question, output_concept = model(input_visual, input_answer, target_question)
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
        output_question = pack_padded_sequence(output_question.index_select(0, new_ids), lengths, batch_first=True)[0] 
        loss = criterion([output_question, output_concept], [target_question, target_concept])
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
    return meters['loss'].avg
