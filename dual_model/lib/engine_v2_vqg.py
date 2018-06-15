import time
import torch
import pdb
import numpy as np
from torch.autograd import Variable
import vqa.lib.utils as utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from models.vec2seq import process_lengths_sort, process_lengths
from nltk.translate.bleu_score import modified_precision as bleu_score
from models.criterions import *
from models.utils import translate_tokens, calculate_bleu_score
from .engine_v2 import evaluate

def train(loader, model, optimizer, logger, epoch, print_freq=10, dual_training=False, alternative_train = -1.):
    # switch to train mode
    model.train()
    model.module.set_testing(False)

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
        target_question = Variable(target_question.cuda())
        input_visual = Variable(sample['visual'].cuda())
        target_answer = Variable(sample['answer'].cuda(async=True))
        
        # compute output
        generated_q =  model(input_visual, target_question, target_answer)
        torch.cuda.synchronize()
        
        # Hack for the compatability of reinforce() and DataParallel()
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
        output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True)[0] 
        loss_q = F.cross_entropy(output, target_question)
        loss = loss_q
        # measure accuracy 
        meters['loss_q'].update(loss_q.data[0], n=batch_size)
        # meters['bleu_score'].update(bleu_score, n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('[Train]\tEpoch: [{0}][{1}/{2}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Q Loss: {loss_q.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   loss_q=meters['loss_q']))

    print('[Train]\tEpoch: [{0}]'
                  'Time {batch_time.avg:.3f}\t'
                  'Q Loss: {loss_q.avg:.3f},\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'],loss_q=meters['loss_q']))

    logger.log_meters('train', n=epoch)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(loader, model, logger, epoch=0, print_freq=10, dual_training=False):
    # switch to train mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        target_question = sample['question']
        # To arrange the length of mini-batch by the descending order of question length
        new_ids, lengths = process_lengths_sort(target_question) 
        target_question = Variable(target_question.cuda(async=True), volatile=True)
        input_visual = Variable(sample['visual'].cuda(async=True), volatile=True)
        target_answer = Variable(sample['answer'].cuda(async=True), volatile=True)
        
                # compute output
        generated_q =  model(input_visual, target_question, target_answer)
        torch.cuda.synchronize()
        
        # Hack for the compatability of reinforce() and DataParallel()
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
        output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True)[0] 
        loss_q = F.cross_entropy(output, target_question)
        loss = loss_q
        # measure accuracy 
        meters['loss_q'].update(loss_q.data[0], n=batch_size)
        # meters['bleu_score'].update(bleu_score, n=batch_size)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

    print('[Val]\tEpoch: [{0}]'
                  'Time {batch_time.avg:.3f}\t'
                  'Q Loss: {loss_q.avg:.3f},\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'],loss_q=meters['loss_q']))


    logger.log_meters('val', n=epoch)
    return -1 * meters['loss_q'].avg, meters['acc5'].avg, meters['acc10'].avg # ugly hack for original main function


# to generate single image result with beam search
def generate(resized_img, cnn_model, vqg_model, ):
    raise NotImplementedError
