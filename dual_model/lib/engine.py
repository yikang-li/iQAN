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
from models.utils import translate_tokens


def train(loader, model, optimizer, logger, epoch, print_freq=10, dual_training=False, alternative_train = -1.):
    # switch to train mode
    model.train()
    model.module.set_testing(False)
    model.module.use_reinforce=False

    meters = logger.reset_meters('train')
    end = time.time()
    for i, sample in enumerate(loader):

        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        target_question = sample['question']
        # To arrange the length of mini-batch by the descending order of question length
        lengths = process_lengths(target_question) 
        target_question = Variable(target_question.cuda())
        input_visual = Variable(sample['visual'].cuda())
        target_answer = Variable(sample['answer'].cuda(async=True))
        
        # compute output
        generated_a, selected_a, generated_q, loss_q = model(input_visual, target_question, target_answer)
        torch.cuda.synchronize()
        
        #pdb.set_trace()
        if target_answer.dim() == 2: # use BCE loss
            loss_a = sampled_bce_loss(generated_a, target_answer)
            vqg_learn = torch.gather(target_answer, 1, selected_a).type(torch.cuda.FloatTensor)
        else:
            loss_a = F.cross_entropy(generated_a, target_answer)
            vqg_learn = selected_a.eq(target_answer).type(torch.cuda.FloatTensor)

        # loss_q = (loss_q * vqg_learn).sum() / (vqg_learn.sum() + float(np.finfo(np.float32).eps))
        loss_q = loss_q.mean()
        if alternative_train > 1. or alternative_train < 0.:
          loss = loss_a + loss_q
        else:
          if torch.rand(1)[0] > alternative_train:
            loss = loss_a
          else:
            loss = loss_q
        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)
        meters['loss_a'].update(loss_a.data[0], n=batch_size)
        meters['loss_q'].update(loss_q.data[0], n=batch_size)

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
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q']))

    print('[Train]\tEpoch: [{0}] '
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, 
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q']))

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
        lengths = process_lengths(target_question) 
        target_question = Variable(target_question.cuda(async=True), volatile=True)
        input_visual = Variable(sample['visual'].cuda(async=True), volatile=True)
        target_answer = Variable(sample['answer'].cuda(async=True), volatile=True)
        
        # compute output
        generated_a, selected_a, generated_q, loss_q = model(input_visual, target_question, target_answer)
        
        #pdb.set_trace()
        if target_answer.dim() == 2: # use BCE loss
            loss_a = sampled_bce_loss(generated_a, target_answer)
            vqg_learn = torch.gather(target_answer, 1, selected_a).type(torch.cuda.FloatTensor)
        else:
            loss_a = F.cross_entropy(generated_a, target_answer)
            vqg_learn = selected_a.eq(target_answer).type(torch.cuda.FloatTensor)
        
        loss_q = (loss_q * vqg_learn).sum() / (vqg_learn.sum() + float(np.finfo(np.float32).eps))
        loss = loss_a + loss_q
        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)
        meters['loss_a'].update(loss_a.data[0], n=batch_size)
        meters['loss_q'].update(loss_q.data[0], n=batch_size)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('[Val]\tEpoch: [{0}][{1}/{2}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q']))

    print('[Val]\tEpoch: [{0}] '
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, 
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q']))

    logger.log_meters('val', n=epoch)
    return meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg


# to generate single image result with beam search
def generate(resized_img, cnn_model, vqg_model, ):
    raise NotImplementedError


def evaluate(loader, model, logger, print_freq=10, sampling_num=5):
    model.eval()
    model.module.set_testing(True, sample_num=sampling_num)
    meters = logger.reset_meters('test')
    results = []
    end = time.time()
    blue_score_all = 0
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        input_visual = Variable(sample['visual'].cuda(async=True), volatile=True )
        target_answer = sample['answer']
        input_question = Variable(sample['question'].cuda(async=True), volatile=True)
        # compute output
        output_answer, g_answers, g_answers_score, generated_q = model(input_visual, input_question)
        acc1, acc5, acc10 = utils.accuracy(output_answer.cpu().data, target_answer, topk=(1, 5, 10))
        meters['acc1'].update(acc1, n=batch_size)
        meters['acc5'].update(acc5, n=batch_size)
        meters['acc10'].update(acc10, n=batch_size)
        _, g_answers = torch.max(output_answer, dim=1)
        g_answers = g_answers.cpu().data
        g_answers_score = g_answers_score.cpu().data

        for j in range(batch_size):
            sampled_aqa = []
            for k in range(sampling_num):
                new_question = generated_q[k].cpu().data[j].tolist()
                new_answer = g_answers[j, k]
                new_answer_score = g_answers_score[j, k]
                sampled_aqa.append([new_question, new_answer, new_answer_score])
            num_result = {  'gt_question': sample['question'][j][1:].tolist(), #sample['question'][j].numpy(),
                            'gt_answer': sample['answer'][j],
                            'augmented_qa': sampled_aqa,}
            readable_result = {  
                            'gt_question': translate_tokens(sample['question'][j][1:], loader.dataset.wid_to_word), 
                            'gt_answer': loader.dataset.aid_to_ans[sample['answer'][j]], 
                            'augmented_qa': [ [
                                        translate_tokens(item[0], loader.dataset.wid_to_word), # translate question
                                        loader.dataset.aid_to_ans[item[1]], # translate answer
                                        ] for item in sampled_aqa],}
            results.append({'image': sample['image'][j], 
                            'numeric_result': num_result, 
                            'readable_result': readable_result}, )
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i + 1, len(loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time']))

    print('** Result: Acc@1:{}\tAcc@5:{}\tAcc@10:{}\tTime: {batch_time.avg:.3f}'.format(
        meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg, batch_time=meters['batch_time']))

    model.module.set_testing(False)
    return results
