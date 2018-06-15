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
        output =  model(input_visual, target_question, target_answer)
        generated_a = output[0]
        generated_q = output[1]
        additional_loss =output[2].mean()
        torch.cuda.synchronize()
        
        # Hack for the compatability of reinforce() and DataParallel()
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
        output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True)[0] 
        loss_q = F.cross_entropy(output, target_question)
        loss_a = F.cross_entropy(generated_a, target_answer)
        if alternative_train > 1. or alternative_train < 0.:
          loss = loss_a + loss_q 
          if dual_training:
            loss += additional_loss
        else:
          if torch.rand(1)[0] > alternative_train:
            loss = loss_a
          else:
            loss = loss_q
        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)
        meters['loss_a'].update(loss_a.data[0], n=batch_size)
        meters['loss_q'].update(loss_q.data[0], n=batch_size)
        meters['dual_loss'].update(additional_loss.data[0], n=batch_size)
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
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
                   loss_d=meters['dual_loss']))

    print('[Train]\tEpoch: [{0}]'
                  'Time {batch_time.avg:.3f}\t'
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'], 
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
                   loss_d=meters['dual_loss']))

    logger.log_meters('train', n=epoch)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(loader, model, logger, epoch=0, print_freq=100):
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
        output =  model(input_visual, target_question, target_answer)
        generated_a = output[0]
        generated_q = output[1]
        additional_loss =output[2].mean()
        torch.cuda.synchronize()
        
        # Hack for the compatability of reinforce() and DataParallel()
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
        output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True)[0] 
        loss_q = F.cross_entropy(output, target_question)
        loss_a = F.cross_entropy(generated_a, target_answer)
        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)
        meters['loss_a'].update(loss_a.data[0], n=batch_size)
        meters['loss_q'].update(loss_q.data[0], n=batch_size)
        meters['dual_loss'].update(additional_loss.data[0], n=batch_size)
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        # meters['bleu_score'].update(bleu_score, n=batch_size)
        end = time.time()
            
    print('[Val]\tEpoch: [{0}]'
                  'Time {batch_time.avg:.3f}\t'
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'], 
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
                   loss_d=meters['dual_loss']))

    logger.log_meters('val', n=epoch)
    return meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg, meters['loss_q'].avg


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
        input_answer = Variable(sample['answer'].cuda(async=True), volatile=True)
        target_answer = sample['answer']
        input_question = Variable(sample['question'].cuda(async=True), volatile=True)
        output_answer, g_answers, g_answers_score, generated_q = model(input_visual, input_question, input_answer)
        bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        acc1, acc5, acc10 = utils.accuracy(output_answer.cpu().data, target_answer, topk=(1, 5, 10))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)
        meters['bleu_score'].update(bleu_score, n=batch_size)
        g_answers = g_answers.cpu().data
        g_answers_score = g_answers_score.cpu().data

        for j in range(batch_size):
            new_question = generated_q.cpu().data[j].tolist()
            new_answer = g_answers[j]
            new_answer_score = g_answers_score[j]
            sampled_aqa = [[new_question, new_answer, new_answer_score],]
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

    print('* [Evaluation] Result: Acc@1:{acc1.avg:.3f}\t'
          'Acc@5:{acc5.avg:.3f}\tAcc@10:{acc10.avg:.3f}\t'
          'Time: {batch_time.avg:.3f}\t'
          'BLEU: {bleu_score.avg:.5f}'.format(
          acc1=meters['acc1'], acc5=meters['acc5'], acc10=meters['acc10'], 
          batch_time=meters['batch_time'], 
          bleu_score=meters['bleu_score']))

    model.module.set_testing(False)
    return results
