import time
import torch
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
import vqa.lib.utils as utils
from vqg.models.vec2seq import process_lengths_sort
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import modified_precision as bleu_score

# this function jointly evaluate the VQG and VQA model

def evaluate(loader, model_vqa, model_vqg, logger, print_freq=10, sampling_num=5):

    model_vqa.eval()
    model_vqg.eval()
    meters = logger.reset_meters('evaluate')
    results = []
    model_vqg.is_testing = True
    end = time.time()
    blue_score_all = 0
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        input_visual = Variable(sample['visual'].cuda(async=True), volatile=True )
        target_answer = sample['answer']
        input_question = Variable(sample['question'].cuda(async=True), volatile=True)
        # compute output
        output_answer = model_vqa(input_visual, input_question)
        acc1, acc5, acc10 = utils.accuracy(output_answer.cpu().data, target_answer, topk=(1, 5, 10))
        meters['acc1'].update(acc1, n=batch_size)
        meters['acc5'].update(acc5, n=batch_size)
        meters['acc10'].update(acc10, n=batch_size)
        g_answers_score, g_answers = torch.topk(F.softmax(output_answer), sampling_num)
        generated_q = []
        # answered_a = []
        for j in range(sampling_num):
            generated_q.append(model_vqg(prepare_input(input_visual, model_vqg.opt['mode']), g_answers[:, j].contiguous()))

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

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, len(loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time']))

    print(' ** Result: Acc@1:{}\tAcc@5:{}\tAcc@10:{}'.format(
        meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg))
    model_vqg.is_testing = False
    pdb.set_trace()
    model_vqa.train()
    model_vqg.train()
    return results

def translate_tokens(token_seq, wid_to_word, end_id = 0):
    sentence = []
    for wid in token_seq:
        if wid == end_id:
            break
        sentence.append(wid_to_word[wid])
    return sentence
def prepare_input(input_visual, mode):
    if mode == 'noatt' and input_visual.dim() == 4:
        return F.avg_pool2d(input_visual, input_visual.size(2)).squeeze()
    else:
        return input_visual
