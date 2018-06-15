import time
import pdb
import torch
from torch.autograd import Variable
import vqa.lib.utils as utils

def train(loader, model, criterion, optimizer, logger, epoch, print_freq=10):
    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input_visual   = Variable(sample['visual']).cuda()
        # pdb.set_trace()
        if model.module.opt['criterion']['type'] == 'bce':
            target_answer  = Variable(sample['answer_all'].cuda(async=True))
        else: # model.opt['criterion'] == 'bce':
            target_answer  = Variable(sample['answer'].cuda(async=True))

        # compute output
        # pdb.set_trace()
        output = model(input_visual)
        torch.cuda.synchronize()
        loss = criterion(output, target_answer)
        meters['loss'].update(loss.data[0], n=batch_size)

        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(output.data, target_answer.data, topk=(1, 5, 10))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i+1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})\t'
                  'Acc@10 {acc10.val:.3f} ({acc10.avg:.3f})'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   loss=meters['loss'], acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10']))
    print('* [Training] Epoch: [{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}'.format(
                   epoch,
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   loss=meters['loss'], acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10']))

    logger.log_meters('train', n=epoch)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(loader, model, criterion, logger, epoch=0, print_freq=10):
    results = []

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        input_visual   = Variable(sample['visual'].cuda(async=True), volatile=True)
        if model.module.opt['criterion']['type'] == 'softmax':
            target_answer  = Variable(sample['answer'].cuda(async=True))
        else: # model.opt['criterion'] == 'bce':
            target_answer  = Variable(sample['answer_all'].cuda(async=True))

        # compute output
        output = model(input_visual)
        loss = criterion(output, target_answer)
        meters['loss'].update(loss.data[0], n=batch_size)

        # measure accuracy and record loss
        acc1, acc5, acc10 = utils.accuracy(output.data, target_answer.data, topk=(1, 5, 10))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        meters['acc10'].update(acc10[0], n=batch_size)
        # compute predictions for OpenEnded accuracy
        _, pred = output.data.cpu().max(1)
        pred.squeeze_()
        for j in range(batch_size):
            results.append({'question_id': sample['question_id'][j],
                            'answer': loader.dataset.aid_to_ans[pred[j]]})

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i+1) % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})\t'
                  'Acc@10 {acc10.val:.3f} ({acc10.avg:.3f})'.format(
                   i+1, len(loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], loss=meters['loss'],
                   acc1=meters['acc1'], acc5=meters['acc5'],
                   acc10=meters['acc10']))
    print('* [validation] Epoch: [{0}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Data {data_time.avg:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Acc@1 {acc1.avg:.3f}\t'
              'Acc@5 {acc5.avg:.3f}\t'
              'Acc@10 {acc10.avg:.3f}'.format(
               epoch,
               batch_time=meters['batch_time'], data_time=meters['data_time'],
               loss=meters['loss'], acc1=meters['acc1'], acc5=meters['acc5'], 
               acc10=meters['acc10']))

    logger.log_meters('val', n=epoch)
    return meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg,


def evaluate(loader, model, logger, epoch=0, print_freq=10):

    results = []

    model.eval()
    meters = logger.reset_meters('test')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        input_visual   = Variable(sample['visual'].cuda(async=True), volatile=True)
        # compute output
        output = model(input_visual)
        # compute predictions for OpenEnded accuracy
        _, pred = torch.topk(output.data.cpu(), 5)
        for j in range(batch_size):
            gt_question = translate_tokens(sample['question'][j], loader.dataset.wid_to_word)
            item = {'gt_answer': loader.dataset.aid_to_ans[sample['answer'][j]],
                    'gt_question': gt_question, 
                    'answer': [loader.dataset.aid_to_ans[a_sample] for a_sample in pred[j]], 
                    'image': sample['image'][j], }
            results.append(item)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                   i, len(loader), batch_time=meters['batch_time']))

    logger.log_meters('test', n=epoch)
    return results


def translate_tokens(token_seq, wid_to_word, end_id = 0):
    sentence = []
    for wid in token_seq:
        if wid == end_id:
            break
        sentence.append(wid_to_word[wid])
    return sentence