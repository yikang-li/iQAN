import torch
import torch.nn as nn

def factory(opt, cuda=True):
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()
    return criterion


def sampled_bce_loss(input, target, retained_num=80):
	if retained_num < 0:
		retained_num = input.size(1)
	retain_rate = retained_num / input.size(1)
	mask = retain_rate * torch.autograd.Variable(torch.ones(input.size()).type(torch.FloatTensor).cuda(), requires_grad=False)
	mask = torch.bernoulli(mask)
	mask[target.type(torch.cuda.ByteTensor)] = 1
	# the following functions is from pytorch source code
	max_val = (-input).clamp(min=0)
	loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
	loss = loss * mask
	return loss.sum() / mask.data.type(torch.cuda.FloatTensor).sum()