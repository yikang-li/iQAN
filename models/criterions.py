import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def sampled_softmax_loss(input, target, retained_num=80):
	batch_size = input.size(0)
	if retained_num < 0:
		retained_num = input.size(1)
	retain_rate = retained_num / input.size(1)
	mask = retain_rate * torch.autograd.Variable(torch.ones(input.size()).type(torch.FloatTensor).cuda(), requires_grad=False)
	mask = torch.bernoulli(mask)
	batch_index = torch.LongTensor(range(batch_size))
	mask[batch_index, target.cpu().data] = 1.
	new_index = torch.cumsum(mask, dim=1)
	new_target = new_index[batch_index, target.cpu().data] - 1
	loss = []
	for i in range(batch_size):
		loss.append(
			F.cross_entropy(
				input[i][mask.type(torch.cuda.ByteTensor)[i]].view(1, -1), new_target[i].type(torch.cuda.LongTensor)))
	return torch.mean(torch.stack(loss, 0))



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


def factory_loss(criterion, cuda=True):
	if criterion['type'] == 'softmax':
		criterion = nn.CrossEntropyLoss()
	elif criterion['type'] == 'bce':
		criterion = sampled_bce_loss
	elif criterion['type'] == 'sampled_softmax':
		criterion = sampled_softmax_loss
	else:
		raise Exception('Unknown criterion type')
	
	return criterion