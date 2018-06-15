import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def sampled_bce_loss(input, target):
	retain_rate = 50. / input.size(1)
	mask = retain_rate * torch.autograd.Variable(torch.ones(input.size()).type(torch.FloatTensor).cuda(), requires_grad=False)
	mask = torch.bernoulli(mask)
	mask[target.type(torch.cuda.ByteTensor)] = 1
	# the following functions is from pytorch source code
	max_val = (-input).clamp(min=0)
	loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
	loss = loss * mask
	return loss.sum() / mask.data.type(torch.cuda.FloatTensor).sum()



def get_loss_function(criterion_opt):
	def question_concept_loss(input, target):
		loss = F.cross_entropy(input[0], target[0]) + sampled_bce_loss(input[1], target[1])
		return loss
	return question_concept_loss

def factory_loss(criterion_opt, cuda=True):
	criterion = criterion_opt['criterion']
	if criterion == 'softmax':
		criterion = nn.CrossEntropyLoss()
		if cuda:
			criterion = criterion.cuda()
	elif criterion == 'concept':
		criterion = get_loss_function(criterion)
	else:
		raise Exception('Unknown criterion type')
	
	return criterion