import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as pytorch_models
from vqg.lib.utils import set_trainable, set_trainable_param

import pdb

import sys
sys.path.append('vqa/external/pretrained-models.pytorch')
import pretrainedmodels as torch7_models

pytorch_resnet_names = sorted(name for name in pytorch_models.__dict__
    if name.islower()
    and name.startswith("resnet")
    and callable(pytorch_models.__dict__[name]))

torch7_resnet_names = sorted(name for name in torch7_models.__dict__
    if name.islower()
    and callable(torch7_models.__dict__[name]))

model_names = pytorch_resnet_names + torch7_resnet_names


def factory(opt, cuda=True, data_parallel=True, define_forward=True):
    opt = copy.copy(opt)

    # forward_* will be better handle in futur release
    def forward_resnet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if opt.get('conv', False):
            return x

        x = self.layer4(x)
        if 'pooling' in opt and opt['pooling']:
            x = self.avgpool(x)
            div = x.size(3) + x.size(2)
            x = x.sum(3)
            x = x.sum(2)
            x = x.view(x.size(0), -1)
            x = x.div(div)
        return x

    def forward_resnext(self, x):
        x = self.features(x)
        if 'pooling' in opt and opt['pooling']:
            x = self.avgpool(x)
            div = x.size(3) + x.size(2)
            x = x.sum(3)
            x = x.sum(2)
            x = x.view(x.size(0), -1)
            x = x.div(div)

        return x

    if opt['arch'] in pytorch_resnet_names:
        model = pytorch_models.__dict__[opt['arch']](pretrained=True)

        

    elif opt['arch'] == 'fbresnet152':
        model = torch7_models.__dict__[opt['arch']](num_classes=1000,
                                                    pretrained='imagenet')

    elif opt['arch'] in torch7_resnet_names:
        model = torch7_models.__dict__[opt['arch']](num_classes=1000,
                                                    pretrained='imagenet')
    else:
        raise ValueError

    # As utilizing the pretrained_model on 224 image, 
    # when applied on 448 images, please set the corresponding [dilation]
    pdb.set_trace()
    set_dilation(model, opt.get('dilation', 1))

    # To use the factory to retrieve the original model
    if define_forward:
        convnet = model # ugly hack in case of DataParallel wrapping
        model.forward = lambda x: forward_resnet(convnet, x)
    
    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model


class ResNet(nn.Module):
    def __init__(self, arch, pooling=False):
        super(ResNet, self).__init__()
        convnet = factory({'arch':arch}, cuda=False, data_parallel=False, define_forward=False)
        self.conv1 = convnet.conv1
        self.bn1 = convnet.bn1
        self.relu = convnet.relu
        self.maxpool = convnet.maxpool
        self.layer1 = convnet.layer1
        self.layer2 = convnet.layer2
        self.layer3 = convnet.layer3
        self.layer4 = convnet.layer4
        self.pooling = pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.pooling:
            x = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = x.view(x.size(0), -1)
        return x

    def set_trainable(self, trainable=False):
        set_trainable(self, False)
        if trainable:
            set_trainable(self.layer4, True)

def set_dilation(model, dilation=2):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.dilation = (dilation, dilation)
            m.padding = tuple([item * dilation for item in m.padding])
