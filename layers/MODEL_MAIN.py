
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .Modules import *
import copy


class BaseModel(nn.Module):
    def __init__(self, opt={}):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt=opt)

        # voice feature
        self.voice_feature = VoiceFeature(opt=opt)

        self.cross_attention_s = CrossLearning(opt=opt)

        self.Eiters = 0

    def forward(self, img, voice):

        # extract features
        image_feature = self.extract_feature(img)

        # voice features
        voice_feature = self.voice_feature(voice)

        image, voice = self.cross_attention_s(image_feature, voice_feature)
        dual_sim = cosine_similarity(image, voice)

        return dual_sim


def factory(opt, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model
