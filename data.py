
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import yaml
import argparse
import utils
from torch import Tensor
from PIL import Image
from joblib import Parallel, delayed
import soundfile as sf
import random


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class PrepareDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, opt):
        self.data_prepare_loc = opt['dataset']['data_path']

        self.img_path = opt['dataset']['image_path']
        self.voice_path = opt['dataset']['voice_path']

        self.images = []
        self.voices = []

        if data_split != 'test':
            with open(self.data_prepare_loc + '%s_voices_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.voices.append(line.strip())

            with open(self.data_prepare_loc + '%s_images_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.data_prepare_loc + '%s_voices.txt' % data_split, 'rb') as f:
                for line in f:
                    self.voices.append(line.strip())

            self.images = []
            with open(self.data_prepare_loc + '%s_images.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.data_split = data_split
        # self.length = len(self.images)
        # if self.data_split != 'train':
        #     self.length = len(self.voices)
        self.length = len(self.voices)

        # data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != len(self.voices):
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform_image = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation((0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform_image = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.transforms_voice = transforms.Compose([
            transforms.Lambda(pad),
            transforms.Lambda(Tensor)])

        # print('length_voices->:', len(self.voices))
        self.voices_data = list(map(self.read_file, self.voices))

        if self.transforms_voice:
            self.voices_data = Parallel(n_jobs=4, prefer='threads')(
                delayed(self.transforms_voice)(x) for x in self.voices_data)
        assert len(self.voices_data) // 5 == len(self.images)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        voice = self.voices_data[index]

        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform_image(image)  # torch.Size([3, 256, 256])

        return image, voice, index
        # if self.data_split != 'train':
        #     img_id = index // self.im_div
        #     voice = self.voices_data[index]
        #
        #     image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        #     image = self.transform_image(image)  # torch.Size([3, 256, 256])
        #
        #     return image, voice, index
        #
        # img_id = index
        # audio_index = index * self.im_div + random.randint(0, 4)
        #
        # voice = self.voices_data[audio_index]
        #
        # image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        # image = self.transform_image(image)  # torch.Size([3, 256, 256])
        #
        # return image, voice, index

    def __len__(self):
        return self.length

    def read_file(self, voice_file):
        # print('str(voice_file)[2:-1]:', str(voice_file)[2:-1])
        data_voice, sample_rate = sf.read(os.path.join(self.voice_path, str(voice_file)[2:-1]))
        # data_voice = np.pad(data_voice, (300, 300), 'constant', constant_values=(0, 0))
        return data_voice


def collate_fn(data):
    images, voices, ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # Merge voices (convert tuple of 1D tensor to 2D tensor)
    voices = torch.stack(voices, 0)

    return images, voices, ids


def get_precomp_loader(data_split, batch_size=100, shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrepareDataset(data_split, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(opt):
    train_loader = get_precomp_loader('train', opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    test_loader = get_precomp_loader('test', opt['dataset']['batch_size_val'], False, opt['dataset']['workers'],
                                     opt=opt)
    return train_loader, test_loader


def get_test_loader(opt):
    test_loader = get_precomp_loader('test',
                                     opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader
