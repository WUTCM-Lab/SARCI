
import torch
import numpy as np
import sys
import math
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import shutil
import json
import os


# 保存结果到txt文件
def log_to_txt( contexts=None,filename="save.txt", mark=False,encoding='UTF-8',mode='a'):
    f = open(filename, mode,encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c)+" | "+ str(contexts[c]) +"\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts,list):
            tmp = ""
            for c in contexts:
                tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)

    f.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def collect_match(input):
    """change the model output to the match matrix"""
    image_size = input.size(0)
    text_size = input.size(1)

    # match_v = torch.zeros(image_size, text_size, 1)
    # match_v = match_v.view(image_size*text_size, 1)
    input_ = nn.LogSoftmax(2)(input)
    output = torch.index_select(input_, 2, Variable(torch.LongTensor([1])).cuda())

    return output


def collect_neg(input):
    """"collect the hard negative sample"""
    if input.dim() != 2:
        return ValueError

    batch_size = input.size(0)
    mask = Variable(torch.eye(batch_size)>0.5).cuda()
    output = input.masked_fill_(mask, 0)
    output_r = output.max(1)[0]
    output_c = output.max(0)[0]
    loss_n = torch.mean(output_r) + torch.mean(output_c)
    return loss_n


def calcul_loss(scores, size, margin, loss_type="mse",max_violation=False, text_sim_matrix=None, param = "0.8 | 5"):
    diagonal = scores.diag().view(size, 1)

    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()


def acc_train(input):
    predicted = input.squeeze().numpy()
    batch_size = predicted.shape[0]
    predicted[predicted > math.log(0.5)] = 1
    predicted[predicted < math.log(0.5)] = 0
    target = np.eye(batch_size)
    recall = np.sum(predicted * target) / np.sum(target)
    precision = np.sum(predicted * target) / np.sum(predicted)
    acc = 1 - np.sum(abs(predicted - target)) / (target.shape[0] * target.shape[1])

    return acc, recall, precision


def acc_i2t(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    # ranks_ = np.zeros(image_size//5)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        # index_ = index // 5
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]

            if tmp < rank:
                rank = tmp
        if rank == 1e20:
            print('error')
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)
    # ranks_ = np.zeros(image_size // 5)
    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def shard_dis(images, captions, model, shard_size=128):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

#        print("======================")
#        print("im_start:",im_start)
#        print("im_end:",im_end)

        for j in range(n_cap_shard):
            # sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).float()

            if torch.cuda.is_available():
                im = im.cuda()
                s = s.cuda()

            sim = model(im, s)
            sim = sim.squeeze()
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    # sys.stdout.write('\n')
    return d


def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def prepare_for_map(sims, options):
    image_predictions_val = {}
    image_retrieval_solution_val = {}
    voice_predictions_val = {}
    voice_retrieval_solution_val = {}

    image_size = sims.shape[0]
    voice_size = sims.shape[1]

    if options['dataset']['datatype'] == 'sydney_iv':
        # I --> V
        for index in range(image_size):
            inds = np.argsort(sims[index])[::-1]
            image_predictions_val[index] = inds
            if index in range(0, 49):
                image_retrieval_solution_val[index] = [i for i in range(0, 49 * 5)]
            elif index in range(49, 54):
                image_retrieval_solution_val[index] = [i for i in range(49 * 5, 54 * 5)]
            elif index in range(54, 64):
                image_retrieval_solution_val[index] = [i for i in range(54 * 5, 64 * 5)]
            elif index in range(64, 73):
                image_retrieval_solution_val[index] = [i for i in range(64 * 5, 73 * 5)]
            elif index in range(73, 92):
                image_retrieval_solution_val[index] = [i for i in range(73 * 5, 92 * 5)]
            elif index in range(92, 112):
                image_retrieval_solution_val[index] = [i for i in range(92 * 5, 112 * 5)]
            elif index in range(112, 126):
                image_retrieval_solution_val[index] = [i for i in range(112 * 5, 126 * 5)]

        # V --> I
        voice_sims = sims.T
        for index in range(voice_size):
            inds = np.argsort(voice_sims[index])[::-1]
            voice_predictions_val[index] = inds
            if index in range(0, 49 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(0, 49)]
            elif index in range(49 * 5, 54 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(49, 54)]
            elif index in range(54 * 5, 64 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(54, 64)]
            elif index in range(64 * 5, 73 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(64, 73)]
            elif index in range(73 * 5, 92 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(73, 92)]
            elif index in range(92 * 5, 112 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(92, 112)]
            elif index in range(112 * 5, 126 * 5):
                voice_retrieval_solution_val[index] = [i for i in range(112, 126)]
        return image_predictions_val, image_retrieval_solution_val, voice_predictions_val, voice_retrieval_solution_val

    elif options['dataset']['datatype'] == 'ucm_iv':
        # I --> V
        for index in range(image_size):
            inds = np.argsort(sims[index])[::-1]
            image_predictions_val[index] = inds
            x = int(index / 20)
            image_retrieval_solution_val[index] = [i for i in range(x * 100, (x + 1) * 100)]

        # V --> I
        voice_sims = sims.T
        for index in range(voice_size):
            inds = np.argsort(voice_sims[index])[::-1]
            voice_predictions_val[index] = inds
            x = int(index / 100)
            voice_retrieval_solution_val[index] = [i for i in range(x * 20, (x + 1) * 20)]

        return image_predictions_val, image_retrieval_solution_val, voice_predictions_val, voice_retrieval_solution_val

    elif options['dataset']['datatype'] == 'rsicd_iv':
        image_retrieval_solution_val = json.load(
            open('data/' + options['dataset']['datatype'] + '_precomp/image_retrieval_solution_val.json',
                 encoding="UTF-8"))
        image_retrieval_solution_val = {int(k): v for k, v in image_retrieval_solution_val.items()}
        voice_retrieval_solution_val = json.load(
            open('data/' + options['dataset']['datatype'] + '_precomp/voice_retrieval_solution_val.json',
                 encoding="UTF-8"))
        voice_retrieval_solution_val = {int(k): v for k, v in voice_retrieval_solution_val.items()}

        # I --> V
        for index in range(image_size):
            inds = np.argsort(sims[index])[::-1]
            image_predictions_val[index] = inds

        # V --> I
        voice_sims = sims.T
        for index in range(voice_size):
            inds = np.argsort(voice_sims[index])[::-1]
            voice_predictions_val[index] = inds

        return image_predictions_val, image_retrieval_solution_val, voice_predictions_val, voice_retrieval_solution_val

    else:
        print('error, please extending preparing for new dataset {}'.format(options['dataset']['datatype']))
        return None


def cal_map_p_k(predictions, retrieval_solution):
    """Computes mean average precision for retrieval prediction.
  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.
  Returns:
    mean_ap: Mean average precision score (float).
  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute mAP.
    mean_ap = 0.0
    top_1 = []
    top_5 = []
    top_10 = []
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError(
                'Test image %s is not part of retrieval_solution' % key)

        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        ap = 0.0
        already_predicted = set()
        num_expected_retrieved = len(retrieval_solution[key])
        num_correct = 0
        top_correct = 0
        for i in range(len(prediction)):
            if i < 10:
                if prediction[i] in retrieval_solution[key]:
                    top_correct += 1
                if i == 0:
                    top_1.append(top_correct / 1.0)
                if i == 4:
                    top_5.append(top_correct / 5.0)
                if i == 9:
                    top_10.append(top_correct / 10.0)
            if prediction[i] not in already_predicted:
                if prediction[i] in retrieval_solution[key]:
                    num_correct += 1
                    ap += num_correct / (i + 1)
                already_predicted.add(prediction[i])

        ap /= num_expected_retrieved
        mean_ap += ap

    p_1 = np.mean(top_1)
    p_5 = np.mean(top_5)
    p_10 = np.mean(top_10)
    mean_ap /= num_test_images

    return mean_ap, p_1, p_5, p_10


def cal_map_p_k_rsicd(predictions, retrieval_solution, max_predictions=150):
    """Computes mean average precision for retrieval prediction.
  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.
  Returns:
    mean_ap: Mean average precision score (float).
  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute mAP.
    mean_ap = 0.0
    top_1 = []
    top_5 = []
    top_10 = []
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError(
                'Test image %s is not part of retrieval_solution' % key)

        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        ap = 0.0
        already_predicted = set()
        num_expected_retrieved = min(
            len(retrieval_solution[key]), max_predictions)
        num_correct = 0
        top_correct = 0
        for i in range(min(len(prediction), max_predictions)):
            if i < 10:
                if prediction[i] in retrieval_solution[key]:
                    top_correct += 1
                if i == 0:
                    top_1.append(top_correct / 1.0)
                if i == 4:
                    top_5.append(top_correct / 5.0)
                if i == 9:
                    top_10.append(top_correct / 10.0)
            if prediction[i] not in already_predicted:
                if prediction[i] in retrieval_solution[key]:
                    num_correct += 1
                    ap += num_correct / (i + 1)
                already_predicted.add(prediction[i])

        ap /= num_expected_retrieved
        mean_ap += ap

    p_1 = np.mean(top_1)
    p_5 = np.mean(top_5)
    p_10 = np.mean(top_10)
    mean_ap /= num_test_images

    return mean_ap, p_1, p_5, p_10


def shard_dis_reg(images, captions, model, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(len(images)):
        # im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        im_index = i
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            im = Variable(torch.from_numpy(images[i]), volatile=True).float().unsqueeze(0).expand(len(s), 3, 256, 256).cuda()

            l = lengths[cap_start:cap_end]

            sim = model(im, s, l)[:, 1]



            sim = sim.squeeze()
            d[i, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def save_checkpoint(state, is_best, filename, prefix='', model_name=None):
    if is_best:
        torch.save(state, prefix + filename)

        files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".tar"),
                              map(lambda f: os.path.join(prefix, f), os.listdir(prefix))),
                       key=os.path.getmtime)
        # delete all but most current file to assure the latest model is available even if process is killed
        for file in files[:-5]:
            print("removing old model: {}".format(file))
            os.remove(file)

    #
    # tries = 15
    # error = None
    #
    # # deal with unstable I/O. Usually not necessary.
    # while tries:
    #     try:
    #         # torch.save(state, prefix + filename)
    #         if is_best:
    #             torch.save(state, prefix + model_name +'_best.pth.tar')
    #
    #     except IOError as e:
    #         error = e
    #         tries -= 1
    #     else:
    #         break
    #     print('model save {} failed, remaining {} trials'.format(filename, tries))
    #     if not tries:
    #         raise error


def adjust_learning_rate(options, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

        if epoch % options['optim']['lr_update_epoch'] == options['optim']['lr_update_epoch'] - 1:
            lr = lr * options['optim']['lr_decay_param']

        param_group['lr'] = lr

    print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))


def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts
