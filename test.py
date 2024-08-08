

import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click
import utils
import data
import engine


def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/SYDNEY_AMFMN.yaml', type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle)

    return options


def main(options):
    # choose model
    from layers import MODEL_MAIN as models

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(options)

    model = models.factory(options['model'],
                           cuda=options['model']['cuda'],
                           data_parallel=False)

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'])
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    # evaluate on test set
    sims = engine.validate_test(test_loader, model, options)

    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    current_score = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "R1i:{} R5i:{} R10i:{} medRi:{} meanRi:{}\n R1t:{} R5t:{} R10t:{} medRt:{} meanRt:{}\n Rsum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, current_score
    )

    print(all_score)

    return current_score, [r1i, r5i, r10i, r1t, r5t, r10t, current_score]


def update_options_savepath(options, file):
    updated_options = copy.deepcopy(options)

    updated_options['optim']['resume'] = file

    return updated_options


if __name__ == '__main__':
    options = parser_options()

    # calc ave k results
    last_score = []
    best_score = []
    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start evaluate {}th fold".format(k))

        prefix = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + "/" + str(k) + "/"

        files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".tar"),
                              map(lambda f: os.path.join(prefix, f), os.listdir(prefix))),
                       key=os.path.getmtime)
        for index in range(len(files)):
            # update save path
            update_options = update_options_savepath(options, files[index])
            # run experiment
            current_score, one_score = main(update_options)
            for score in one_score:
                print(score, end=' ')
            print()
            if index == len(files) - 1:
                best_score.append(current_score)
                last_score.append(one_score)
                print('**********best_score are above ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑')

        print("Complete evaluate {}th fold".format(k))

    # ave
    print("\n===================== Ave Score ({}-fold verify) =================".format(options['k_fold']['nums']))
    max_index = best_score.index(np.max(best_score))
    print()
    print()
    print('Best Fold is {} th fold (0 base)'.format(max_index))
    print()
    print('Best Scores are: ', end='')
    for score in last_score[max_index]:
        print(score, end=' ')
    print()
    print()
    print('Average Scores are below ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
    last_score = np.average(last_score, axis=0)
    names = ['R_1_i2v', 'R_5_i2v', 'R_10_i2v', 'R_1_v2i', 'R_5_v2i', 'R_10_v2i', 'mR']
    for name, score in zip(names, last_score):
        print("{}:{}".format(name, score))
    for score in last_score:
        print(score, end=' ')
    print("\n==================================================================")

