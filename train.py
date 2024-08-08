
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click
import random
import utils
import data
import engine


def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/UCM_OURS.yaml', type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle)

    return options


def main(options):
    # choose model
    from layers import MODEL_MAIN as models

    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])

    # Create dataset, model, criterion and optimizer
    train_loader, test_loader = data.get_loaders(options)
   
    model = models.factory(options['model'],
                           cuda=options['model']['cuda'],
                           data_parallel=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=options['optim']['lr'])

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if options['optim']['resume']:
        if os.path.isfile(options['optim']['resume']):
            print("=> loading checkpoint '{}'".format(options['optim']['resume']))
            checkpoint = torch.load(options['optim']['resume'])
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
         
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
   
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(options['optim']['resume'], start_epoch, best_rsum))
            currunt_score, all_scores = engine.validate(test_loader, model)
            print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
    else:
        start_epoch = 0

    # Train the Model
    best_rsum = 0
    best_score = ""

    for epoch in range(start_epoch, options['optim']['epochs']):

        utils.adjust_learning_rate(options, optimizer, epoch)

        # train for one epoch
        engine.train(train_loader, model, optimizer, epoch, opt=options)

        # evaluate on validation set
        if epoch % options['logs']['eval_step'] == 0:
            currunt_score, all_scores = engine.validate(test_loader, model, opt=options)

            is_best = currunt_score > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(currunt_score, best_rsum)

            # save ckpt
            utils.save_checkpoint(
                {
                'epoch': epoch + 1,
                'arch': 'baseline',
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'options': options,
                'Eiters': model.Eiters,
                },
                is_best,
                filename='ckpt_{}_{}_{:.6f}.pth.tar'.format(options['model']['name'], epoch, best_rsum),
                prefix=options['logs']['ckpt_save_path'],
                model_name=options['model']['name']
            )

            print("Current {}th fold.".format(options['k_fold']['current_num']))
            print("Now  score:")
            print(all_scores)
            print("Best score:")
            print(best_score)

            utils.log_to_txt(
                contexts="Epoch:{} ".format(epoch) + all_scores,
                filename=options['logs']['ckpt_save_path'] + options['model']['name'] + "_" + options['dataset'][
                    'datatype'] + "_scores.txt"
            )
            utils.log_to_txt(
                contexts="Best:   " + best_score,
                filename=options['logs']['ckpt_save_path'] + options['model']['name'] + "_" + options['dataset'][
                    'datatype'] + "_scores.txt"
            )

            utils.log_to_txt(
                contexts="Epoch:{} ".format(epoch) + all_scores,
                filename=options['logs']['ckpt_save_path'] + options['model']['name'] + "_" + options['dataset'][
                    'datatype'] + ".txt"
            )
            utils.log_to_txt(
                contexts="Best:   " + best_score,
                filename=options['logs']['ckpt_save_path'] + options['model']['name'] + "_" + options['dataset'][
                    'datatype'] + ".txt"
            )


def generate_random_samples(options):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load all anns
    images = utils.load_from_txt(options['dataset']['data_path'] + 'train_images.txt')
    voices = utils.load_from_txt(options['dataset']['data_path'] + 'train_voices.txt')

    # merge
    assert len(voices) // 5 == len(images)
    all_infos = []
    for img_id in range(len(images)):
        voice_id = [img_id * 5, (img_id + 1) * 5]
        all_infos.append([voices[voice_id[0]:voice_id[1]], images[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split
    percent = 0.75

    train_infos = all_infos[:int(len(all_infos) * percent)]
    val_infos = all_infos[int(len(all_infos) * percent):]

    # save to txt
    train_voices = []
    train_images = []
    for item in train_infos:
        for voice_item in item[0]:
            train_voices.append(voice_item)
        train_images.append(item[1])
    utils.log_to_txt(train_voices, options['dataset']['data_path'] + 'train_voices_verify.txt', mode='w')
    utils.log_to_txt(train_images, options['dataset']['data_path'] + 'train_images_verify.txt', mode='w')

    val_voices = []
    val_iamges = []
    for item in val_infos:
        for voice_item in item[0]:
            val_voices.append(voice_item)
        val_iamges.append(item[1])
    utils.log_to_txt(val_voices, options['dataset']['data_path'] + 'val_voices_verify.txt', mode='w')
    utils.log_to_txt(val_iamges, options['dataset']['data_path'] + 'val_images_verify.txt', mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))


def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options


if __name__ == '__main__':
    options = parser_options()
    # make logger
    tb_logger.configure(options['logs']['logger_name'], flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start {}th fold".format(k))
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if options['model']['cuda']:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # gpu

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        main(update_options)
