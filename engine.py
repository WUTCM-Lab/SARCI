
import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import tensorboard_logger as tb_logger
import logging
from torch.nn.utils.clip_grad import clip_grad_norm


def train(train_loader, model, optimizer, epoch, opt={}):
    # extract value
    margin = opt['optim']['margin']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()

    for i, train_data in enumerate(train_loader):
        images, voices, ids = train_data

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_voice = Variable(voices)

        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_voice = input_voice.cuda()

        scores = model(input_visual, input_voice)
        # torch.cuda.synchronize()
        loss = utils.calcul_loss(scores, input_visual.size(0), margin)

        train_logger.update('L', loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        # torch.cuda.synchronize()
        optimizer.step()
        # torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                    opt['logs']['ckpt_save_path'] + opt['model']['name'] + "_" + opt['dataset']['datatype'] + ".txt"
            )
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)


def validate(val_loader, model, opt={}):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_voice = np.zeros((len(val_loader.dataset), opt['dataset']['voice_samp']))

    for i, val_data in enumerate(val_loader):

        images, voices, ids = val_data

        for (id, img, vc) in zip(ids, (images.numpy().copy()), (voices.numpy().copy())):
            input_visual[id] = img
            input_voice[id] = vc

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    val_bs = opt['dataset']['batch_size_val']
    d = utils.shard_dis(input_visual, input_voice, model, val_bs)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to Voice: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Voice to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1v:{} r5v:{} r10v:{} medrv:{} meanrv:{}\n mR:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1v', r1t, step=model.Eiters)
    tb_logger.log_value('r5v', r5t, step=model.Eiters)
    tb_logger.log_value('r10v', r10t, step=model.Eiters)
    tb_logger.log_value('medrv', medrt, step=model.Eiters)
    tb_logger.log_value('meanrv', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score


def validate_test(val_loader, model, opt={}):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_voice = np.zeros((len(val_loader.dataset), opt['dataset']['voice_samp']))

    for i, val_data in enumerate(val_loader):

        images, voices, ids = val_data

        for (id, img, vc) in zip(ids, (images.numpy().copy()), (voices.numpy().copy())):
            input_visual[id] = img
            input_voice[id] = vc

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    val_bs = opt['dataset']['batch_size_val']
    d = utils.shard_dis(input_visual, input_voice, model, val_bs)

    end = time.time()
    print("calculate similarity time:", end - start)

    return d


def validate_test_when_train(val_loader, model, opt={}):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_voice = np.zeros((len(val_loader.dataset), opt['dataset']['voice_samp']))

    for i, val_data in enumerate(val_loader):

        images, voices, ids = val_data

        for (id, img, vc) in zip(ids, (images.numpy().copy()), (voices.numpy().copy())):
            input_visual[id] = img
            input_voice[id] = vc

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    val_bs = opt['dataset']['batch_size_val']
    d = utils.shard_dis(input_visual, input_voice, model, val_bs)

    end = time.time()
    print("calculate similarity time:", end - start)

    image_predictions_val, image_retrieval_solution_val, voice_predictions_val, voice_retrieval_solution_val = \
        utils.prepare_for_map(d, opt)

    if opt['dataset']['datatype'] == 'rsicd_iv':
        # I --> V
        mean_ap_i2v, p_1_i2v, p_5_i2v, p_10_i2v = utils.cal_map_p_k_rsicd(image_predictions_val,
                                                                          image_retrieval_solution_val)

        # V --> I
        mean_ap_v2i, p_1_v2i, p_5_v2i, p_10_v2i = utils.cal_map_p_k_rsicd(voice_predictions_val,
                                                                          voice_retrieval_solution_val)
    else:
        # I --> V
        mean_ap_i2v, p_1_i2v, p_5_i2v, p_10_i2v = utils.cal_map_p_k(image_predictions_val, image_retrieval_solution_val)

        # V --> I
        mean_ap_v2i, p_1_v2i, p_5_v2i, p_10_v2i = utils.cal_map_p_k(voice_predictions_val, voice_retrieval_solution_val)

    all_score = " mean_ap_i2v:{} p_1_i2v:{} p_5_i2v:{} p_10_i2v:{} \n mean_ap_v2i:{} p_1_v2i:{} p_5_v2i:{} p_10_v2i:{} \n all_numbers:{} {} {} {} {} {} {} {} \n ------\n".format(
        mean_ap_i2v, p_1_i2v, p_5_i2v, p_10_i2v, mean_ap_v2i, p_1_v2i, p_5_v2i, p_10_v2i,
        mean_ap_i2v, p_1_i2v, p_5_i2v, p_10_i2v, mean_ap_v2i, p_1_v2i, p_5_v2i, p_10_v2i
    )

    print(all_score)

    currut_score = mean_ap_i2v + p_1_i2v + p_5_i2v + p_10_i2v + mean_ap_v2i + p_1_v2i + p_5_v2i + p_10_v2i

    return currut_score, all_score

