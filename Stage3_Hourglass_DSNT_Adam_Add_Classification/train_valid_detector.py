from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import runpy
import numpy as np
import os
import cv2
import random
import torchvision
from data import get_train_valid_set
from dsnt.model import build_mpii_pose_model

from tensorboardX import SummaryWriter
from log import Logger
from dsnt.hyperparam_scheduler import make_1cycle
import seaborn
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.FloatTensor)
logger = Logger('_train')


def train(args, train_loader, valid_loader, model, criterion, optimizer, device, scheduler=None):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []
    for epoch_id in range(epoch):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        ######################
        # training the model #
        ######################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
            cls_conf = batch['confidence']

            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)
            target_conf = cls_conf.to(device)

            # target_pts are the points on 128 * 128, output are the points on 64 * 64 feature maps
            # target_pts are real coordinates, like(64,64), outputs are normalized in (-63 / 64, 63 / 64)
            target_pts = target_pts.view(-1, 21, 2)
            target_pts = target_pts * 63 / (64 * 64) - 63 / 64

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            # output is a list of tensors, each tensor's size is B * n_channels * 2, here, it's B * 21 * 2
            output_pts, output_conf = model(input_img)

            # observe loss
            print('Output_pts:', output_pts, "Output_conf: ", output_conf)
            print('Target_pts:', target_pts, 'Target_conf:', target_conf)

            # MSE in original image size
            # print(output_pts[0].shape)
            mse_loss_train_oi = pts_criterion((output_pts[0] + 63 / 64) * 64 * 64 / 63
                                              * target_conf.reshape(-1, 1, 1).float(),
                                              (target_pts + 63 / 64) * 64 * 64 / 63
                                              * target_conf.reshape(-1, 1, 1).float())
            # MSE in feature map
            mse_loss_train_fm = pts_criterion(output_pts[0] * target_conf.reshape(-1, 1, 1).float(),
                                              target_pts * target_conf.reshape(-1, 1, 1).float())
            train_loss = model.forward_loss(output_pts, target_pts, target_conf.reshape(-1, 1).float())
            # print('Result of model.foward_loss(train):', train_loss.item())
            logger.info('Train==> model.forward_loss: {:.6f}, MSE_loss_fm: {:.6f}, MSE_loss_oi: {:.6f})'.format
                        (float(train_loss.item()),
                         float(mse_loss_train_fm.item()),
                         float(mse_loss_train_oi.item())
                         )
                        )
            cls_criterion = nn.BCELoss()
            train_cls_loss = cls_criterion(output_conf, target_conf.reshape(-1, 1, 1, 1).float())
            logger.info('Train==> model.cls_loss: {:.6f}, pos_acc: {}/{}, neg_acc: {}/{}'.format
                        (float(train_cls_loss.item()),
                         ((output_conf.flatten() >= 0.5) & (target_conf == 1)).sum().item(),
                         (target_conf == 1).sum().item(),
                         ((output_conf.flatten() < 0.5) & (target_conf == 0)).sum().item(),
                         (target_conf == 0).sum().item())
                        )

            train_total_loss = train_loss + train_cls_loss
            train_losses.append(train_total_loss.item())
            # do BP automatically
            train_total_loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(epoch_id, batch_idx * len(img),
                                                                                    len(train_loader.dataset),
                                                                                    100. * batch_idx/len(train_loader),
                                                                                    train_total_loss.item()
                                                                                    )
                      )
        if scheduler:
            print(scheduler.get_lr())
            scheduler.step()
        ######################
        # validate the model #
        ######################
        valid_mean_pts_loss = 0.0
        valid_mean_cls_loss = 0.0

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']
                cls_conf = batch['confidence']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)
                target_conf = cls_conf.to(device)

                # target_pts are the points on 128 * 128, output are the points on 64 * 64 feature maps
                # target_pts are real coordinates, like(64,64), outputs are normalized in (-63 / 64, 63 / 64)
                target_pts = target_pts.view(-1, 21, 2)
                target_pts = target_pts * 63 / (64 * 64) - 63 / 64

                output_pts, output_conf = model(input_img)

                # get loss
                print('Output_pts:', output_pts, "Output_conf: ", output_conf)
                print('Target_pts:', target_pts, 'Target_conf:', target_conf)

                # MSE in original image size
                mse_loss_valid_oi = pts_criterion((output_pts[0] + 63 / 64) * 64 * 64 / 63
                                                  * target_conf.reshape(-1, 1, 1).float(),
                                                  (target_pts + 63 / 64) * 64 * 64 / 63
                                                  * target_conf.reshape(-1, 1, 1).float())

                # MSE in feature map
                mse_loss_valid_fm = pts_criterion(output_pts[0] * target_conf.reshape(-1, 1, 1).float(),
                                                  target_pts * target_conf.reshape(-1, 1, 1).float())
                valid_loss = model.forward_loss(output_pts, target_pts, target_conf.reshape(-1, 1).float())

                print('Result of model.foward_loss(valid):', valid_loss.item())
                logger.info('Valid==> model.forward_loss: {:.6f}, MSE_loss_fm: {:.6f}, MSE_loss_oi: {:.6f})'.format
                            (float(valid_loss.item()),
                             float(mse_loss_valid_fm.item()),
                             float(mse_loss_valid_oi.item())
                             )
                            )

                cls_criterion = nn.BCELoss()
                valid_cls_loss = cls_criterion(output_conf, target_conf.reshape(-1, 1, 1, 1).float())
                logger.info('Valid==> model.cls_loss: {:.6f}, pos_acc: {}/{}, neg_acc: {}/{}'.format
                            (float(valid_cls_loss.item()),
                             ((output_conf.flatten() >= 0.5) & (target_conf == 1)).sum().item(),
                             (target_conf == 1).sum().item(),
                             ((output_conf.flatten() < 0.5) & (target_conf == 0)).sum().item(),
                             (target_conf == 0).sum().item())
                            )

                valid_total_loss = valid_loss + valid_cls_loss
                valid_losses.append(valid_total_loss.item())

                valid_mean_pts_loss += valid_loss.item()
                valid_mean_cls_loss += valid_cls_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            valid_mean_cls_loss /= valid_batch_cnt * 1.0
            print('Valid: pts_loss: {:.6f}, cls_loss: {:.6f}'.format(
                valid_mean_pts_loss,
                valid_mean_cls_loss
                )
            )
        print('====================================================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses


def test(test_loader, model, criterion, device):
    pts_criterion = criterion

    # train_losses = []
    test_losses = []

    test_mean_pts_loss = 0.0
    test_mean_cls_loss = 0.0

    model.eval()  # prep model for evaluation
    with torch.no_grad():
        test_batch_cnt = 0

        for test_batch_idx, batch in enumerate(test_loader):
            test_batch_cnt += 1
            test_img = batch['image']
            landmark = batch['landmarks']
            cls_conf = batch['confidence']

            input_img = test_img.to(device)
            target_pts = landmark.to(device)
            target_conf = cls_conf.to(device)
            # target_pts are the points on 128 * 128, output are the points on 64 * 64 feature maps
            # target_pts are real coordinates, like(64,64), outputs are normalized in (-63 / 64, 63 / 64)
            target_pts = target_pts.view(-1, 21, 2)
            target_pts = target_pts * 63 / (64 * 64) - 63 / 64

            output_pts, output_conf = model(input_img)
            # get loss
            print('Output_pts:', output_pts, "Output_conf: ", output_conf)
            print('Target_pts:', target_pts, 'Target_conf:', target_conf)

            # MSE in original image size
            mse_loss_test_oi = pts_criterion((output_pts[0] + 63 / 64) * 64 * 64 / 63
                                              * target_conf.reshape(-1, 1, 1).float(),
                                              (target_pts + 63 / 64) * 64 * 64 / 63
                                              * target_conf.reshape(-1, 1, 1).float())

            # MSE in feature map
            mse_loss_test_fm = pts_criterion(output_pts[0] * target_conf.reshape(-1, 1, 1).float(),
                                              target_pts * target_conf.reshape(-1, 1, 1).float())
            test_loss = model.forward_loss(output_pts, target_pts, target_conf.reshape(-1, 1).float())

            # print('Result of model.foward_loss(test):', test_loss.item())
            logger.info('Test==> model.forward_loss: {:.6f}, MSE_loss_fm: {:.6f}, MSE_loss_oi: {:.6f})'.format
                        (float(test_loss.item()),
                         float(mse_loss_test_fm.item()),
                         float(mse_loss_test_oi.item())
                         )
                        )

            cls_criterion = nn.BCELoss()
            test_cls_loss = cls_criterion(output_conf, target_conf.reshape(-1, 1, 1, 1).float())
            logger.info('Test==> model.cls_loss: {:.6f}, pos_acc: {}/{}, neg_acc: {}/{}'.format
                        (float(test_cls_loss.item()),
                         ((output_conf.flatten() >= 0.5) & (target_conf == 1)).sum().item(),
                         (target_conf == 1).sum().item(),
                         ((output_conf.flatten() < 0.5) & (target_conf == 0)).sum().item(),
                         (target_conf == 0).sum().item())
                        )

            test_total_loss = test_loss + test_cls_loss
            test_losses.append(test_total_loss.item())

            test_mean_pts_loss += test_loss.item()
            test_mean_cls_loss += test_cls_loss.item()

        test_mean_pts_loss /= test_batch_cnt * 1.0
        test_mean_cls_loss /= test_batch_cnt * 1.0
        print('Test: pts_loss: {:.6f}, cls_loss: {:.6f}'.format(
            test_mean_pts_loss,
            test_mean_cls_loss
            )
        )
    print('====================================================')
    return test_losses


def main_test():
    ##################################################
    # arguments set
    ##################################################
    parser = argparse.ArgumentParser(description='Train Facial Landmarks Detector via DSNT')
    # train test data size set
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    # optimizer choose and related arguments
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='Adam Beta1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam Beta2 (default: 0.999)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--schedule-milestones', type=int, nargs='+',
                        help='list of epochs at which to drop the learning rate')
    parser.add_argument('--schedule-gamma', type=float, metavar='G',
                        help='factor to multiply the LR by at each drop')
    parser.add_argument('--optim', type=str, default='Adam', metavar='S',
                        choices=['sgd', 'rmsprop', '1cycle', 'Adam'],
                        help='optimizer to use (default=rmsprop)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # log related arguments
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='train',  # train, predict, finetune, Test
                        help='training, predicting or finetuning')
    # DSNT model related part
    parser.add_argument('--base-model', type=str, default='hg', metavar='BM',
                        help='base model type (default="hg")')
    # Resnet based arguments
    parser.add_argument('--dilate', type=int, default=0, metavar='N',
                        help='number of ResNet layer groups to dilate (default=0)')
    parser.add_argument('--truncate', type=int, default=0, metavar='N',
                        help='number of ResNet layer groups to cut off (default=0)')
    # HG based arguments
    parser.add_argument('--stacks', type=int, default=1, metavar='N',
                        help='number of Hourglass stacked in the moedl (default=1)')
    parser.add_argument('--blocks', type=int, default=1, metavar='N',
                        help='numbers of Residual blocks in a Residual Unit (default=0)')
    # dsnt related arguments
    parser.add_argument('--output-strat', type=str, default='dsnt', metavar='S',
                        choices=['dsnt', 'gauss', 'fc'],
                        help='strategy for outputting coordinates (default="dsnt")')
    parser.add_argument('--preact', type=str, default='softmax', metavar='S',
                        choices=['softmax', 'thresholded_softmax', 'abs', 'relu', 'sigmoid'],
                        help='heatmap preactivation function (default="softmax")')
    parser.add_argument('--reg', type=str, default='js',
                        choices=['none', 'var', 'js', 'kl', 'mse'],
                        help='set the regularizer (default="js")')
    parser.add_argument('--reg-coeff', type=float, default=5.0,
                        help='coefficient controlling regularization strength (default=5.0, corresponding to paper)')
    parser.add_argument('--hm-sigma', type=float, default=1.0,
                        help='target standard deviation for heatmap, in pixels(default=1.0)')

    args = parser.parse_args()

    if args.optim == 'sgd':
        args.lr = args.lr or 0.0001
        args.schedule_gamma = args.schedule_gamma or 0.5
        args.schedule_milestones = args.schedule_milestones or [20, 40, 60, 80, 120, 140, 160, 180]

    elif args.optim == 'rmsprop':
        args.lr = args.lr or 2.5e-4
        args.schedule_gamma = args.schedule_gamma or 0.1
        args.schedule_milestones = args.schedule_milestones or [60, 90]

    elif args.optim == '1cycle':
        args.lr = args.lr or 1
        args.schedule_gamma = None
        args.schedule_milestones = None

    elif args.optim == 'Adam':
        args.lr = args.lr or 0.005

    ##################################################
    # Some configuration
    ##################################################
    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")    # cuda:0
    # For multi GPUs
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ##################################################
    # Data
    ##################################################
    print('===> Loading Datasets')
    train_set, valid_set, test_set = get_train_valid_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    ##################################################
    # Model
    ##################################################
    print('===> Building Model')
    # for hg base model
    model_desc_hg_base = {
        'base': args.base_model,
        'stacks': args.stacks,
        'blocks': args.blocks,
        'output_strat': args.output_strat,
        'preact': args.preact,
        'reg': args.reg,
        'reg_coeff': args.reg_coeff,
        'hm_sigma': args.hm_sigma,
    }
    # for resnet base model
    model_desc_resnet_base = {
        'base': args.base_model,
        'dilate': args.dilate,
        'truncate': args.truncate,
        'output_strat': args.output_strat,
        'preact': args.preact,
        'reg': args.reg,
        'reg_coeff': args.reg_coeff,
        'hm_sigma': args.hm_sigma,
    }

    if args.base_model.startswith('hg'):
        model = build_mpii_pose_model(**model_desc_hg_base).to(device)
        print("hg's nchans:", model.n_chans)
    elif args.base.model.startswith('resnet'):
        model = build_mpii_pose_model(**model_desc_resnet_base).to(device)
    else:
        raise Exception("Invalid base model:" + args.base_model)

    ##################################################
    # Optimiser
    ##################################################
    # Initialize optimiser and learning rate scheduler
    if args.optim == '1cycle':
        optimizer = optim.SGD(model.parameters(), lr=0)
        scheduler = make_1cycle(optimizer, args.epochs * len(train_loader), lr_max=args.lr, momentum=args.momentum)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        scheduler = None
    else:
        if args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optim == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise Exception('unrecognised optimizer: {}'.format(args.optim))

        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule_milestones, gamma=args.schedule_gamma)

    # (self, params, lr=required, momentum=0, dampening=0,weight_decay=0, nesterov=False)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    ##################################################
    # Loss Select
    ##################################################
    # Notice there'are some Loss Calculation functions im models
    criterion_pts = nn.MSELoss()
    # criterion_pts = nn.SmoothL1Loss()
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = train(args,
                                           train_loader,
                                           valid_loader,
                                           model,
                                           criterion_pts,
                                           optimizer,
                                           device,
                                           scheduler)
        train_losses_ = pd.DataFrame(columns=['train loss'], data=train_losses)
        valid_losses_ = pd.DataFrame(columns=['test loss'], data=valid_losses)
        print(train_losses_.head())
        print(valid_losses_.head())
        train_losses_.to_csv('train_loss.csv')
        valid_losses_.to_csv('valid_loss.csv')
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        model_para_path = 'trained_models/detector_epoch_0.pt'
        model.load_state_dict(torch.load(model_para_path))
        test_losses = test(test_loader, model, criterion_pts, device)
        test_losses_ = pd.DataFrame(columns=['test loss'], data=test_losses)
        print(test_losses_.head())
        test_losses_.to_csv('test_loss.csv')
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        pretrain_para_path = 'trained_models/first_train_with_SGD_lr0.00005_with_flip_similarity/detector_epoch_99.pt'
        model.load_state_dict(torch.load(pretrain_para_path))
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        train_losses_ = pd.DataFrame(columns=['train loss'], data=train_losses)
        valid_losses_ = pd.DataFrame(columns=['test loss'], data=valid_losses)
        print(train_losses_.head())
        print(valid_losses_.head())
        train_losses_.to_csv('train_loss.csv')
        valid_losses_.to_csv('valid_loss.csv')
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?
        model_para_path = 'trained_models/detector_epoch_0.pt'
        model.load_state_dict(torch.load(model_para_path))

        for test_batch_idx, batch in enumerate(test_loader):
            test_img = batch['image']
            gt_landmarks = batch['landmarks']
            gt_conf = batch['confidence']

            # print("GT", gt_landmarks)
            test_img = test_img.to(device)
            test_landmarks, test_conf = model(test_img)
            test_landmarks = test_landmarks[0]
            test_landmarks = (test_landmarks + 63 / 64) * 64 * 64 / 63
            # print("Predict", test_landmarks)
            test_landmarks = test_landmarks.view(-1, 42)

            # print("Gt_conf", gt_conf.shape, gt_conf)
            test_conf = test_conf.flatten()
            # print("test_conf", test_conf.shape, test_conf)
            # heatmap = model.heatmaps
            # for i in range(21):
            #    seaborn.heatmap(heatmap[0, i, :, :].cpu().detach().numpy(), cmap="magma")
            #    plt.show()

            test_img = test_img.cpu().numpy().transpose((0, 2, 3, 1))
            test_landmarks = test_landmarks.cpu().detach().numpy()
            for idx in range(test_img.shape[0]):
                img = test_img[idx].copy()
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for j in range(0, test_landmarks.shape[1], 2):
                    print("Ground Truth: ", (gt_landmarks[idx][j], gt_landmarks[idx][j + 1]))
                    print("Predict point: ", (test_landmarks[idx][j], test_landmarks[idx][j + 1]))
                    cv2.circle(img, (test_landmarks[idx][j], test_landmarks[idx][j + 1]), 2, (0, 0, 255), -1)
                    cv2.circle(img, (gt_landmarks[idx][j], gt_landmarks[idx][j + 1]), 1, (0, 255, 0), -1)
                cv2.imshow('Check the keypoint and image' + str(idx), img)
                print("Ground Truth confidence:", gt_conf[idx])
                print("Test confidence:", test_conf[idx])
                key = cv2.waitKey()
                if key == 27:
                    exit(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main_test()








