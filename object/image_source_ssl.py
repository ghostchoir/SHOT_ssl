import argparse
import os
import sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_update
from data import *
import random
import pdb
import math
import copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth, NTXentLoss, SupConLoss, LabelSmoothedSCLLoss, Entropy, FocalLoss, JSDivLoss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(args, optimizer, iter_num, max_iter, gamma=10, power=0.75):
    if args.scheduler == 'default':
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
    elif args.scheduler == 'warmupcos':
        warmup_iter = max_iter * args.warmup_ratio
        if iter_num < warmup_iter:
            decay = iter_num / warmup_iter
        else:
            decay = np.cos((iter_num - warmup_iter) * np.pi /
                           (2 * (max_iter - warmup_iter)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = args.weight_decay
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def temp_scheduler(args, classifier, iter_num, max_iter):
    start = args.angular_temp
    end = args.angular_temp_end
    if args.angular_temp_schedule == 'cosine':
        temp = start + (end-start) * \
            (1 - np.cos(iter_num * np.pi / max_iter)) / 2
        classifier.temp = temp
    else:
        pass


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def data_load(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    if args.dset == 'CIFAR-10-C':
        try:
            dsets["source_tr"] = datasets.CIFAR10(root=args.folder + 'CIFAR-10-C',
                                                  train=True,
                                                  download=False,
                                                  transform=cifar_train(args)
                                                  )
        except:
            dsets["source_tr"] = datasets.CIFAR10(root=args.folder + 'CIFAR-10-C',
                                                  train=True,
                                                  download=True,
                                                  transform=cifar_train(args)
                                                  )
        dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                               num_workers=args.worker, drop_last=True)
        dsets["source_te"] = datasets.CIFAR10(root=args.folder + 'CIFAR-10-C',
                                              train=False,
                                              download=False,
                                              transform=cifar_test()
                                              )
        dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs * 4, shuffle=False,
                                               num_workers=args.worker, drop_last=False)
        dsets["test"] = cifar10c_dset(args)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 4, shuffle=False,
                                          num_workers=args.worker, drop_last=False)

    elif args.dset == 'CIFAR-100-C':
        try:
            dsets["source_tr"] = datasets.CIFAR100(root=args.folder + 'CIFAR-100-C',
                                                   train=True,
                                                   download=False,
                                                   transform=cifar_train(args)
                                                   )
        except:
            dsets["source_tr"] = datasets.CIFAR100(root=args.folder + 'CIFAR-100-C',
                                                   train=True,
                                                   download=True,
                                                   transform=cifar_train(args)
                                                   )
        dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                               num_workers=args.worker, drop_last=True)
        dsets["source_te"] = datasets.CIFAR100(root=args.folder + 'CIFAR-100-C',
                                               train=False,
                                               download=False,
                                               transform=cifar_test()
                                               )
        dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs * 4, shuffle=False,
                                               num_workers=args.worker, drop_last=False)
        dsets["test"] = cifar100c_dset(args)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 4, shuffle=False,
                                          num_workers=args.worker, drop_last=False)
    else:
        if not args.multisource:
            txt_src = open(args.s_dset_path).readlines()
            if args.dset == 'domainnet':
                txt_src_test = open(args.s_dset_test_path).readlines()
            txt_test = open(args.test_dset_path).readlines()

            if not args.da == 'uda':
                label_map_s = {}
                for i in range(len(args.src_classes)):
                    label_map_s[args.src_classes[i]] = i

                new_src = []
                for i in range(len(txt_src)):
                    rec = txt_src[i]
                    reci = rec.strip().split(' ')
                    if int(reci[1]) in args.src_classes:
                        line = reci[0] + ' ' + \
                            str(label_map_s[int(reci[1])]) + '\n'
                        new_src.append(line)
                txt_src = new_src.copy()

                new_tar = []
                for i in range(len(txt_test)):
                    rec = txt_test[i]
                    reci = rec.strip().split(' ')
                    if int(reci[1]) in args.tar_classes:
                        if int(reci[1]) in args.src_classes:
                            line = reci[0] + ' ' + \
                                str(label_map_s[int(reci[1])]) + '\n'
                            new_tar.append(line)
                        else:
                            line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                            new_tar.append(line)
                txt_test = new_tar.copy()
        else:
            if args.dset == 'domainnet':
                txt_src = []
                txt_src_test = []
                for i in range(len(args.names)):
                    if i != args.t:
                        print(args.names[i] + ' added to src dset')
                        dset_path = folder + args.dset + \
                            '/' + names[i] + '_train.txt'
                        txt_src += open(dset_path).readlines()
                        dset_path = folder + args.dset + \
                            '/' + names[i] + '_test.txt'
                        txt_src_test += open(dset_path).readlines()
                txt_test = open(args.test_dset_path).readlines()
            else:
                txt_src = []
                txt_test = open(args.test_dset_path).readlines()
                for i in range(len(args.names)):
                    if i != args.t:
                        print(args.names[i] + ' added to src dset')
                        dset_path = folder + args.dset + \
                            '/' + names[i] + '_list.txt'
                        txt_src += open(dset_path).readlines()

        if args.dset == 'domainnet':
            tr_txt = txt_src
            te_txt = txt_src_test
        else:
            if args.trte == "val":
                dsize = len(txt_src)
                tr_size = int(args.split_ratio * dsize)
                # print(dsize, tr_size, dsize - tr_size)
                tr_txt, te_txt = torch.utils.data.random_split(
                    txt_src, [tr_size, dsize - tr_size])
            elif args.trte == "nosplit":
                tr_txt = txt_src
                te_txt = txt_src
            else:
                dsize = len(txt_src)
                tr_size = int(args.split_ratio * dsize)
                _, te_txt = torch.utils.data.random_split(
                    txt_src, [tr_size, dsize - tr_size])
                tr_txt = txt_src

        dsets["source_tr"] = ImageList(tr_txt, transform=image_train(args))
        if args.ce_weighting:
            cls_dist = [0] * args.class_num
            for img in dsets["source_tr"].imgs:
                _, cls = img
                cls = int(cls)
                cls_dist[cls] += 1 / len(dsets["source_tr"].imgs)

            cls_dist_inv = [1 / p for p in cls_dist]
            min_dist = min(cls_dist_inv)
            cls_dist_inv_norm = [p / min_dist for p in cls_dist_inv]
            args.ce_weight = cls_dist_inv_norm

        if args.class_stratified:
            from paws import ClassStratifiedSampler
            dsets["source_tr"] = ImageList_update(
                tr_txt, transform=image_train(args))
            sampler = ClassStratifiedSampler(dsets["source_tr"], 1, 0, args.per_class_batch_size, args.class_num,
                                             epochs=len(
                                                 dsets["source_tr"]) // (args.per_class_batch_size * args.class_num),
                                             seed=args.seed)
            dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_sampler=sampler, num_workers=args.worker,
                                                   shuffle=False)
        else:
            dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                                   num_workers=args.worker,
                                                   drop_last=False if args.ssl_task == 'none' else True)
        dsets["source_te"] = ImageList(te_txt, transform=image_test(args))
        dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=False,
                                               num_workers=args.worker, drop_last=False)
        dsets["test"] = ImageList(txt_test, transform=image_test(args))
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 4, shuffle=False,
                                          num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netH, netB, netC, args, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[-1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            if args.layer in ['add_margin', 'arc_margin', 'sphere']:
                labels_forward = labels
            else:
                labels_forward = None
            outputs = netC(netB(netF(inputs)), labels_forward)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_acc_oda(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            if len(data) == 3:
                inputs = data[0]
                labels = data[2]
            else:
                inputs = data[0]
                labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output +
                    args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc,
                    n_init=1).fit(ent.reshape(-1, 1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent > threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int), :]

    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])


def train_source(args):
    dset_loaders = data_load(args)
    # set base network
    if args.norm_layer == 'batchnorm':
        norm_layer = nn.BatchNorm2d
    elif args.norm_layer == 'groupnorm':
        def gn_helper(planes):
            return nn.GroupNorm(8, planes)

        norm_layer = gn_helper
    if args.net[0:3] == 'res':
        if '26' in args.net:
            netF = network.ResCifarBase(26, norm_layer=norm_layer)
            args.bottleneck = netF.in_features // 2
        else:
            netF = network.ResBase(res_name=args.net, args=args)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)

    if args.ssl_before_btn:
        netH = network.ssl_head(
            ssl_task=args.ssl_task, feature_dim=netF.in_features, embedding_dim=args.embedding_dim)
    else:
        netH = network.ssl_head(
            ssl_task=args.ssl_task, feature_dim=args.bottleneck, embedding_dim=args.embedding_dim)
    if args.bottleneck != 0:
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck, norm_btn=args.norm_btn)
        netC = network.feat_classifier(type=args.layer,
                                       class_num=args.class_num,
                                       bottleneck_dim=args.bottleneck,
                                       bias=args.classifier_bias,
                                       temp=args.angular_temp,
                                       args=args)
    else:
        netB = nn.Identity()
        netC = network.feat_classifier(type=args.layer,
                                       class_num=args.class_num,
                                       bottleneck_dim=netF.in_features,
                                       bias=args.classifier_bias,
                                       temp=args.angular_temp,
                                       args=args)

    if args.dataparallel:
        netF = nn.DataParallel(netF).cuda()
        netH = nn.DataParallel(netH).cuda()
        netB = nn.DataParallel(netB).cuda()
        netC = nn.DataParallel(netC).cuda()
    else:
        netF.cuda()
        netH.cuda()
        netB.cuda()
        netC.cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        if args.separate_wd and ('bias' in k or 'norm' in k):
            param_group += [{'params': v,
                             'lr': learning_rate * 0.1, 'weight_decay': 0}]
        else:
            param_group += [{'params': v, 'lr': learning_rate *
                             0.1, 'weight_decay': args.weight_decay}]
    for k, v in netH.named_parameters():
        if args.separate_wd and ('bias' in k or 'norm' in k):
            param_group += [{'params': v,
                             'lr': learning_rate, 'weight_decay': 0}]
        else:
            param_group += [{'params': v, 'lr': learning_rate,
                             'weight_decay': args.weight_decay}]
    for k, v in netB.named_parameters():
        if args.separate_wd and ('bias' in k or 'norm' in k):
            param_group += [{'params': v,
                             'lr': learning_rate, 'weight_decay': 0}]
        else:
            param_group += [{'params': v, 'lr': learning_rate,
                             'weight_decay': args.weight_decay}]
    for k, v in netC.named_parameters():
        if args.separate_wd and ('bias' in k or 'norm' in k):
            param_group += [{'params': v,
                             'lr': learning_rate, 'weight_decay': 0}]
        else:
            param_group += [{'params': v, 'lr': learning_rate,
                             'weight_decay': args.weight_decay}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    if args.class_stratified:
        max_iter = args.max_epoch * \
            len(dset_loaders["source_tr"].batch_sampler)
    else:
        max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    epoch = 0

    netF.train()
    netH.train()
    netB.train()
    netC.train()

    if args.use_focal_loss:
        cls_loss_fn = FocalLoss(alpha=args.focal_alpha,
                                gamma=args.focal_gamma, reduction='mean')
    else:
        if args.ce_weighting:
            w = torch.Tensor(args.ce_weight).cuda()
            w.requires_grad = False
            if args.smooth == 0:
                cls_loss_fn = nn.CrossEntropyLoss(weight=w).cuda()
            else:
                cls_loss_fn = CrossEntropyLabelSmooth(
                    num_classes=args.class_num, epsilon=args.smooth, weight=w).cuda()
        else:
            if args.smooth == 0:
                cls_loss_fn = nn.CrossEntropyLoss().cuda()
            else:
                cls_loss_fn = CrossEntropyLabelSmooth(
                    num_classes=args.class_num, epsilon=args.smooth).cuda()

    if args.ssl_task in ['simclr', 'crs']:
        if args.use_new_ntxent:
            ssl_loss_fn = SupConLoss(
                temperature=args.temperature, base_temperature=args.temperature).cuda()
        else:
            ssl_loss_fn = NTXentLoss(
                args.batch_size, args.temperature, True).cuda()
    elif args.ssl_task in ['supcon', 'crsc']:
        ssl_loss_fn = SupConLoss(
            temperature=args.temperature, base_temperature=args.temperature).cuda()
    elif args.ssl_task == 'ls_supcon':
        ssl_loss_fn = LabelSmoothedSCLLoss(
            args.batch_size, args.temperature, args.class_num, args.ssl_smooth)

    if args.cr_weight > 0:
        if args.cr_metric == 'cos':
            dist = nn.CosineSimilarity(dim=1).cuda()
        elif args.cr_metric == 'l1':
            dist = nn.PairwiseDistance(p=1).cuda()
        elif args.cr_metric == 'l2':
            dist = nn.PairwiseDistance(p=2).cuda()
        elif args.cr_metric == 'bce':
            dist = nn.BCEWithLogitsLoss(reduction='sum').cuda()
        elif args.cr_metric == 'kl':
            dist = nn.KLDivLoss(reduction='sum').cuda()
        elif args.cr_metric == 'js':
            dist = JSDivLoss(reduction='sum').cuda()

    use_second_pass = (args.ssl_task in ['simclr', 'supcon', 'ls_supcon']) and (
        args.ssl_weight > 0)
    use_third_pass = (args.cr_weight > 0) or (args.ssl_task in [
        'crsc', 'crs'] and args.ssl_weight > 0) or (args.cls3)

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            if args.class_stratified:
                dset_loaders["source_tr"].batch_sampler.set_epoch(epoch)
            epoch += 1
            inputs_source, labels_source = iter_source.next()

        try:
            if inputs_source.size(0) == 1:
                continue
        except:
            if inputs_source[0].size(0) == 1:
                continue
        temp_scheduler(args, netC, iter_num, max_iter)
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source1 = None
        inputs_source2 = None
        inputs_source3 = None
        labels_source = labels_source.cuda()

        if args.layer in ['add_margin', 'arc_margin', 'shpere']:
            labels_forward = labels_source
        else:
            labels_forward = None

        if type(inputs_source) is list:
            inputs_source1 = inputs_source[0].cuda()
            inputs_source2 = inputs_source[1].cuda()
            if len(inputs_source) == 3:
                inputs_source3 = inputs_source[2].cuda()
        else:
            inputs_source1 = inputs_source.cuda()

        if args.mixup:
            inputs_source1, y_a, y_b, lam = mixup_data(inputs_source1, labels_source,
                                                       args.mixup_alpha)

        if inputs_source1 is not None:
            f1 = netF(inputs_source1)
            b1 = netB(f1)
            outputs_source = netC(b1, labels_forward)
        if use_second_pass:
            f2 = netF(inputs_source2)
            b2 = netB(f2)
        if use_third_pass:
            if args.sg3:
                with torch.no_grad():
                    f3 = netF(inputs_source3)
                    b3 = netB(f3)
                    c3 = netC(b3, labels_forward)
                    conf = torch.max(F.softmax(c3, dim=1), dim=1)[0]
            else:
                f3 = netF(inputs_source3)
                b3 = netB(f3)
                c3 = netC(b3, labels_forward)
                conf = torch.max(F.softmax(c3, dim=1), dim=1)[0]

        if args.cr_weight > 0:
            if args.cr_site == 'feat':
                f_hard = f1
                f_weak = f3
            elif args.cr_site == 'btn':
                f_hard = b1
                f_weak = b3
            elif args.cr_site == 'cls':
                f_hard = outputs_source
                f_weak = c3
                if args.cr_metric in ['kl', 'js']:
                    f_hard = F.softmax(f_hard, dim=-1)
                if args.cr_metric in ['bce', 'kl', 'js']:
                    f_weak = F.softmax(f_weak, dim=-1)
            else:
                raise NotImplementedError

        if args.mixup:
            classifier_loss = mixup_criterion(
                cls_loss_fn, outputs_source, y_a, y_b, lam)
        else:
            classifier_loss = cls_loss_fn(outputs_source, labels_source)
        # if args.cls3:
        #    classifier_loss += cls_loss_fn(c3, labels_source)

        if args.ssl_weight > 0:
            if args.ssl_before_btn:
                z1 = netH(f1, args.norm_feat)
                if use_second_pass:
                    z2 = netH(f2, args.norm_feat)
                if use_third_pass:
                    z3 = netH(f3, args.norm_feat)
            else:
                z1 = netH(b1, args.norm_feat)
                if use_second_pass:
                    z2 = netH(b2, args.norm_feat)
                if use_third_pass:
                    z3 = netH(b3, args.norm_feat)

            if args.ssl_task in 'simclr':
                if args.use_new_ntxent:
                    z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                    ssl_loss = ssl_loss_fn(z)
                else:
                    ssl_loss = ssl_loss_fn(z1, z2)
            elif args.ssl_task == 'supcon':
                z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                ssl_loss = ssl_loss_fn(z, labels=labels_source)
            elif args.ssl_task == 'ls_supcon':
                ssl_loss = ssl_loss_fn(z1, z2, labels_source)
            elif args.ssl_task == 'crsc':
                z = torch.cat([z1.unsqueeze(1), z3.unsqueeze(1)], dim=1)
                ssl_loss = ssl_loss_fn(z, labels_source)
            elif args.ssl_task == 'crs':
                if args.use_new_ntxent:
                    z = torch.cat([z1.unsqueeze(1), z3.unsqueeze(1)], dim=1)
                    ssl_loss = ssl_loss_fn(z)
                else:
                    ssl_loss = ssl_loss_fn(z1, z3)
        else:
            ssl_loss = torch.tensor(0.0).cuda()

        if args.cr_weight > 0:
            try:
                cr_loss = dist(f_hard[conf <= args.cr_threshold],
                               f_weak[conf <= args.cr_threshold]).mean()

                if args.cr_metric == 'cos':
                    cr_loss *= -1
            except:
                print('Error computing CR loss')
                cr_loss = torch.tensor(0.0).cuda()
        else:
            cr_loss = torch.tensor(0.0).cuda()

        if args.ent_weight > 0:
            softmax_out = nn.Softmax(dim=1)(outputs_source)
            entropy_loss = torch.mean(Entropy(softmax_out))
            classifier_loss += args.ent_weight * entropy_loss

        if args.gent_weight > 0:
            softmax_out = nn.Softmax(dim=1)(outputs_source)
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax *
                                      torch.log(msoftmax + args.epsilon))
            classifier_loss -= args.gent_weight * gentropy_loss

        loss = classifier_loss + args.ssl_weight * ssl_loss + args.cr_weight * cr_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netH.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'visda-c':
                acc_s_te, acc_list = cal_acc(
                    dset_loaders['source_te'], netF, netH, netB, netC, args, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(
                    dset_loaders['source_te'], netF, netH, netB, netC, args, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(
                    args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te

                if args.dataparallel:
                    best_netF = netF.module.state_dict()
                    best_netH = netH.module.state_dict()
                    best_netB = netB.module.state_dict()
                    best_netC = netC.module.state_dict()
                else:
                    best_netF = netF.state_dict()
                    best_netH = netH.state_dict()
                    best_netB = netB.state_dict()
                    best_netC = netC.state_dict()

            netF.train()
            netH.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netH, osp.join(args.output_dir_src, "source_H.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netH, netB, netC


def test_target(args):
    dset_loaders = data_load(args)
    # set base network
    if args.norm_layer == 'batchnorm':
        norm_layer = nn.BatchNorm2d
    elif args.norm_layer == 'groupnorm':
        def gn_helper(planes):
            return nn.GroupNorm(8, planes)

        norm_layer = gn_helper
    if args.net[0:3] == 'res':
        if '26' in args.net:
            netF = network.ResCifarBase(26, norm_layer=norm_layer)
        else:
            netF = network.ResBase(res_name=args.net, args=args)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)

    if args.ssl_before_btn:
        netH = network.ssl_head(
            ssl_task=args.ssl_task, feature_dim=netF.in_features, embedding_dim=args.embedding_dim)
    else:
        netH = network.ssl_head(
            ssl_task=args.ssl_task, feature_dim=args.bottleneck, embedding_dim=args.embedding_dim)
    if args.bottleneck != 0:
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck, norm_btn=args.norm_btn)
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck,
                                       bias=args.classifier_bias, args=args)
    else:
        netB = nn.Identity()
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=netF.in_features,
                                       bias=args.classifier_bias, args=args)

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_H.pt'
    netH.load_state_dict(torch.load(args.modelpath))
    try:
        args.modelpath = args.output_dir_src + '/source_B.pt'
        netB.load_state_dict(torch.load(args.modelpath))
    except:
        print('Skipped loading btn for version compatibility')
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))

    if args.dataparallel:
        netF = nn.DataParallel(netF).cuda()
        netH = nn.DataParallel(netH).cuda()
        netB = nn.DataParallel(netB).cuda()
        netC = nn.DataParallel(netC).cuda()
    else:
        netF.cuda()
        netH.cuda()
        netB.cuda()
        netC.cuda()

    netF.eval()
    netH.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(
            dset_loaders['test'], netF, netH, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name,
                                                                                            acc_os2, acc_os1,
                                                                                            acc_unknown)
    else:
        if args.dset in ['visda-c', 'CIFAR-10-C', 'CIFAR-100-C']:
            acc, acc_list = cal_acc(
                dset_loaders['test'], netF, netH, netB, netC, args, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(
                args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'],
                             netF, netH, netB, netC, args, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(
                args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--split_ratio', type=float, default=0.9)
    parser.add_argument('--max_epoch', type=int,
                        default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int,
                        default=64, help="batch_size")
    parser.add_argument('--scheduler', type=str,
                        default='default', choices=['default', 'warmupcos'])
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--norm_layer', type=str,
                        default='batchnorm', choices=['batchnorm', 'groupnorm'])
    parser.add_argument('--worker', type=int, default=8,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['visda-c', 'office', 'office-home', 'office-caltech', 'CIFAR-10-C', 'CIFAR-100-C',
                                 'image-clef', 'modern-office', 'domainnet'])
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--folder', type=str, default='/SSD/euntae/data/')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50',
                        help="vgg16, resnet50, resnet101")
    parser.add_argument('--nopretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn",
                        choices=["linear", "wn", "angular", 'add_margin', 'arc_margin', 'sphere'])
    parser.add_argument('--classifier', type=str,
                        default="bn", choices=["ori", "bn", "ln"])
    parser.add_argument('--classifier_bias_off', action='store_true')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda',
                        choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val',
                        choices=['full', 'val', 'nosplit'])
    parser.add_argument('--ssl_task', type=str, default='crsc',
                        choices=['none', 'simclr', 'supcon', 'ls_supcon', 'crsc', 'crs'])
    parser.add_argument('--ssl_smooth', type=float, default=0.1)
    parser.add_argument('--ssl_weight', type=float, default=0.1)
    parser.add_argument('--cr_weight', type=float, default=0.0)
    parser.add_argument('--cr_metric', type=str, default='cos',
                        choices=['cos', 'l1', 'l2', 'bce', 'kl', 'js'])
    parser.add_argument('--cr_site', type=str, default='btn',
                        choices=['feat', 'btn', 'cls'])
    parser.add_argument('--cr_threshold', type=float, default=1.0)
    parser.add_argument('--angular_temp', type=float, default=0.1)
    parser.add_argument('--angular_temp_end', type=float, default=1.0)
    parser.add_argument('--angular_temp_schedule', type=str, default='cosine')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ssl_before_btn', action='store_true')
    parser.add_argument('--no_norm_img', action='store_true')
    parser.add_argument('--norm_img_mode', type=str,
                        choices=['whitening', 'pmone'], default='whitening')
    parser.add_argument('--norm_feat', action='store_true')
    parser.add_argument('--norm_btn', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--aug1', type=str, default='simclr',
                        choices=['none', 'weak', 'simclr', 'randaug', 'test', 'augmix', 'trivial'])
    parser.add_argument('--aug2', type=str, default='simclr',
                        choices=['none', 'weak', 'simclr', 'randaug', 'test', 'augmix', 'trivial'])
    parser.add_argument('--aug3', type=str, default='weak',
                        choices=['none', 'weak', 'simclr', 'randaug', 'test', 'augmix', 'trivial'])
    parser.add_argument('--ra_n', type=int, default=1)
    parser.add_argument('--ra_m', type=int, default=10)
    parser.add_argument('--sg3', type=str2bool, default=True)
    parser.add_argument('--cls3', type=str2bool, default=False)
    parser.add_argument('--aug_strength', type=float, default=1.0)
    parser.add_argument('--custom_scale', default=True, type=str2bool)
    parser.add_argument('--use_rrc', default=True, type=str2bool)
    parser.add_argument('--nojitter', action='store_true')
    parser.add_argument('--nograyscale', action='store_true')
    parser.add_argument('--nogaussblur', action='store_true')
    parser.add_argument('--disable_aug_for_shape',
                        type=str2bool, default=False)

    parser.add_argument('--dropout_1', type=float, default=0.0)
    parser.add_argument('--dropout_2', type=float, default=0.0)
    parser.add_argument('--dropout_3', type=float, default=0.0)
    parser.add_argument('--dropout_4', type=float, default=0.0)

    parser.add_argument('--ce_weighting', type=str2bool, default=False)

    parser.add_argument('--use_new_ntxent', type=str2bool, default=True)

    parser.add_argument('--metric_s', type=float, default=30.0)
    parser.add_argument('--metric_m', type=float, default=0.5)
    parser.add_argument('--easy_margin', type=str2bool, default=False)

    parser.add_argument('--use_rrc_on_wa', type=str2bool, default=False)

    parser.add_argument('--use_focal_loss', type=str2bool, default=False)
    parser.add_argument('--focal_alpha', type=float, default=0.5)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--separate_wd', type=str2bool, default=False)

    parser.add_argument('--ent_weight', type=float, default=0)
    parser.add_argument('--gent_weight', type=float, default=0)

    parser.add_argument('--class_stratified', type=str2bool, default=False)
    parser.add_argument('--per_class_batch_size', type=int, default=6)

    parser.add_argument('--multisource', type=str2bool, default=False)

    parser.add_argument('--aug_prob_mult', type=float, default=1.0)

    parser.add_argument('--mixup', type=str2bool, default=False)
    parser.add_argument('--mixup-alpha', type=float, default=1.0)

    args = parser.parse_args()

    args.pretrained = not args.nopretrained
    args.norm_img = not args.no_norm_img
    args.jitter = not args.nojitter
    args.grayscale = not args.nograyscale
    args.gaussblur = not args.nogaussblur
    args.classifier_bias = not args.classifier_bias_off

    assert not (args.cr_weight > 0 and args.aug3 == 'none')
    assert not (args.sg3 and args.cls3)

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset in ['office', 'modern-office']:
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'visda-c':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'CIFAR-10-C':
        names = corruptions
        args.class_num = 10
    if args.dset == 'CIFAR-100-C':
        names = corruptions
        args.class_num = 100
    if args.dset == 'image-clef':
        names = ['c', 'i', 'p']
        args.class_num = 12
    if args.dset == 'domainnet':
        names = ['clipart', 'infograph', 'painting',
                 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    args.names = names

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if len(args.gpu_id) > 1:
        args.dataparallel = True
    else:
        args.dataparallel = False
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = args.folder

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
        args.output_dir_src = osp.join(
            args.output, args.da, args.dset, 'source')
        args.name_src = 'source'
    else:
        if not args.multisource:
            args.output_dir_src = osp.join(
                args.output, args.da, args.dset, names[args.s][0].upper())
            args.name_src = names[args.s][0].upper()
        else:
            args.output_dir_src = osp.join(
                args.output, args.da, args.dset, names[args.t][0].upper())
            args.name_src = names[args.t][0].upper()

        if args.dset == 'domainnet':
            args.s_dset_path = folder + args.dset + \
                '/' + names[args.s] + '_train.txt'
            args.s_dset_test_path = folder + args.dset + \
                '/' + names[args.s] + '_test.txt'
            args.test_dset_path = folder + args.dset + \
                '/' + names[args.t] + '_test.txt'
        else:
            args.s_dset_path = folder + args.dset + \
                '/' + names[args.s] + '_list.txt'
            args.test_dset_path = folder + args.dset + \
                '/' + names[args.t] + '_list.txt'

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
            args.t = i
            args.name = names[args.t]
        elif args.multisource:
            if i != args.t:
                continue
            else:
                args.t = i
                args.name = 'MS_' + names[args.t][0].upper()
                args.test_dset_path = folder + args.dset + \
                    '/' + names[args.t] + '_list.txt'
        else:
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            folder = args.folder
            if args.dset == 'domainnet':
                args.s_dset_path = folder + args.dset + \
                    '/' + names[args.s] + '_train.txt'
                args.s_dset_test_path = folder + args.dset + \
                    '/' + names[args.s] + '_test.txt'
                args.test_dset_path = folder + args.dset + \
                    '/' + names[args.t] + '_list.txt'
            else:
                args.s_dset_path = folder + args.dset + \
                    '/' + names[args.s] + '_list.txt'
                args.test_dset_path = folder + args.dset + \
                    '/' + names[args.t] + '_list.txt'

            if args.dset == 'office-home':
                if args.disable_aug_for_shape:
                    if args.s in [1, 2]:
                        args.jitter = False
                        args.grayscale = False
                        args.gaussblur = False
                    else:
                        args.jitter = not args.nojitter
                        args.grayscale = not args.grayscale
                        args.gaussblur = not args.nogaussblur
                if args.da == 'pda':
                    args.class_num = 65
                    args.src_classes = [i for i in range(65)]
                    args.tar_classes = [i for i in range(25)]
                if args.da == 'oda':
                    args.class_num = 25
                    args.src_classes = [i for i in range(25)]
                    args.tar_classes = [i for i in range(65)]

        test_target(args)
