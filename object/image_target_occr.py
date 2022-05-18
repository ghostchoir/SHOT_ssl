import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx, CIFAR10_idx, ImageList_update, ImageList_pl_update
from data import *
from loss import NTXentLoss, SupConLoss, CrossEntropyLabelSmooth, LabelSmoothedSCLLoss, FocalLoss, JSDivLoss
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from scipy.special import softmax
from sklearn.preprocessing import normalize
from paws import get_outputs, get_topk_conf_indices, snn, paws, ClassStratifiedSampler, sharpen


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
        param_group['wd0'] = param_group['weight_decay']
    return optimizer


def lr_scheduler(args, optimizer, iter_num, max_iter, gamma=10, power=0.75):
    if args.scheduler == 'default':
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
    elif args.scheduler == 'warmupcos':
        warmup_iter = max_iter * args.warmup_ratio
        if iter_num < warmup_iter:
            decay = iter_num / warmup_iter
        else:
            decay = np.cos((iter_num - warmup_iter) * np.pi / (2 * (max_iter - warmup_iter)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        # param_group['weight_decay'] = param_group
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    if args.dset == 'CIFAR-10-C':
        dsets["target"] = cifar10c_dset_idx(args)
        dsets["target"].transform = cifar_train(args)
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                            drop_last=False if args.ssl_task == 'none' else True)

        dsets["pl"] = cifar10c_dset_idx(args)

        dset_loaders["pl"] = DataLoader(dsets["pl"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                        num_workers=args.worker,
                                        drop_last=False)

        dsets["test"] = cifar10c_dset_idx(args)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                          num_workers=args.worker, drop_last=False)
    elif args.dset == 'CIFAR-100-C':
        dsets["target"] = cifar100c_dset_idx(args)
        dsets["target"].transform = cifar_train(args)
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                            drop_last=False if args.ssl_task == 'none' else True)

        dsets["pl"] = cifar100c_dset_idx(args)

        dset_loaders["pl"] = DataLoader(dsets["pl"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                        num_workers=args.worker,
                                        drop_last=False)
        dsets["test"] = cifar100c_dset_idx(args)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                          num_workers=args.worker, drop_last=False)
    else:
        txt_tar = open(args.t_dset_path).readlines()
        txt_test = open(args.test_dset_path).readlines()

        if not args.da == 'uda':
            label_map_s = {}
            for i in range(len(args.src_classes)):
                label_map_s[args.src_classes[i]] = i

            new_tar = []
            for i in range(len(txt_tar)):
                rec = txt_tar[i]
                reci = rec.strip().split(' ')
                if int(reci[1]) in args.tar_classes:
                    if int(reci[1]) in args.src_classes:
                        line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                        new_tar.append(line)
                    else:
                        line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                        new_tar.append(line)
            txt_tar = new_tar.copy()
            txt_test = txt_tar.copy()

        dsets["target"] = ImageList_update(txt_tar, transform=image_train(args), return_idx=True)
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                            drop_last=True)

        dsets["pl"] = ImageList_idx(txt_test, transform=image_pl(args))

        dset_loaders["pl"] = DataLoader(dsets["pl"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                        num_workers=args.worker, drop_last=False)

        dsets["cal"] = ImageList_idx(txt_test, transform=image_cal(args))

        dset_loaders["cal"] = DataLoader(dsets["cal"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                         num_workers=args.worker, drop_last=False)

        dsets["test"] = ImageList_idx(txt_test, transform=image_test(args))
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * args.eval_batch_mult, shuffle=False,
                                          num_workers=args.worker, drop_last=False)

    return dset_loaders, txt_tar


def cal_acc(loader, netF, netH, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
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
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_target(args):
    dset_loaders, txt_tar = data_load(args)
    ## set base network
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

    # print(args.ssl_before_btn)
    if args.ssl_before_btn:
        netH = network.ssl_head(ssl_task=args.ssl_task, feature_dim=netF.in_features, embedding_dim=args.embedding_dim)
    else:
        netH = network.ssl_head(ssl_task=args.ssl_task, feature_dim=args.bottleneck, embedding_dim=args.embedding_dim)
    if args.bottleneck != 0:
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck, norm_btn=args.norm_btn)

        if args.reset_running_stats and args.classifier == 'bn':
            netB.norm.running_mean.fill_(0.)
            netB.norm.running_var.fill_(1.)

        if args.reset_bn_params and args.classifier == 'bn':
            netB.norm.weight.data.fill_(1.)
            netB.norm.bias.data.fill_(0.)
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck,
                                       bias=args.classifier_bias, temp=args.angular_temp, args=args)
    else:
        netB = nn.Identity()
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=netF.in_features,
                                       bias=args.classifier_bias, temp=args.angular_temp, args=args)

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath), strict=False)
    modelpath = args.output_dir_src + '/source_H.pt'
    netH.load_state_dict(torch.load(modelpath), strict=False)
    try:
        modelpath = args.output_dir_src + '/source_B.pt'
        netB.load_state_dict(torch.load(modelpath), strict=False)
    except:
        print('Skipped loading btn for version compatibility')
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath), strict=False)
    cls_weights = copy.deepcopy(netC.fc.weight.data).numpy()
    # netC.eval()
    # for k, v in netC.named_parameters():
    #    v.requires_grad = False

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

    if not (args.f_calibrate_mode != 'none' and args.b_calibrate_mode != 'none'):
        calibrate_bn_stats(dset_loaders['pl'], netF, netB, args)

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            if args.separate_wd and ('bias' in k or 'norm' in k):
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay1, 'weight_decay': 0}]
            else:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay1, 'weight_decay': args.weight_decay}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            if args.separate_wd and ('bias' in k or 'norm' in k):
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2, 'weight_decay': 0}]
            else:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2, 'weight_decay': args.weight_decay}]
        else:
            v.requires_grad = False
    for k, v in netH.named_parameters():
        if args.lr_decay2 > 0:
            if args.separate_wd and ('bias' in k or 'norm' in k):
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2, 'weight_decay': 0}]
            else:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2, 'weight_decay': args.weight_decay}]
        else:
            v.requires_grad = False

    c_param_group = []

    for k, v in netC.named_parameters():
        if args.separate_wd and ('bias' in k or 'norm' in k):
            c_param_group += [{'params': v, 'lr': args.c_lr, 'weight_decay': 0}]
        else:
            c_param_group += [{'params': v, 'lr': args.c_lr, 'weight_decay': args.c_weight_decay}]

    if args.cls_smooth == 0:
        cls_loss_fn = nn.CrossEntropyLoss()
    else:
        cls_loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.cls_smooth)

    if args.cr_weight != 0 or args.paws_cr_weight != 0:
        if args.cr_metric == 'cos':
            dist = nn.CosineSimilarity(dim=1).cuda()
        elif args.cr_metric == 'l1':
            dist = nn.PairwiseDistance(p=1)
        elif args.cr_metric == 'l2':
            dist = nn.PairwiseDistance(p=2)
        elif args.cr_metric == 'bce':
            dist = nn.BCEWithLogitsLoss(reduction=args.cr_reduction).cuda()
        elif args.cr_metric == 'kl':
            dist = nn.KLDivLoss(reduction=args.cr_reduction).cuda()
        elif args.cr_metric == 'js':
            dist = JSDivLoss(reduction=args.cr_reduction).cuda()

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    c_optimizer = optim.SGD(c_param_group)
    c_optimizer = op_copy(c_optimizer)

    if args.initial_memax > 0:
        from paws import maxent_step
        memax_iter = 0
        memax_max_iter = args.initial_memax

        iter_memax = iter(dset_loaders["target"])

        while memax_iter < memax_max_iter:
            try:
                inputs, labels, tar_idx = iter_memax.next()
            except:
                iter_memax = iter(dset_loaders["target"])
                inputs, labels, tar_idx = iter_memax.next()

            if args.wa_to_memax:
                inputs = inputs[1].cuda()
            else:
                inputs = inputs[0].cuda()

            ent, gent = maxent_step(inputs, netF, netH, netB, netC, optimizer)
            memax_iter += 1
            if memax_iter % args.memax_print_freq == 0:
                print("Iter {:3d}/{:3d} Ent {:.3f} Gent {:.3f}".format(memax_iter, memax_max_iter, ent, gent))

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    backup_dset = copy.deepcopy(dset_loaders["target"].dataset)
    use_hc = (args.paws_weight != 0 or args.paws_cr_weight != 0 or args.paws_cls_weight != 0)
    interval_iter = max_iter // args.interval
    hc_interval_iter = max_iter // args.hc_interval
    gt_labels = np.array(copy.deepcopy(dset_loaders["target"].dataset.targets))
    iter_num = 0

    if args.paws_cls_weight != 0 or args.paws_cr_weight != 0 or args.to_nn_weight != 0:
        if args.paws_cls_smooth == 0:
            paws_cls_fn = nn.CrossEntropyLoss()
        else:
            paws_cls_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.paws_cls_smooth)

    mem_label = None
    hc_active = False
    logs = {}

    while iter_num < max_iter:

        if iter_num % interval_iter == 0:
            netF.eval()
            netH.eval()
            netB.eval()
            netC.eval()

            args.label_mutate_p = max(0, args.label_mutate_p + args.mutate_p_delta)
            c, p, f, o = get_outputs(dset_loaders['cal'], netF, netH, netB, netC)
            hc_idxs = get_topk_conf_indices(c, p, n_classes=args.class_num, k=args.hc_topk, threshold=args.hc_threshold)

            if len(hc_idxs) != 0:
                print('HC set mined')
                prev_topk = args.hc_topk
                args.hc_topk = min(args.hc_topk + args.hc_topk_increase, args.hc_topk_max)
                print('HC topk change', prev_topk, '->', args.hc_topk)
                min_conf = np.min(p[hc_idxs])
                if args.exclude_lc:
                    if args.reset_lc:
                        tgt_dataset = copy.deepcopy(backup_dset)
                    tgt_dataset.exclude(hc_idxs)
                    dset_loaders["target"] = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.worker, drop_last=True, pin_memory=True)
                    max_iter = args.max_epoch * len(dset_loaders["target"])
                    interval_iter = max_iter // args.interval
                    hc_interval_iter = max_iter // args.hc_interval
                    # iter_test = iter(dset_loaders["target"])

                hc_set = ImageList_pl_update(txt_tar, p[hc_idxs], transform=image_hc(args), idxs=hc_idxs)
                hc_bsize = min(args.paws_batch_size, len(hc_set) // args.class_num)
                hc_sampler = ClassStratifiedSampler(hc_set, 1, 0, hc_bsize, args.class_num, seed=args.seed)
                print('HC bsize is set to', hc_bsize)

                hc_loader = DataLoader(hc_set, batch_sampler=hc_sampler, shuffle=False)

                iter_hc = iter(hc_loader)

                args.hc_threshold = min(args.hc_threshold + args.hc_threshold_increase, args.hc_threshold_max)
                hc_active = True
            else:
                hc_active = False
                print('HC set mining failed. Falling back to default learning..')

            eval_off = False
            if args.aug_pl == args.aug_cal:
                pl, mem_conf, centroids, labelset, pred = obtain_label(dset_loaders['pl'], netF, netH, netB, netC,
                                                                       args, mem_label, eval_off, f, o, gt_labels)
            else:
                pl, mem_conf, centroids, labelset, pred = obtain_label(dset_loaders['pl'], netF, netH, netB, netC,
                                                                       args, mem_label, eval_off)
            mem_label = torch.from_numpy(pl).cuda()

            netF.train()
            netH.train()
            netB.train()
            netC.train()

            if use_hc and args.sspl_agreement and hc_active:
                agree_idxs = np.where(pl == p)[0]
                hc_idxs = np.where(c >= args.agreement_threshold)[0]
                filter_idxs = np.intersect1d(agree_idxs, hc_idxs)

                topk_idxs = get_topk_conf_indices(c[filter_idxs], p[filter_idxs], args.class_num, args.agreement_topk,
                                                  args.agreement_threshold)
                print(len(filter_idxs), len(topk_idxs))
                hc_set.include(pl[topk_idxs], topk_idxs)
                hc_bsize = min(args.paws_batch_size, len(hc_set) // args.class_num)
                hc_sampler = ClassStratifiedSampler(hc_set, 1, 0, hc_bsize, args.class_num, seed=args.seed)
                print('HC bsize is set to', hc_bsize)
                hc_loader = DataLoader(hc_set, batch_sampler=hc_sampler, shuffle=False)

                iter_hc = iter(hc_loader)
                if args.exclude_agreement_lc:
                    if args.reset_lc and not args.exclude_lc:
                        tgt_dataset = copy.deepcopy(backup_dset)
                    tgt_dataset.exclude(topk_idxs)
                    dset_loaders["target"] = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.worker, drop_last=True, pin_memory=True)
                    max_iter = args.max_epoch * len(dset_loaders["target"])
                    interval_iter = max_iter // args.interval
                    hc_interval_iter = max_iter // args.hc_interval
                    # iter_test = iter(dset_loaders["target"])
                    print(len(hc_set), len(tgt_dataset), iter_num, max_iter, interval_iter)

        try:
            inputs_test, labels_test, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            print("Tgt dataloader reloaded", len(iter_test))
            inputs_test, labels_test, tar_idx = iter_test.next()

        try:
            if inputs_test.size(0) == 1:
                continue
        except:
            if inputs_test[0].size(0) == 1:
                continue

        inputs_test1 = None
        inputs_test2 = None
        inputs_test3 = None

        if args.upper_bound_run:
            pred = labels_test.cuda()
        else:
            if mem_label is not None:
                pred = mem_label[tar_idx]
            else:
                pred = None

        inputs_test1 = inputs_test[0].cuda()
        inputs_test2 = inputs_test[1].cuda()
        # if len(inputs_test) == 3:
        #    inputs_test3 = inputs_test[2].cuda()

        labels_forward = None

        if inputs_test is not None:
            f1 = netF(inputs_test1)
            b1 = netB(f1)
            try:
                for p in netC.parameters():
                    p.requires_grad = False
            except:
                for p in netC.module.parameters():
                    p.requires_grad = False
            outputs_test = netC(b1, labels_forward)
            conf1 = torch.max(F.softmax(outputs_test, dim=1), dim=1)[0]
        if args.sg2:
            with torch.no_grad():
                f2 = netF(inputs_test2)
                b2 = netB(f2)
                c2 = netC(b2, labels_forward)
                conf2 = torch.max(F.softmax(c2, dim=1), dim=1)[0]
        else:
            f2 = netF(inputs_test2)
            b2 = netB(f2)
            c2 = netC(b2, labels_forward)
            conf2 = torch.max(F.softmax(c2, dim=1), dim=1)[0]

        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter, gamma=args.gamma, power=args.power)
        lr_scheduler(args, c_optimizer, iter_num=iter_num, max_iter=max_iter, gamma=args.gamma, power=args.power)

        if args.cr_weight > 0:
            if args.cr_site == 'feat':
                f_hard = f1
                f_weak = f2
            elif args.cr_site == 'btn':
                f_hard = b1
                f_weak = b2
            elif args.cr_site == 'cls':
                f_hard = outputs_test
                f_weak = c2
                if args.cr_metric == 'kl':
                    f_hard = F.log_softmax(f_hard, dim=-1)
                if args.cr_metric in ['bce', 'kl']:
                    f_weak = F.softmax(f_weak, dim=-1)
            else:
                raise NotImplementedError

        if hc_active:
            try:
                inputs_hc, labels_hc, _ = iter_hc.next()
            except:
                iter_hc = iter(hc_loader)
                inputs_hc, labels_hc, _ = iter_hc.next()

            inputs_hc = inputs_hc.cuda()
            b_hc = netB(netF(inputs_hc))
            c_hc = netC(b_hc, labels_forward)

            labels_hc = labels_hc.cuda()
            labels_hc_onehot = F.one_hot(labels_hc, num_classes=args.class_num) \
                               * (1 - (1 + 1 / args.class_num) * args.paws_smoothing)
            labels_hc_onehot += (1 / args.class_num) * args.paws_smoothing

            # b3: (b_x, d_b) | b_hc: (b_hc, d_b) | labels_hc_onehot: (b_hc, n_c) => (b_x, n_c)
            if args.paws_path == 1:
                paws_p1 = snn(b2, b_hc, labels_hc_onehot, tau=args.paws_temp)
            elif args.paws_path == 0:
                paws_p1 = snn(b1, b_hc, labels_hc_onehot, tau=args.paws_temp)

            gt = torch.from_numpy(gt_labels[tar_idx])
            _, paws_pred = torch.max(paws_p1, dim=1)

            agree = pred == paws_pred
            disagree = pred != paws_pred

            hc = conf1 >= args.batch_threshold
            lc = conf1 < args.batch_threshold

            ah = torch.logical_and(agree, hc)
            al = torch.logical_and(agree, lc)
            dh = torch.logical_and(disagree, hc)
            dl = torch.logical_and(disagree, lc)

            ah_correct = pred[ah].cpu() == gt[ah]
            al_correct = pred[al].cpu() == gt[al]
            dh_correct = pred[dh].cpu() == gt[dh]
            dl_correct = pred[dl].cpu() == gt[dl]
            dh_correct_paws = paws_pred[dh].cpu() == gt[dh]
            dl_correct_paws = paws_pred[dl].cpu() == gt[dl]

            if args.full_logging:
                print('AH', ah.sum().item(), ah_correct.sum().item(), '\nAL',
                      al.sum().item(), al_correct.sum().item(), '\nDH',
                      dh.sum().item(), dh_correct.sum().item(), dh_correct_paws.sum().item(), '\nDL',
                      dl.sum().item(), dl_correct.sum().item(), dl_correct_paws.sum().item())

            if random.random() <= args.label_mutate_p and args.mutate_mode != '':
                mut_idx = torch.zeros_like(agree)
                if 'conf' in args.mutate_mode:
                    mut_idx = torch.rand(conf1.size()).cuda() > conf1
                    if 'ah' in args.mutate_mode:
                        mut_idx = torch.logical_and(mut_idx, ah)
                    if 'al' in args.mutate_mode:
                        mut_idx = torch.logical_and(mut_idx, al)
                    if 'dh' in args.mutate_mode:
                        mut_idx = torch.logical_and(mut_idx, dh)
                    if 'dl' in args.mutate_mode:
                        mut_idx = torch.logical_and(mut_idx, dl)
                else:
                    if 'ah' in args.mutate_mode:
                        mut_idx = torch.logical_or(mut_idx, ah)
                    if 'al' in args.mutate_mode:
                        mut_idx = torch.logical_or(mut_idx, al)
                    if 'dh' in args.mutate_mode:
                        mut_idx = torch.logical_or(mut_idx, dh)
                    if 'dl' in args.mutate_mode:
                        mut_idx = torch.logical_or(mut_idx, dl)
                if torch.count_nonzero(mut_idx) > 0:
                    if args.mutate_mode != '':
                        if args.mutate_to == 'random':
                            pred[mut_idx] = torch.randint(low=0, high=args.class_num, size=pred[mut_idx].size()).cuda()
                        elif args.mutate_to == 'second':
                            second_largest = torch.kthvalue(paws_p1, k=2, dim=1)[1].cuda()
                            pred[mut_idx] = second_largest[mut_idx]

            if args.cls_weight != 0 and args.cls_mode != '':
                cls_idx = torch.zeros_like(agree)
                if 'ah' in args.cls_mode:
                    cls_idx = torch.logical_or(cls_idx, ah)
                if 'al' in args.cls_mode:
                    cls_idx = torch.logical_or(cls_idx, al)
                if 'dh' in args.cls_mode:
                    cls_idx = torch.logical_or(cls_idx, dh)
                if 'dl' in args.cls_mode:
                    cls_idx = torch.logical_or(cls_idx, dl)
                if torch.count_nonzero(cls_idx) > 0:
                    classifier_loss = cls_loss_fn(outputs_test, pred)
                    if args.cls_scheduling in ['const', 'step']:
                        classifier_loss *= args.cls_weight
                    elif args.cls_scheduling == 'linear':
                        classifier_loss *= (args.cls_weight * iter_num / max_iter)
                    if iter_num < interval_iter * args.skip_multiplier and args.cls_scheduling == 'step':
                        classifier_loss *= 0
                else:
                    classifier_loss = torch.tensor(0.0).cuda()
                logs['kmeans'] = classifier_loss.item() / args.cls_weight
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            softmax_out = nn.Softmax(dim=1)(outputs_test)

            if args.memax_weight != 0 and args.memax_mode != '':
                memax_idx = torch.zeros_like(agree)
                if 'ah' in args.memax_mode:
                    memax_idx = torch.logical_or(memax_idx, ah)
                if 'al' in args.memax_mode:
                    memax_idx = torch.logical_or(memax_idx, al)
                if 'dh' in args.memax_mode:
                    memax_idx = torch.logical_or(memax_idx, dh)
                if 'dl' in args.memax_mode:
                    memax_idx = torch.logical_or(memax_idx, dl)
                msoftmax = softmax_out[memax_idx].mean(dim=0)
                me = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                classifier_loss -= args.memax_weight * me
                logs['memax'] = me.item()
            if args.ce_hc:
                ce_hc = cls_loss_fn(c_hc, labels_hc)
                classifier_loss += ce_hc
                logs['ce_hc'] = ce_hc.item()
            if args.minent_hc:
                hc_softmax_out = nn.Softmax(dim=1)(c_hc)
                paws_entropy_loss = torch.mean(loss.Entropy(hc_softmax_out))
                classifier_loss += paws_entropy_loss
                logs['ent_hc'] = paws_entropy_loss.item()

            if args.minent_weight != 0 and args.minent_mode != '':
                minent_idx = torch.zeros_like(agree)
                if 'ah' in args.minent_mode:
                    minent_idx = torch.logical_or(minent_idx, ah)
                if 'al' in args.minent_mode:
                    minent_idx = torch.logical_or(minent_idx, al)
                if 'dh' in args.minent_mode:
                    minent_idx = torch.logical_or(minent_idx, dh)
                if 'dl' in args.minent_mode:
                    minent_idx = torch.logical_or(minent_idx, dl)
                m_ent = torch.mean(loss.Entropy(softmax_out[minent_idx]))
                if args.minent_scheduling in ['const', 'step']:
                    m_ent = m_ent * args.minent_weight
                elif args.minent_scheduling == 'linear':
                    m_ent = m_ent * args.minent_weight * iter_num / max_iter
                if iter_num < interval_iter * args.skip_multiplier and args.minent_scheduling == 'step':
                    m_ent *= 0
                classifier_loss += m_ent
                logs['ment'] = m_ent.item() / args.minent_weight

            if args.maxent_weight != 0 and args.maxent_mode != '':
                maxent_idx = torch.zeros_like(agree)
                if 'ah' in args.maxent_mode:
                    maxent_idx = torch.logical_or(maxent_idx, ah)
                if 'al' in args.maxent_mode:
                    maxent_idx = torch.logical_or(maxent_idx, al)
                if 'dh' in args.maxent_mode:
                    maxent_idx = torch.logical_or(maxent_idx, dh)
                if 'dl' in args.maxent_mode:
                    maxent_idx = torch.logical_or(maxent_idx, dl)
                M_ent = torch.mean(loss.Entropy(softmax_out[maxent_idx]))
                if args.maxent_scheduling in ['const', 'step']:
                    M_ent = M_ent * args.maxent_weight
                elif args.maxent_scheduling == 'linear':
                    M_ent = M_ent * args.maxent_weight * iter_num / max_iter
                if iter_num < interval_iter * args.skip_multiplier and args.maxent_scheduling == 'step':
                    M_ent *= 0
                classifier_loss -= M_ent
                logs['Ment'] = M_ent.item()

            if args.paws_weight != 0:
                if args.paws_detach:
                    b1_detach = b1.clone().detach()
                    b_hc_detach = b_hc.clone().detach()
                else:
                    b1_detach = b1
                    b_hc_detach = b_hc
                paws_p2 = snn(b1_detach, b_hc_detach, labels_hc_onehot, tau=args.paws_temp)
                paws_loss = paws(paws_p1, paws_p2)
                classifier_loss += args.paws_weight * paws_loss
                logs['PAWS'] = paws_loss.item()

            if args.paws_cls_weight != 0 and args.paws_cls_mode != '':
                paws_cls_idx = torch.zeros_like(agree)
                if 'ah' in args.paws_cls_mode:
                    paws_cls_idx = torch.logical_or(paws_cls_idx, ah)
                if 'al' in args.paws_cls_mode:
                    paws_cls_idx = torch.logical_or(paws_cls_idx, al)
                if 'dh' in args.paws_cls_mode:
                    paws_cls_idx = torch.logical_or(paws_cls_idx, dh)
                if 'dl' in args.paws_cls_mode:
                    paws_cls_idx = torch.logical_or(paws_cls_idx, dl)

                if torch.count_nonzero(paws_cls_idx) > 0:
                    if args.soft_paws_cls:
                        if args.paws_cls_detach:
                            paws_p = paws_p1[paws_cls_idx].detach()
                        else:
                            paws_p = paws_p1[paws_cls_idx]
                        if args.sharpen_paws_p:
                            paws_p = sharpen(paws_p[paws_cls_idx])
                        paws_cls = F.binary_cross_entropy_with_logits(outputs_test[paws_cls_idx], paws_p, reduction='mean')
                    else:
                        _, paws_pl = torch.max(paws_p1[paws_cls_idx].detach(), dim=1)
                        paws_cls = paws_cls_fn(outputs_test[paws_cls_idx], paws_pl)
                else:
                    paws_cls = torch.tensor(0.0).cuda()

                classifier_loss += args.paws_weight * paws_cls
                logs['PAWScls'] = paws_cls.item()

            if args.paws_cr_weight != 0 and args.paws_cr_mode != '':
                # with torch.no_grad():
                #    sspl_onehot = F.one_hot(pred, num_classes=args.class_num) \
                #                       * (1 - (1 + 1 / args.class_num) * args.paws_smoothing)
                # paws_cr = dist(paws_p1, sspl_onehot)
                paws_cr_idx = torch.zeros_like(agree)
                if 'ah' in args.paws_cr_mode:
                    paws_cr_idx = torch.logical_or(paws_cr_idx, ah)
                if 'al' in args.paws_cr_mode:
                    paws_cr_idx = torch.logical_or(paws_cr_idx, al)
                if 'dh' in args.paws_cr_mode:
                    paws_cr_idx = torch.logical_or(paws_cr_idx, dh)
                if 'dl' in args.paws_cr_mode:
                    paws_cr_idx = torch.logical_or(paws_cr_idx, dl)
                if torch.count_nonzero(paws_cr_idx) > 0:
                    paws_cr = paws_cls_fn(paws_p1[paws_cr_idx], pred[paws_cr_idx])
                else:
                    paws_cr = torch.tensor(0.0).cuda()
                if args.minent_scheduling in ['const', 'step']:
                    paws_cr = paws_cr * args.paws_cr_weight
                if iter_num < interval_iter * args.skip_multiplier and args.paws_cr_scheduling == 'step':
                    paws_cr *= 0
                classifier_loss += paws_cr
                logs['PAWScr'] = paws_cr.item() / args.paws_cr_weight

            if args.to_nn_weight != 0 and args.to_nn_mode != '':
                to_nn_idx = torch.zeros_like(agree)
                if 'ah' in args.to_nn_mode:
                    to_nn_idx = torch.logical_or(to_nn_idx, ah)
                if 'al' in args.to_nn_mode:
                    to_nn_idx = torch.logical_or(to_nn_idx, al)
                if 'dh' in args.to_nn_mode:
                    to_nn_idx = torch.logical_or(to_nn_idx, dh)
                if 'dl' in args.to_nn_mode:
                    to_nn_idx = torch.logical_or(to_nn_idx, dl)

                if torch.count_nonzero(to_nn_idx) > 0:
                    to_nn_pl = torch.kthvalue(paws_p1[to_nn_idx], k=2, dim=1)[1].cuda()
                    to_nn = paws_cls_fn(outputs_test[to_nn_idx], to_nn_pl)
                else:
                    to_nn = torch.tensor(0.0).cuda()
                if args.to_nn_scheduling in ['const', 'step']:
                    to_nn = to_nn * args.to_nn_weight
                if iter_num < interval_iter * args.skip_multiplier and args.to_nn_scheduling == 'step':
                    to_nn *= 0
                classifier_loss += to_nn
                logs['toNN'] = to_nn.item() / args.to_nn_weight

        else:
            if args.cls_weight != 0:
                # with torch.no_grad():
                #    conf, _ = torch.max(F.softmax(outputs_test, dim=-1), dim=-1)
                #    conf = conf.cpu().numpy()
                conf_cls = mem_conf[tar_idx]

                classifier_loss = cls_loss_fn(outputs_test[conf_cls >= args.conf_threshold],
                                              pred[conf_cls >= args.conf_threshold])
                if args.cls2:
                    classifier_loss += cls_loss_fn(c2[conf_cls >= args.conf_threshold],
                                                   pred[conf_cls >= args.conf_threshold])
                if args.cls_scheduling in ['const', 'step']:
                    classifier_loss *= args.cls_weight
                elif args.cls_scheduling == 'linear':
                    classifier_loss *= (args.cls_weight * iter_num / max_iter)
                if iter_num < interval_iter * args.skip_multiplier and args.cls_scheduling == 'step':
                    classifier_loss *= 0
                logs['kmeans'] = classifier_loss.item() / args.cls_weight
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            if args.minent_weight != 0:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.minent_scheduling in ['const', 'step']:
                    im_loss = entropy_loss * args.minent_weight
                elif args.minent_scheduling == 'linear':
                    im_loss = entropy_loss * args.minent_weight * iter_num / max_iter
                if iter_num < interval_iter * args.skip_multiplier and args.minent_scheduling == 'step':
                    im_loss *= 0
                logs['ment'] = im_loss.item() / args.minent_weight
                classifier_loss += im_loss

            if args.memax_weight > 0:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                classifier_loss -= args.memax_weight * gentropy_loss
                logs['memax'] = gentropy_loss.item()

        if args.cr_weight != 0:
            try:
                if args.sg2_cr and not args.sg2:
                    cr_loss = dist(f_hard[conf1 >= args.cr_threshold], f_weak.detach()[conf1 >= args.cr_threshold]).mean()
                else:
                    cr_loss = dist(f_hard[conf1 >= args.cr_threshold], f_weak[conf1 >= args.cr_threshold]).mean()

                if args.cr_metric == 'cos':
                    cr_loss *= -1
            except:
                print('Error computing CR loss')
                cr_loss = torch.tensor(0.0).cuda()
            if args.cr_scheduling == 'const':
                classifier_loss += args.cr_weight * cr_loss
            elif args.cr_scheduling == 'linear':
                classifier_loss += args.cr_weight * cr_loss * iter_num / max_iter
            elif args.cr_scheduling == 'step':
                if iter_num < interval_iter * args.skip_multiplier:
                    pass
                else:
                    classifier_loss += args.cr_weight * cr_loss
            logs['cr'] = cr_loss.item()

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if hc_active and args.hc_cls_weight != 0 and args.hc_cls_mode != '':
            hc_cls_idx = torch.zeros_like(agree)
            if 'ah' in args.hc_cls_mode:
                hc_cls_idx = torch.logical_or(hc_cls_idx, ah)
            if 'al' in args.hc_cls_mode:
                hc_cls_idx = torch.logical_or(hc_cls_idx, al)
            if 'dh' in args.hc_cls_mode:
                hc_cls_idx = torch.logical_or(hc_cls_idx, dh)
            if 'dl' in args.hc_cls_mode:
                hc_cls_idx = torch.logical_or(hc_cls_idx, dl)
            if torch.count_nonzero(hc_cls_idx) > 0:
                try:
                    for p in netC.parameters():
                        p.requires_grad = True
                except:
                    for p in netC.module.parameters():
                        p.requires_grad = True
                cls_pred = netC(b2[hc_cls_idx].detach(), None)

                if args.hc_cls_target == 'paws':
                    _, cls_pl = torch.max(paws_p1[hc_cls_idx].detach(), dim=1)
                elif args.hc_cls_target == 'kmeans':
                    cls_pl = pred[hc_cls_idx]
                cls_loss = cls_loss_fn(cls_pred, cls_pl)
                if args.hc_cls_scheduling in ['const', 'step']:
                    cls_loss = cls_loss * args.hc_cls_weight
                if iter_num < interval_iter * args.skip_multiplier and args.hc_cls_scheduling == 'step':
                    cls_loss *= 0
            else:
                cls_loss = torch.tensor(0.0).cuda()
            logs['HCcls'] = cls_loss.item() / args.hc_cls_weight
        else:
            cls_loss = torch.tensor(0.0).cuda()

        try:
            c_optimizer.zero_grad()
            cls_loss.backward()
            c_optimizer.step()
        except:
            pass

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netH.eval()
            netB.eval()
            netC.eval()
            try:
                print(len(hc_set.idxs), 'samples are in HC set')
            except:
                pass

            if args.dset in ['visda-c', 'CIFAR-10-C', 'CIFAR-100-C']:
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netH, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list + '\n'
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netH, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n'

            for k, v in logs.items():
                log_str += k + ' {:.3f}'.format(v) + ', '

            logs = {}
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netH.train()
            netB.train()
            netC.train()

    if args.issave:
        if args.dataparallel:
            torch.save(netF.module.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netH.module.state_dict(), osp.join(args.output_dir, "target_H_" + args.savename + ".pt"))
            torch.save(netB.module.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.module.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        else:
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netH.state_dict(), osp.join(args.output_dir, "target_H_" + args.savename + ".pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netH, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_bn_stats(loader, netF, netB, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()

            inputs = data[0].cuda()
            feas = netB(netF(inputs))
            if start_test:
                all_fea = feas.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)

        mean = torch.mean(all_fea, dim=0)
        var = torch.var(all_fea, dim=0, unbiased=args.unbiased_var)

    return mean, var


def obtain_label(loader, netF, netH, netB, netC, args, mem_label, eval_off=False, f=None, o=None, l=None):
    if args.pl_eval_off_f and eval_off:
        netF.train()
    if args.pl_eval_off_b and eval_off:
        netB.train()

    if f is not None:
        all_fea = torch.from_numpy(f)
        all_output = torch.from_numpy(o)
        all_label = torch.from_numpy(l)
    else:
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()

                inputs = data[0]
                labels = data[1]
                tar_idx = data[2]
                inputs = inputs.cuda()
                feas = netB(netF(inputs))

                if (mem_label is not None) and (
                        args.layer in ['add_margin', 'arc_margin', 'sphere']) and args.use_margin_pl:
                    labels_forward = mem_label[tar_idx]
                else:
                    labels_forward = None

                outputs = netC(feas, labels_forward)
                raw_preds = torch.argmax(outputs, dim=1)

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    if args.pl_type == 'naive':
        all_output = nn.Softmax(dim=1)(all_output / args.pl_temperature)
        conf, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        pred_label = torch.squeeze(predict).numpy()
    elif args.pl_type in ['kmeans', 'spherical_kmeans']:
        if args.init_centroids_with_cls:
            try:
                weights = copy.deepcopy(netC.module.fc.weight.data).detach().cpu().numpy()
            except:
                weights = copy.deepcopy(netC.fc.weight.data).detach().cpu().numpy()
        else:
            weights = 'k-means++'
        all_output = nn.Softmax(dim=1)(all_output / args.pl_temperature)
        conf, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        conf_thres_idx = np.where(conf >= args.pl_threshold)
        if args.pl_type == 'spherical_kmeans':
            if args.init_centroids_with_cls:
                weights = normalize(weights)
            all_fea_norm = F.normalize(all_fea, dim=1)
            kmeans = KMeans(n_clusters=args.class_num, init=weights, max_iter=1000) \
                .fit(all_fea_norm[conf_thres_idx],
                     sample_weight=conf[conf_thres_idx] if args.weighted_samples else None)
        else:
            kmeans = KMeans(n_clusters=args.class_num, init=weights, max_iter=1000) \
                .fit(all_fea[conf_thres_idx], sample_weight=conf[conf_thres_idx] if args.weighted_samples else None)

        initc = kmeans.cluster_centers_

        cdists = cdist(all_fea, initc, metric='cosine')
        pred_label = cdists.argmin(axis=1)
        conf = softmax((1 - cdists) / args.pl_temperature, axis=1).max(axis=1)
        cls_count = np.eye(args.class_num)[pred_label].sum(axis=0)
        labelset = np.where(cls_count > args.threshold)
        labelset = labelset[0]
    else:
        if args.pl_weight_term == 'softmax':
            all_output = nn.Softmax(dim=1)(all_output / args.pl_temperature)
        elif args.pl_weight_term == 'ls':
            all_output = nn.Softmax(dim=1)(all_output / args.pl_temperature)
            pred = torch.argmax(all_output, dim=1)
            all_output = torch.ones(all_output.size(0), args.class_num) * args.pl_smooth / args.class_num
            all_output[range(all_output.size(0)), pred] = (1. - args.pl_smooth) + args.pl_smooth / args.class_num
        elif args.pl_weight_term == 'uniform':
            all_output = torch.ones(all_output.size(0), args.class_num) / args.class_num
        else:
            raise NotImplementedError
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
        unknown_weight = 1 - ent / np.log(args.class_num)
        conf, predict = torch.max(all_output, 1)
        # print('predict', predict.size())

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if args.distance == 'cosine':
            # all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > args.threshold)
        labelset = labelset[0]
        # print(labelset)

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for round in range(args.pl_rounds):
            aff = np.eye(K)[pred_label]
            c = aff.transpose().dot(all_fea)
            c = c / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, c[labelset], args.distance)
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    if args.initial_centroid == 'raw':
        centroids = initc
    elif args.initial_centroid == 'hard':
        centroids = c

    if args.momentum_cls < 1:
        m = args.momentum_cls
        device = inputs.get_device()
        if args.dataparallel:
            netC.module.fc.weight.data = m * netC.module.fc.weight.data + (1 - m) * torch.from_numpy(
                centroids).float().to(device)
        else:
            netC.fc.weight.data = m * netC.fc.weight.data + (1 - m) * torch.from_numpy(centroids).float().to(device)

    try:
        return pred_label.astype('int'), conf.cpu().numpy(), centroids, labelset, predict
    except:
        return pred_label.astype('int'), conf, centroids, labelset, predict


def calibrate_bn_stats(loader, netF, netB, args):
    if args.f_calibrate_mode in ['reset', 'offline']:
        bn_count = 0
        for layer in netF.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_count += 1
                layer.running_mean.fill_(0.)
                layer.running_var.fill_(1.)
        print('Reset' + str(bn_count) + 'BN layers in netF')
    if args.b_calibrate_mode in ['reset', 'offline'] and args.classifier == 'bn':
        bn_count = 0
        for layer in netB.modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                bn_count += 1
                layer.running_mean.fill_(0.)
                layer.running_var.fill_(1.)
        print('Reset' + str(bn_count) + 'BN layers in netB')

    if args.f_calibrate_mode in ['offline', 'mixed'] or args.b_calibrate_mode in ['offline', 'mixed']:
        netF.train()
        netB.train()
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()

                inputs = data[0]
                # labels = data[1]
                # tar_idx = data[2]
                inputs = inputs.cuda()
                feas = netB(netF(inputs))
        netF.eval()
        netB.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--hc_interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--eval_batch_mult', type=int, default=8)
    parser.add_argument('--scheduler', type=str, default='default', choices=['default', 'warmupcos'])
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--power', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--norm_layer', type=str, default='batchnorm', choices=['batchnorm', 'groupnorm'])
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['visda-c', 'office', 'office-home', 'office-caltech', 'CIFAR-10-C', 'CIFAR-100-C',
                                 'image-clef', 'modern-office'])
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--folder', type=str, default='/SSD/euntae/data/')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--nopretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--nogent', action='store_true')
    parser.add_argument('--noent', action='store_true')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_weight', type=float, default=0.3)
    parser.add_argument('--cls_smooth', type=float, default=0)
    parser.add_argument('--minent_weight', type=float, default=0.0)
    parser.add_argument('--memax_weight', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="angular",
                        choices=["linear", "wn", "angular", 'add_margin', 'arc_margin', 'sphere'])
    parser.add_argument('--classifier', type=str, default="ori", choices=["ori", "bn", "ln"])
    parser.add_argument('--classifier_bias_off', action='store_true')
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--pl_type', type=str, default="sspl", choices=["naive", "sspl", "kmeans", "spherical_kmeans"])
    parser.add_argument('--weighted_samples', type=str2bool, default=False)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--ssl_task', type=str, default='crsc',
                        choices=['none', 'simclr', 'supcon', 'ls_supcon', 'crsc', 'crs'])
    parser.add_argument('--ssl_weight', type=float, default=0.1)
    parser.add_argument('--ssl_smooth', type=float, default=0.1)
    parser.add_argument('--cr_weight', type=float, default=0.0)
    parser.add_argument('--cr_metric', type=str, default='cos', choices=['cos', 'l1', 'l2', 'bce', 'kl', 'js'])
    parser.add_argument('--cr_site', type=str, default='btn', choices=['feat', 'btn', 'cls'])
    parser.add_argument('--cr_threshold', type=float, default=0.0)
    parser.add_argument('--cr_reduction', type=str, default='sum', choices=['sum', 'mean', 'batchmean'])
    parser.add_argument('--angular_temp', type=float, default=0.1)
    parser.add_argument('--conf_threshold', type=float, default=0)
    parser.add_argument('--paws_weight', type=float, default=0.0)
    parser.add_argument('--paws_cls_weight', type=float, default=0.0)
    parser.add_argument('--paws_cr_weight', type=float, default=0.0)
    parser.add_argument('--initial_memax', type=int, default=100)
    parser.add_argument('--wa_to_memax', type=str2bool, default=True)
    parser.add_argument('--memax_print_freq', type=int, default=10)
    parser.add_argument('--paws_temperature', type=float, default=0.1)
    parser.add_argument('--paws_smoothing', type=float, default=0.1)
    parser.add_argument('--paws_detach', type=str2bool, default=True)
    parser.add_argument('--paws_batch_size', type=int, default=6)
    parser.add_argument('--sharpening_temperature', type=float, default=0.25)
    parser.add_argument('--hc_topk', type=int, default=100)
    parser.add_argument('--hc_topk_increase', type=int, default=12)
    parser.add_argument('--hc_topk_max', type=int, default=120)
    parser.add_argument('--hc_threshold', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ssl_before_btn', action='store_true')
    parser.add_argument('--no_norm_img', action='store_true')
    parser.add_argument('--norm_img_mode', type=str, choices=['whitening', 'pmone'], default='whitening')
    parser.add_argument('--norm_feat', action='store_true')
    parser.add_argument('--norm_btn', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--pl_rounds', type=int, default=1)
    parser.add_argument('--pl_weight_term', type=str, default='softmax', choices=['softmax', 'naive', 'ls', 'uniform'])
    parser.add_argument('--pl_smooth', type=float, default=0.1)
    parser.add_argument('--pl_temperature', type=float, default=1.0)
    parser.add_argument('--aug1', type=str, default='weak', choices=['none', 'weak', 'simclr', 'randaug', 'test'])
    parser.add_argument('--aug2', type=str, default='weak', choices=['none', 'weak', 'simclr', 'randaug', 'test'])
    parser.add_argument('--aug3', type=str, default='none', choices=['none', 'weak', 'simclr', 'randaug', 'test'])
    parser.add_argument('--aug_pl', type=str, default='test', choices=['none', 'weak', 'simclr', 'randaug', 'test'])
    parser.add_argument('--aug_cal', type=str, default='test', choices=['none', 'weak', 'simclr', 'randaug', 'test'])
    parser.add_argument('--aug_hc', type=str, default='weak', choices=['none', 'weak', 'simclr', 'randaug', 'test'])
    parser.add_argument('--ra_n', type=int, default=1)
    parser.add_argument('--ra_m', type=int, default=10)
    parser.add_argument('--sg2', type=str2bool, default=False)
    parser.add_argument('--sg2_cr', type=str2bool, default=False)
    parser.add_argument('--cls2', type=str2bool, default=False)
    parser.add_argument('--aug_strength', type=float, default=1.0)
    parser.add_argument('--custom_scale', default=True, type=str2bool)
    parser.add_argument('--use_rrc', default=True, type=str2bool)
    parser.add_argument('--nojitter', action='store_true')
    parser.add_argument('--nograyscale', action='store_true')
    parser.add_argument('--nogaussblur', action='store_true')
    parser.add_argument('--duplicated', default=False, type=str2bool)
    parser.add_argument('--disable_aug_for_shape', type=str2bool, default=False)

    parser.add_argument('--dropout_1', type=float, default=0)
    parser.add_argument('--dropout_2', type=float, default=0)
    parser.add_argument('--dropout_3', type=float, default=0)
    parser.add_argument('--dropout_4', type=float, default=0)

    parser.add_argument('--metric_s', type=float, default=30.0)
    parser.add_argument('--metric_m', type=float, default=0.5)
    parser.add_argument('--easy_margin', type=str2bool, default=False)

    parser.add_argument('--use_margin_forward', type=str2bool, default=False)
    parser.add_argument('--use_margin_pl', type=str2bool, default=False)

    parser.add_argument('--initial_btn_iter', type=int, default=0)
    parser.add_argument('--reset_running_stats', type=str2bool, default=False)
    parser.add_argument('--reset_bn_params', type=str2bool, default=False)
    parser.add_argument('--f_calibrate_mode', type=str, choices=['none', 'reset', 'offline', 'mixed'], default='none')
    parser.add_argument('--b_calibrate_mode', type=str, choices=['none', 'reset', 'offline', 'mixed'], default='none')
    parser.add_argument('--unbiased_var', type=str2bool, default=True)

    parser.add_argument('--use_rrc_on_wa', type=str2bool, default=False)

    parser.add_argument('--upper_bound_run', type=str2bool, default=False)

    parser.add_argument('--initial_centroid', type=str, choices=['raw', 'hard'], default='raw')
    parser.add_argument('--calibrate_cls_weights', type=str2bool, default=False)

    parser.add_argument('--pl_eval_off_f', type=str2bool, default=False)
    parser.add_argument('--pl_eval_off_b', type=str2bool, default=False)
    parser.add_argument('--eval_off_once', type=str2bool, default=True)

    parser.add_argument('--use_focal_loss', type=str2bool, default=False)
    parser.add_argument('--focal_alpha', type=float, default=0.5)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--skip_multiplier', type=float, default=1.0)

    parser.add_argument('--momentum_cls', type=float, default=1)

    parser.add_argument('--pl_threshold', type=float, default=0)

    parser.add_argument('--init_centroids_with_cls', type=str2bool, default=True)

    parser.add_argument('--minent_scheduling', type=str, choices=['const', 'linear', 'step'], default='const')
    parser.add_argument('--cr_scheduling', type=str, choices=['const', 'linear', 'step'], default='const')
    parser.add_argument('--cls_scheduling', type=str, choices=['const', 'linear', 'step'], default='const')

    parser.add_argument('--paws_pl_weight', type=float, default=0.0)
    parser.add_argument('--paws_temp', type=float, default=0.1)
    parser.add_argument('--sharpen_paws_p', type=str2bool, default=False)

    parser.add_argument('--separate_wd', type=str2bool, default=False)

    parser.add_argument('--reset_lc', type=str2bool, default=True)
    parser.add_argument('--exclude_lc', type=str2bool, default=True)
    parser.add_argument('--exclude_agreement_lc', type=str2bool, default=False)
    parser.add_argument('--hc_threshold_increase', type=float, default=0.0)
    parser.add_argument('--hc_threshold_max', type=float, default=0.9)
    parser.add_argument('--soft_paws_cls', type=str2bool, default=True)
    parser.add_argument('--paws_cls_detach', type=str2bool, default=False)
    parser.add_argument('--paws_cls_smooth', type=float, default=0.1)
    parser.add_argument('--agreement_threshold', type=float, default=0.8)
    parser.add_argument('--agreement_topk', type=int, default=12)

    parser.add_argument('--sspl_agreement', type=str2bool, default=False)
    parser.add_argument('--batch_threshold', type=float, default=0.8)
    parser.add_argument('--full_logging', type=str2bool, default=True)
    parser.add_argument('--label_mutate_p', type=float, default=0.0)
    parser.add_argument('--mutate_p_delta', type=float, default=0)
    parser.add_argument('--mutate_mode', type=str, default='dhdl')
    parser.add_argument('--mutate_to', type=str, default='random', choices=['random', 'second', 'top3'])
    parser.add_argument('--maxent_weight', type=float, default=0.0)
    parser.add_argument('--maxent_mode', type=str, default='')
    parser.add_argument('--maxent_scheduling', type=str, default='const', choices=['const', 'step'])
    parser.add_argument('--minent_mode', type=str, default='ahaldhdl')
    parser.add_argument('--memax_mode', type=str, default='ahaldhdl')
    parser.add_argument('--ce_hc', type=str2bool, default=False)
    parser.add_argument('--minent_hc', type=str2bool, default=False)
    parser.add_argument('--paws_cls_mode', type=str, default='')
    parser.add_argument('--paws_cr_mode', type=str, default='')
    parser.add_argument('--paws_cr_scheduling', type=str, default='const', choices=['const', 'step'])
    parser.add_argument('--paws_path', type=int, default=1, choices=[0, 1])

    parser.add_argument('--to_nn_weight', type=float, default=0.0)
    parser.add_argument('--to_nn_mode', type=str, default='')
    parser.add_argument('--to_nn_scheduling', type=str, default='const', choices=['const', 'step'])

    parser.add_argument('--cls_mode', type=str, default='ahaldhdl')

    parser.add_argument('--c_lr', type=float, default=1e-4)
    parser.add_argument('--c_weight_decay', type=float, default=1e-3)
    parser.add_argument('--hc_cls_weight', type=float, default=0)
    parser.add_argument('--hc_cls_mode', type=str, default='')
    parser.add_argument('--hc_cls_target', type=str, default='paws', choices=['paws', 'kmeans'])
    parser.add_argument('--hc_cls_scheduling', type=str, default='const', choices=['const', 'step'])

    parser.add_argument('--multisource', type=str2bool, default=False)

    args = parser.parse_args()

    args.pretrained = not args.nopretrained
    args.norm_img = not args.no_norm_img
    args.jitter = not args.nojitter
    args.grayscale = not args.nograyscale
    args.gaussblur = not args.nogaussblur
    args.classifier_bias = not args.classifier_bias_off
    args.ent = not args.noent
    args.gent = not args.nogent

    assert not (args.cr_weight > 0 and args.aug2 == 'none')
    assert not (args.sg2 and args.cls2)

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

    for i in range(len(names)):
        if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
            args.t = i
            args.name = names[args.t]
            args.savename = names[args.t]

            args.output_dir_src = osp.join(args.output_src, args.da, args.dset, 'source')
            args.output_dir = osp.join(args.output, args.da, args.dset, args.name)

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)

            args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()
        elif args.multisource:
            if i != args.t:
                continue
            else:
                args.t = i
                args.name = 'MS_' + names[args.t][0].upper()
                args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

                args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.t][0].upper())
                args.output_dir = osp.join(args.output, args.da, args.dset, 'MS_' + names[args.t][0].upper())

                if not osp.exists(args.output_dir):
                    os.system('mkdir -p ' + args.output_dir)
                if not osp.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                args.savename = 'par_' + str(args.cls_weight)
                if args.da == 'pda':
                    args.gent = ''
                    args.savename = 'par_' + str(args.cls_weight) + '_thr' + str(args.threshold)
                args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
                args.out_file.write(print_args(args) + '\n')
                args.out_file.flush()
        else:
            if i == args.s:
                continue
            args.t = i

            folder = args.folder
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            if args.dset == 'office-home':
                if args.disable_aug_for_shape:
                    if args.t in [1, 2]:
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

            args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
            args.output_dir = osp.join(args.output, args.da, args.dset,
                                       names[args.s][0].upper() + names[args.t][0].upper())
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)

            args.savename = 'par_' + str(args.cls_weight)
            if args.da == 'pda':
                args.gent = ''
                args.savename = 'par_' + str(args.cls_weight) + '_thr' + str(args.threshold)
            args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()

        train_target(args)
