import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
from data import *
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth, NTXentLoss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    
    if args.dset == 'CIFAR-10-C':
        try:
            dsets["source_tr"] = datasets.CIFAR10(root=args.folder+'CIFAR-10-C',
                                                  train=True,
                                                  download=False,
                                                  transform=cifar_train(args)
                                                 )
        except:
            dsets["source_tr"] = datasets.CIFAR10(root=args.folder+'CIFAR-10-C',
                                                  train=True,
                                                  download=True,
                                                  transform=cifar_train(args)
                                                 )
        dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
        dsets["source_te"] = datasets.CIFAR10(root=args.folder+'CIFAR-10-C',
                                              train=False,
                                              download=False,
                                              transform=cifar_test()
                                             )
        dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs*4, shuffle=False, num_workers=args.worker, drop_last=False)
        dsets["test"] = cifar10c_dset(args)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*4, shuffle=False, num_workers=args.worker, drop_last=False)
    
    elif args.dset == 'CIFAR-100-C':
        try:
            dsets["source_tr"] = datasets.CIFAR100(root=args.folder+'CIFAR-100-C',
                                                  train=True,
                                                  download=False,
                                                  transform=cifar_train(args)
                                                 )
        except:
            dsets["source_tr"] = datasets.CIFAR100(root=args.folder+'CIFAR-100-C',
                                                  train=True,
                                                  download=True,
                                                  transform=cifar_train(args)
                                                 )
        dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
        dsets["source_te"] = datasets.CIFAR100(root=args.folder+'CIFAR-100-C',
                                              train=False,
                                              download=False,
                                              transform=cifar_test()
                                             )
        dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs*4, shuffle=False, num_workers=args.worker, drop_last=False)
        dsets["test"] = cifar100c_dset(args)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*4, shuffle=False, num_workers=args.worker, drop_last=False)
        
    else:
        txt_src = open(args.s_dset_path).readlines()
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
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_src.append(line)
            txt_src = new_src.copy()
    
            new_tar = []
            for i in range(len(txt_test)):
                rec = txt_test[i]
                reci = rec.strip().split(' ')
                if int(reci[1]) in args.tar_classes:
                    if int(reci[1]) in args.src_classes:
                        line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                        new_tar.append(line)
                    else:
                        line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                        new_tar.append(line)
            txt_test = new_tar.copy()
    
        if args.trte == "val":
            dsize = len(txt_src)
            tr_size = int(0.9*dsize)
            # print(dsize, tr_size, dsize - tr_size)
            tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        else:
            dsize = len(txt_src)
            tr_size = int(0.9*dsize)
            _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
            tr_txt = txt_src
            
        dsets["source_tr"] = ImageList(tr_txt, transform=image_train(args))
        dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False if args.ssl_task=='none' else True)
        dsets["source_te"] = ImageList(te_txt, transform=image_test())
        dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
        dsets["test"] = ImageList(txt_test, transform=image_test())
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*4, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netH, netB, netC, flag=False):
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
            if args.bottleneck != 0:
                outputs = netC(netB(netF(inputs)))
            else:
                outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

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
            if args.bottleneck != 0:
                outputs = netC(netB(netF(inputs)))
            else:
                outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        if '26' in args.net:
            netF = network.ResCifarBase(26)
            args.bottleneck = netF.in_features // 2
        else:
            netF = network.ResBase(res_name=args.net)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)
    
    if args.ssl_before_btn:
        netH = network.ssl_head(ssl_task=args.ssl_task, feature_dim=netF.in_features, embedding_dim=args.embedding_dim)
    else:
        netH = network.ssl_head(ssl_task=args.ssl_task, feature_dim=args.bottleneck, embedding_dim=args.embedding_dim)
    if args.bottleneck != 0:
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
        netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)
    else:
        netB = None
        netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=netF.in_features)
    
    if args.dataparallel:
        netF = nn.DataParallel(netF).cuda()
        netH = nn.DataParallel(netH).cuda()
        if args.bottleneck != 0:
            netB = nn.DataParallel(netB).cuda()
        netC = nn.DataParallel(netC).cuda()
    else:
        netF.cuda()
        netH.cuda()
        if args.bottleneck != 0:
            netB.cuda()
        netC.cuda()
    
    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netH.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    if args.bottleneck != 0:
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
        
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netH.train()
    if args.bottleneck != 0:
        netB.train()
    netC.train()
    
    if args.ssl_task.lower() == 'simclr':
        ssl_loss_fn = NTXentLoss(args.batch_size, args.temperature, True).cuda()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        try:
            if inputs_source.size(0) == 1:
                continue
        except:
            if inputs_source[0].size(0) == 1:
                continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source1, inputs_source2, labels_source = inputs_source[0].cuda(), inputs_source[1].cuda(), labels_source.cuda()
        if args.bottleneck != 0:
            if args.ssl_before_btn:
                f1, f2 = netF(inputs_source1), netF(inputs_source2)
            
                z1, z2 = netH(f1, args.norm_feat), netH(f2, args.norm_feat)
            
                outputs_source = netC(netB(f1))
                
            else:
                f1, f2 = netB(netF(inputs_source1)), netB(netF(inputs_source2))
                
                z1, z2 = netH(f1, args.norm_feat), netH(f2, args.norm_feat)
                
                outputs_source = netC(f1)
        else:
            f1, f2 = netF(inputs_source1), netF(inputs_source2)
            z1, z2 = netH(f1, args.norm_feat), netH(f2, args.norm_feat)
            outputs_source = netC(f1)
            
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        ssl_loss = ssl_loss_fn(z1, z2)
        
        loss = classifier_loss + args.ssl_weight * ssl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netH.eval()
            if args.bottleneck != 0:
                netB.eval()
            netC.eval()
            if args.dset=='visda-c':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netH, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netH, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                           
                if args.dataparallel:
                    best_netF = netF.module.state_dict()
                    best_netH = netH.module.state_dict()
                    if args.bottleneck != 0:
                        best_netB = netB.module.state_dict()
                    best_netC = netC.module.state_dict()
                else:
                    best_netF = netF.state_dict()
                    best_netH = netH.state_dict()
                    if args.bottleneck != 0:
                        best_netB = netB.state_dict()
                    best_netC = netC.state_dict()

            netF.train()
            netH.train()
            if args.bottleneck != 0:
                netB.train()
            netC.train()
                
    
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netH, osp.join(args.output_dir_src, "source_H.pt"))
    if args.bottleneck != 0:
        torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netH, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
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
            netF = network.ResBase(res_name=args.net)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)
    
    if args.ssl_before_btn:
        netH = network.ssl_head(ssl_task=args.ssl_task, feature_dim=netF.in_features, embedding_dim=args.embedding_dim)
    else:
        netH = network.ssl_head(ssl_task=args.ssl_task, feature_dim=args.bottleneck, embedding_dim=args.embedding_dim)
    if args.bottleneck != 0:
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
        netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)
    else:
        netB = None
        netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=netF.in_features)
    
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_H.pt'   
    netH.load_state_dict(torch.load(args.modelpath))
    if args.bottleneck != 0:
        args.modelpath = args.output_dir_src + '/source_B.pt'   
        netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    
    if args.dataparallel:
        netF = nn.DataParallel(netF).cuda()
        netH = nn.DataParallel(netH).cuda()
        if args.bottleneck != 0:
            netB = nn.DataParallel(netB).cuda()
        netC = nn.DataParallel(netC).cuda()
    else:
        netF.cuda()
        netH.cuda()
        if args.bottleneck != 0:
            netB.cuda()
        netC.cuda()
    
    netF.eval()
    netH.eval()
    if args.bottleneck != 0:
        netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netH, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dset in ['visda-c', 'CIFAR-10-C', 'CIFAR-100-C']:
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netH, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netH, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

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
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--norm_layer', type=str, default='batchnorm', choices=['batchnorm', 'groupnorm'])
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['visda-c', 'office', 'office-home', 'office-caltech', 'CIFAR-10-C', 'CIFAR-100-C'])
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--folder', type=str, default='/SSD/euntae/data/')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--nopretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--ssl_task', type=str, default='simclr', choices=['none', 'simclr'])
    parser.add_argument('--ssl_weight', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ssl_before_btn', action='store_true')
    parser.add_argument('--no_norm_img', action='store_true')
    parser.add_argument('--no_norm_feat', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--aug_type', type=str, default='simclr', choices=['none', 'simclr', 'simsiam', 'randaug'])
    parser.add_argument('--aug_strength', type=float, default=1.0)
    parser.add_argument('--nojitter', action='store_true')
    parser.add_argument('--nograyscale', action='store_true')
    parser.add_argument('--nogaussblur', action='store_true')
    args = parser.parse_args()
    
    args.pretrained = not args.nopretrained
    args.norm_img = not args.no_norm_img
    args.norm_feat = not args.no_norm_feat
    args.jitter = not args.nojitter
    args.grayscale = not args.nograyscale
    args.gaussblur = not args.nogaussblur

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
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
        args.output_dir_src = osp.join(args.output, args.da, args.dset, 'source')
        args.name_src = 'source'
    else:
        args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
        args.name_src = names[args.s][0].upper()
        
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
            args.t = i
            args.name = names[args.t]
        else:
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
    
            folder = args.folder
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    
            if args.dset == 'office-home':
                if args.da == 'pda':
                    args.class_num = 65
                    args.src_classes = [i for i in range(65)]
                    args.tar_classes = [i for i in range(25)]
                if args.da == 'oda':
                    args.class_num = 25
                    args.src_classes = [i for i in range(25)]
                    args.tar_classes = [i for i in range(65)]

        test_target(args)