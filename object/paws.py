import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import loss


def get_topk_conf_indices(confs, preds, n_classes=12, k=100, threshold=0.5):
    idxs = np.arange(len(preds))

    c_counts = [0] * n_classes
    c_thres_counts = [0] * n_classes
    c_confs = []
    c_idxs = []

    for c in range(n_classes):
        idx = preds == c
        c_thres_idxs = confs[idx] >= threshold
        c_counts[c] = np.sum(idx)
        c_thres_counts[c] = np.sum(c_thres_idxs)
        c_confs.append(confs[idx])
        c_idxs.append(idxs[idx])

    k = min(k, min(c_counts))
    thres_k = min(k, min(c_thres_counts))

    k = min(k, thres_k)

    assert k != 0, 'HC sampling cannot be done in a class-equal manner'
    print("Filtering top {:d} elements per class".format(k))

    hc_idxs = []

    for c in range(n_classes):
        c_conf = c_confs[c]
        c_idx = c_idxs[c]

        topk_idxs = np.argpartition(c_conf, -k)[-k:]

        c_idx = c_idx[topk_idxs]
        hc_idxs += c_idx.tolist()

    return hc_idxs


def sharpen(p, T=0.25):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p


def ce(p, q, eps=1e-8):
    return -sharpen(q) * torch.log(p + eps)


def paws(p1, p2):
    loss_1 = ce(p1, p2)
    loss_2 = ce(p2, p1)

    loss = torch.mean(torch.sum(loss_1 + loss_2, dim=1)) / 2

    return loss


def snn(query, supports, labels, tau=0.1):
    """ Soft Nearest Neighbours similarity classifier """
    # Step 1: normalize embeddings
    query = F.normalize(query)
    supports = F.normalize(supports)

    # Step 2: gather embeddings from all workers
    #supports = AllGather.apply(supports)

    # Step 3: compute similarlity between local embeddings
    return F.softmax(query @ supports.T / tau, dim=1) @ labels


def maxent_step(inputs, netF, netH, netB, netC, optim, epsilon=1e-8):
    netF.train()
    netH.train()
    netB.train()

    optim.zero_grad()

    c3 = netC(netB(netF(inputs)), None)

    softmax_out = nn.Softmax(dim=1)(c3)

    msoftmax = softmax_out.mean(dim=0)

    gentropy = -torch.sum(-msoftmax * torch.log(msoftmax + epsilon))

    with torch.no_grad():
        entropy = torch.mean(loss.Entropy(softmax_out))

    gentropy.backward()
    optim.step()

    return entropy.item(), gentropy.item()


def get_outputs(loader, netF, netH, netB, netC):
    confs = []
    preds = []
    features = []
    netF.eval()
    netH.eval()
    netB.eval()
    netC.eval()

    with torch.no_grad():
        for idx, (inputs, labels, tar_idxs) in enumerate(loader):
            inputs = inputs.cuda()
            feats = netB(netF(inputs))

            outputs = netC(feats, None)
            outputs = nn.Softmax(dim=1)(outputs / 1.0)
            conf, predict = torch.max(outputs, 1)

            confs += conf.tolist()
            preds += predict.tolist()
            features += feats.tolist()

    return np.asarray(confs), np.asarray(preds), np.asarray(features)
