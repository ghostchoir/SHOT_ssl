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

    if k == 0:
        print('HC sampling cannot be done in a class-equal manner')
        return []
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
    outs = []
    netF.eval()
    netH.eval()
    netB.eval()
    netC.eval()

    with torch.no_grad():
        for idx, (inputs, labels, tar_idxs) in enumerate(loader):
            inputs = inputs.cuda()
            feats = netB(netF(inputs))

            outputs = netC(feats, None)
            outs += outputs.tolist()
            outputs = nn.Softmax(dim=1)(outputs / 1.0)
            conf, predict = torch.max(outputs, 1)

            confs += conf.tolist()
            preds += predict.tolist()
            features += feats.tolist()

    return np.asarray(confs), np.asarray(preds), np.asarray(features), np.asarray(outs)


class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            data_source,
            world_size,
            rank,
            batch_size=1,
            classes_per_batch=10,
            epochs=1,
            seed=0,
            unique_classes=False
    ):
        """
        ClassStratifiedSampler
        Batch-sampler that samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Sampler, samples images WITH REPLACEMENT (i.e., not epoch-based)
        :param data_source: dataset of type "TransImageNet" or "TransCIFAR10'
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        :param unique_classes: true ==> each worker samples a distinct set of classes; false ==> all workers sample the same classes
        """
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.unique_cpb = unique_classes
        self.batch_size = batch_size
        self.num_classes = len(data_source.classes)
        self.epochs = epochs
        self.outer_epoch = 0

        print(self.batch_size, self.num_classes)

        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

    def set_epoch(self, epoch):
        self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.data_source.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank * i_size:(self.rank + 1) * i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank * self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe

