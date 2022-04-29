import torch
import numpy as np
import random
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset
import os
import os.path
import cv2

import torchvision


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class CIFAR10_idx(datasets.CIFAR10):
    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample)
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class CIFAR100_idx(datasets.CIFAR100):
    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample)
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


class ImageList_rotation(ImageList):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        super(ImageList_rotation, self).__init__(image_list,
                                                 labels,
                                                 transform,
                                                 target_transform,
                                                 mode)

    def __getitem__(self, index):
        path, target = self.imgs[index]

        if self.transform is not None:
            img = self.loader(path)
            img = self.transform(img)
            rotated_imgs = [
                img,
                tensor_rot_90(img),
                tensor_rot_180(img),
                tensor_rot_270(img)
            ]
            rotated_imgs = torch.stack(rotated_imgs, dim=0)
            rotation_target = torch.LongTensor([0, 1, 2, 3])
        else:
            img = cv2.imread(path)
            # print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(img.shape)
            img = kornia.image_to_tensor(img, keepdim=False)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, rotated_imgs, rotation_target


class ImageList_update(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', idxs=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        # self.imgs = imgs
        if idxs is not None:
            self.idxs = idxs
        else:
            self.idxs = list(range(len(imgs)))

        self.imgs = imgs
        self.images = []
        self.targets = []
        for i, img in enumerate(imgs):
            if i in self.idxs:
                self.images.append(img[0])
                self.targets.append(img[1])
        self.classes = np.unique(self.targets)

        self.target_indices = []
        for t in self.classes:
            indices = np.squeeze(np.argwhere(
                self.targets == t)).tolist()
            self.target_indices.append(indices)

        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.images[index], self.targets[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.images)

    def include(self, idxs):
        success = 0
        fail = 0
        for idx in idxs:
            try:
                if idx not in self.idxs:
                    self.idxs.append(idx)
                    self.images.append(self.imgs[idx][0])
                    self.targets.append(self.targets[idx][0])
                #success += 1
            except:
                pass
                #fail += 1
        #print("Include success: {:d} fail: {:d}".format(success, fail))

    def exclude(self, idxs):
        success = 0
        fail = 0
        for idx in idxs:
            try:
                if idx in self.idxs:
                    true_idx = self.idxs.index(idx)
                    del self.idxs[true_idx]
                    del self.images[true_idx]
                    del self.targets[true_idx]
                #success += 1
            except:
                pass
                #fail += 1
        #print("Exclude success: {:d} fail: {:d}".format(success, fail))


class ImageList_pl_update(Dataset):
    def __init__(self, image_list, pls, labels=None, transform=None, target_transform=None, mode='RGB', idxs=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.idxs = idxs

        self.imgs = imgs
        self.images = []
        self.targets = pls
        for idx in self.idxs:
            self.images.append(self.imgs[idx][0])
        self.classes = np.unique(self.targets)

        self.target_indices = []
        for t in self.classes:
            indices = np.squeeze(np.argwhere(
                self.targets == t)).tolist()
            self.target_indices.append(indices)

        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        true_idx = self.idxs[index]
        path, target = self.images[true_idx], self.targets[true_idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.images)

    def include(self, pls, idxs):
        success = 0
        fail = 0
        for idx in idxs:
            try:
                if idx not in self.idxs:
                    self.idxs.append(idx)
                    self.images.append(self.imgs[idx][0])
                    self.targets.append(pls[idx])
                    self.target_indices[pls[idx]].append(idx)
                #success += 1
            except:
                pass
                #fail += 1
        #print("Include success: {:d} fail: {:d}".format(success, fail))

    def exclude(self, idxs):
        success = 0
        fail = 0
        for idx in idxs:
            try:
                if idx in self.idxs:
                    true_idx = self.idxs.index(idx)
                    cls = self.targets[true_idx]
                    idx_in_cls = self.target_indices[cls].index(idx)
                    del self.idxs[true_idx]
                    del self.images[true_idx]
                    del self.targets[true_idx]
                    del self.target_indices[cls][idx_in_cls]
                #success += 1
            except:
                pass
                #fail += 1
        #print("Exclude success: {:d} fail: {:d}".format(success, fail))
