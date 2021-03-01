from torchvision import transforms, datasets
import numpy as np
import cv2
cv2.setNumThreads(0)
from data_list import CIFAR10_idx, CIFAR100_idx


corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2

class DualCompose(object):
    def __init__(self, trfs1, trfs2):
        self.trfs1 = trfs1
        self.trfs2 = trfs2

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t1, t2 in zip(self.trfs1, self.trfs2):
            img1 = t1(img1)
            img2 = t2(img2)
        return img1, img2

class TripletCompose(object):
    def __init__(self, trfs1, trfs2, trfs3):
        self.trfs1 = trfs1
        self.trfs2 = trfs2
        self.trfs3 = trfs3

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        for t1, t2, t3 in zip(self.trfs1, self.trfs2, self.trfs3):
            img1 = t1(img1)
            img2 = t2(img2)
            img3 = t3(img3)
        return img1, img2, img3

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0, p=0.5):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        self.prob = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.prob:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

# def get_train_aug(args, resize_size=256, img_size=224):
#     if args.aug_type == 'simclr':
#         s = args.aug_strength
#         color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
#         if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
#             trfs = [transforms.RandomCrop(img_size, padding=4)]
#         else:
#             trfs = [transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0))]
#         trfs.append(transforms.RandomHorizontalFlip())
#
#         if args.jitter:
#             trfs.append(transforms.RandomApply([color_jitter], p=0.8))
#         if args.grayscale:
#             trfs.append(transforms.RandomGrayscale(p=0.2))
#         if args.gaussblur:
#             trfs.append(GaussianBlur(kernel_size=int(0.1 * img_size)))
#
#     else:
#         if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
#             trfs = [transforms.RandomCrop(32, padding=4)]
#         else:
#             trfs = [transforms.Resize((resize_size, resize_size)),
#                     transforms.RandomCrop(img_size)]
#         trfs.append(transforms.RandomHorizontalFlip())
#
#     trfs.append(transforms.ToTensor())
#
#     if args.norm_img:
#         if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
#             normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         else:
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#         trfs.append(normalize)
#
#     return trfs
#
# def get_test_aug(args, resize_size=256, img_size=224):
#     if args.dset in ['CIFAR-10-C', 'CIFAR-100-C']:
#         trfs = [transforms.ToTensor()]
#         if args.norm_img:
#             trfs.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
#     else:
#         trfs = [transforms.Resize((resize_size, resize_size)),
#                 transforms.CenterCrop(img_size),
#                 transforms.ToTensor()]
#         if args.norm_img:
#             trfs.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#
#     return trfs
    
def cifar_train(args):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if args.cr_weight > 0:
        s = args.aug_strength
        color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)

        trfs1 = [
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
        ]

        trfs2 = trfs1

        trfs3 = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        if args.norm_img:
            trfs1.append(normalize)
            trfs3.append(normalize)


        return TripletCompose(trfs1, trfs2, trfs3)

    if args.aug_type == 'simclr':
        s = args.aug_strength
        color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
        trfs = [
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
        ]
        if args.jitter:
            trfs.append(transforms.RandomApply([color_jitter], p=0.8))
        if args.grayscale:
            trfs.append(transforms.RandomGrayscale(p=0.2))
        if args.gaussblur:
            trfs.append(GaussianBlur(kernel_size=int(0.1 * 32)))
            
        trfs.append(transforms.ToTensor())
    else:
        trfs = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        
    if args.norm_img:
        trfs.append(normalize)
    
    if args.ssl_task in ['simclr', 'supcon']:
        return DuplicatedCompose(trfs)
    else:
        return transforms.Compose(trfs)

def cifar_test():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

def cifar10c_dset(args):
    
    dset = datasets.CIFAR10(root=args.folder+'CIFAR-10-C',
                            train=False,
                            download=False,
                            transform=cifar_test()
                           )
    
    tsize = 10000
    tset_raw = np.load(args.folder + 'CIFAR-10-C/%s.npy' %(corruptions[args.t]))
    tset_raw = tset_raw[(args.level-1)*tsize: args.level*tsize]
    
    dset.data = tset_raw
    
    return dset

def cifar10c_dset_idx(args):
    
    dset = CIFAR10_idx(root=args.folder+'CIFAR-10-C',
                            train=False,
                            download=False,
                            transform=cifar_test()
                           )
    
    tsize = 10000
    tset_raw = np.load(args.folder + 'CIFAR-10-C/%s.npy' %(corruptions[args.t]))
    tset_raw = tset_raw[(args.level-1)*tsize: args.level*tsize]
    
    dset.data = tset_raw
    
    return dset

def cifar100c_dset(args):
    
    dset = datasets.CIFAR100(root=args.folder+'CIFAR-100-C',
                            train=False,
                            download=False,
                            transform=cifar_test()
                           )
    
    tsize = 10000
    tset_raw = np.load(args.folder + 'CIFAR-100-C/%s.npy' %(corruptions[args.t]))
    tset_raw = tset_raw[(args.level-1)*tsize: args.level*tsize]
    
    dset.data = tset_raw
    
    return dset

def cifar100c_dset_idx(args):
    
    dset = CIFAR100_idx(root=args.folder+'CIFAR-100-C',
                            train=False,
                            download=False,
                            transform=cifar_test()
                           )
    
    tsize = 10000
    tset_raw = np.load(args.folder + 'CIFAR-100-C/%s.npy' %(corruptions[args.t]))
    tset_raw = tset_raw[(args.level-1)*tsize: args.level*tsize]
    
    dset.data = tset_raw
    
    return dset

def image_train(args, resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    if args.cr_weight > 0:
        s = args.aug_strength
        color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)

        trfs1 = [
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * crop_size)),
            transforms.ToTensor(),
        ]

        trfs2 = trfs1

        trfs3 = [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        if args.norm_img:
            trfs1.append(normalize)
            trfs3.append(normalize)

        return TripletCompose(trfs1, trfs2, trfs3)

    if args.aug_type == 'simclr':
        s = args.aug_strength
        color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
        trfs = [
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * crop_size)),
            transforms.ToTensor(),
        ]
    else:
        trfs = [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        
    if args.norm_img:
        trfs.append(normalize)
    
    if args.ssl_task in ['simclr', 'supcon']:
        return DuplicatedCompose(trfs)
    else:
        return transforms.Compose(trfs)


def image_test(args, resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    trfs = [transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),]

    if args.norm_img:
        trfs.append(normalize)

    return transforms.Compose(trfs)