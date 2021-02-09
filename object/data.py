from torchvision import transforms, datasets
import numpy as np
import cv2
cv2.setNumThreads(0)
from data_list import CIFAR10_idx


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

    
def cifar_train(args):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    if args.aug_type == 'simclr':
        s = args.aug_strength
        color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
        trfs = [
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #GaussianBlur(kernel_size=int(0.1 * crop_size)),
            transforms.ToTensor(),
        ]
    else:
        trfs = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        
    if args.norm_img:
        trfs.append(normalize)
    
    if args.ssl_task == 'simclr':
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

def image_train(args, resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
        
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
    
    if args.ssl_task == 'simclr':
        return DuplicatedCompose(trfs)
    else:
        return transforms.Compose(trfs)


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
      ])