import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models.resnet import conv3x3, conv1x1
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name, pretrained=True):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=False)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name, pretrained=True):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori", norm_btn=False):
        super(feat_bootleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

        if self.type == "ori":
            self.norm = None
        elif self.type == "bn":
            self.norm = nn.BatchNorm1d(bottleneck_dim)
        elif self.type == "ln":
            self.norm = nn.LayerNorm(bottleneck_dim)

        self.norm_btn = norm_btn

    def forward(self, x):
        x = self.bottleneck(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.norm_btn:
            x = F.normalize(x, dim=-1)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear", bias=True):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num, bias=bias), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=bias)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class ssl_head(nn.Module):
    def __init__(self, ssl_task, feature_dim, embedding_dim=128):
        super(ssl_head, self).__init__()
        if ssl_task.lower() in ['simclr', 'supcon']:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, embedding_dim)
            )
        
        
    def forward(self, x, norm_feat=False):
        if norm_feat:
            x = F.normalize(x, dim=1)
        
        x = self.head(x)
        
        x = F.normalize(x, dim=1)
        
        return x


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y
    
    
class BasicBlockCifar(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlockCifar, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x 
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)    
    
class ResCifarBase(nn.Module):
    def __init__(self, depth, width=1, classes=10, channels=3, norm_layer=nn.BatchNorm2d):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResCifarBase, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.in_features = 64 * width

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlockCifar(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlockCifar(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x