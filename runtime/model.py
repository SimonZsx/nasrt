import torch
import utils
import numpy as np
# from apex import amp
from tqdm import tqdm
import torch.nn as nn



## import choices 
from ops import Choice_Block, Choice_Block_x


channel = [16,
           64, 64, 64, 64,
           160, 160, 160, 160,
           320, 320, 320, 320, 320, 320, 320, 320,
           640, 640, 640, 640]
last_channel = 1024


class NasModel(nn.Module):
    def __init__(self, dataset, resize, classes, layers):
        super(NasModel, self).__init__()

        ### Configs
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # Persistent Ops
        self.persist = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0], affine=False),
            nn.ReLU6(inplace=True)
        )

        # op list 
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            layer_cb = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer_cb.append(Choice_Block_x(inp, oup, stride=stride))
                else:
                    layer_cb.append(Choice_Block(inp, oup, kernel=j, stride=stride))
            self.choice_block.append(layer_cb)
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel, affine=False),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(last_channel, self.classes, bias=False)
        self._initialize_weights()

    def forward(self, x, choice=np.random.randint(4, size=20)):

        # Non block move in 
        ###
        x = self.persist(x)
        # repeat
        for i, j in enumerate(choice):
            x = self.choice_block[i][j](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)

        #Sync Move Out

        #self.move_to_cpu
        return x

    def persist_to_cuda(self, device):
        self.persist.to(device)

    def move_to_cpu(self, choice):
        for i, j in enumerate(choice):
            self.choice_block[i][j].to('cpu', non_blocking=True)

    def move_to_cuda(self, choice, device):
        for i, j in enumerate(choice):
            self.choice_block[i][j].to(device, non_blocking=True)


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
