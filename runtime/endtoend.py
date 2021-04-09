import os
import time
import utils
import torch
import torchvision
import torch.nn as nn
from config import get_args
from model import NasModel
from nasrt import validate
from nasrt import train

from utils import data_transforms
from torchvision import datasets
from thop import profile
from torchsummary import summary


if __name__ == '__main__':
    args = get_args()

    if args.schedule == 0:
        print("tianxiang i am a scheduler")
    
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # dataset
        assert args.dataset in ['cifar10', 'imagenet']
        train_transform, valid_transform = data_transforms(args)
        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=True,
                                                download=True, transform=train_transform)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
            valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
                                              download=True, transform=valid_transform)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        elif args.dataset == 'imagenet':
            train_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'train'), train_transform)
            val_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'valid'), valid_transform)
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True, sampler=None)
            val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)


        # one-shot
        model = NasModel(args.dataset, args.resize, args.classes, args.layers).to(device)


        # ckpt_path = os.path.join('snapshots', args.exp_name + '_ckpt_' + "{:0>4d}".format(args.epochs) + '.pth.tar')
        # print('Load checkpoint from:', ckpt_path)
        # checkpoint = torch.load(ckpt_path, map_location=device)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / 50))

        #model.to(device)


        # dataset
        # _, valid_transform = utils.data_transforms(args)
        # valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
        #                                       download=False, transform=valid_transform)
        # val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
        #                                          shuffle=False, pin_memory=True, num_workers=8)

        # random search
        start = time.time()
        best_acc = 0.0
        acc_list = list()
        best_choice = list()
        for itr in range(args.random_search): ## total num of searchs
            choices  = []
            for total in range(args.search_batch): ## per batch 
                choice = utils.random_choice(args.num_choices, args.layers)
                choices.append(choice)
            print('Choice:' + str(choice))

        # ### Schedule 

        # sched = schedule(choices, previous_choices, )

        # ## Per worker train
        # dits
        ####

        for epoch in range(50):
            graft = False
            if epoch == 49:
                graft = True
            train(args, epoch, train_loader, device, model, criterion, optimizer, scheduler, supernet=True, choice=choice, graft = graft)
        ##


        top1_acc = validate(args, itr, val_loader, device, model, criterion, supernet=True, choice=choice)
        acc_list.append([top1_acc, itr, choice])
        if best_acc < top1_acc:
            best_acc = top1_acc
            best_choice = choice
        print('acc_list:')
        for i in acc_list:
            print(i)
        print('best_acc:{} \nbest_choice:{}'.format(best_acc, best_choice))
        utils.plot_hist(acc_list, name=args.exp_name)
        utils.time_record(start)
    
    else:
        print ("tianxiang i am a worker")


    
