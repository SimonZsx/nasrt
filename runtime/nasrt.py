import torch
import utils
import numpy as np
# from apex import amp
from tqdm import tqdm

import torch.multiprocessing as mp


#### set start method to spawn ####

mp.set_start_method('spawn')

activation_out = {}
activation_in = {}
def get_activation(name):
    def hook(model, input_, output_):
        #print("input size and type"+str(output_))
        activation_in[name] = input_[0].detach()
        activation_out[name] = output_.detach()
    return hook

def get_activation_input_by_index(index, q):
    def hook(model, input_, _):
        activation_in[index] = input_[0].detach()
        q.put([index, input_[0]])
    return hook

def get_activation_output_by_index(index, q):
    def hook(model, _, output_):
        activation_out[name] = output_.detach()
        q.put([index, onput_])
    
    return hook

def save_io(input_q, output_q):

    while True:


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler, supernet, choice=None, graft=False):
    model.train()


    ##### queue #########

    input_q = mp.Queue()
    output_q = mp.Queue()

    #####################

    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    train_data = tqdm(train_data)
    eps = args.epochs

    if supernet == 'supernet':
        if choice is not None:
            eps = 50

    train_data.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, eps, 'lr:', scheduler.get_lr()[0]))

    if graft:
        model.hook_(i1, j1, get_activation_input_by_index(i))
        model.hook_(i2, j2, get_activation_output_by_index(i))
    
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if supernet == 'supernet':
            if choice is None:
                choice = utils.random_choice(args.num_choices, args.layers)
            outputs = model(inputs, choice)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # if args.dataset == 'cifar10':
        loss.backward()
        # elif args.dataset == 'imagenet':
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        optimizer.step()

        #model.move_to_cpu(choice)
        
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (step + 1)), 'train_acc': '%.6f' % top1.avg}
        train_data.set_postfix(log=postfix)

#def extractgraft(model, root_choice, graft_choice):


def traingraft(args, epoch, train_data, device, rootmodel, graftmodel, criterion, optimizer, scheduler, supernet, choice=None):


    rootmodel.val()
    graftmodel.train()

    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    train_data = tqdm(train_data)
    eps = args.epochs

    if supernet == 'supernet':
        if choice is not None:
            eps = 50

    train_data.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, eps, 'lr:', scheduler.get_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if supernet == 'supernet':
            if choice is None:
                choice = utils.random_choice(args.num_choices, args.layers)
            outputs = model(inputs, choice)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # if args.dataset == 'cifar10':
        loss.backward()
        # elif args.dataset == 'imagenet':
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        optimizer.step()

        #model.move_to_cpu(choice)
        
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (step + 1)), 'train_acc': '%.6f' % top1.avg}
        train_data.set_postfix(log=postfix)



def validate(args, epoch, val_data, device, model, criterion, supernet=False, choice=None):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            if supernet:
                if choice == None:
                    choice = utils.random_choice(args.num_choices, args.layers)
                outputs = model(inputs, choice)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
        print('[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f'
              % (epoch + 1, val_loss / (step + 1), val_top1.avg))
        return val_top1.avg
