import copy
import math
import statistics

import torch
import torchvision
from torch import nn, autograd
# from torch.distributed import tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
import numpy as np
import random
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from models.rdp_accountant import compute_rdp, get_privacy_spent
from itertools import cycle

class DatasetSplit(Dataset):

    """
    Class DatasetSplit - To get datasamples corresponding to the indices of samples a particular client has from the actual complete dataset

    """

    def __init__(self, dataset, idxs):

        """

        Constructor Function

        Parameters:

            dataset: The complete dataset

            idxs : List of indices of complete dataset that is there in a particular client

        """
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):

        """

        returns length of local dataset

        """

        return len(self.idxs)

    def __getitem__(self, item):

        """
        Gets individual samples from complete dataset

        returns image and its label

        """
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

# function to train a client
# 参数上加噪
# def train_client_w(args, dp_epsilon, dataset,train_idx,net):
def train_client(args, dataset, train_idx, net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    for iter in range(args.local_ep):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def grad_clip(net, clip):
    # 逐样本裁剪
    grads_list = []
    for param in net.parameters():
        if param.grad is not None:
            grads_list.append(param.grad.clone())
        else:
            continue
    # 将梯度列表展平为一维张量
    flatten_grads = torch.cat([grad.flatten() for grad in grads_list])
    grads_clipped = flatten_grads / max(1.0, float(torch.norm(flatten_grads, p=2)) / clip)

    start = 0
    for param in net.parameters():
        size = param.numel()
        if param.grad is None:
            continue
        else:
            param.grad.copy_(grads_clipped[start:start + size].view_as(param.grad))
        start += size

# =================================================用户级======================================================

# def train_client_w(args, clip, base_layers, dataset, train_idx, net):
#
#     loss_func = nn.CrossEntropyLoss()
#     train_idx = list(train_idx)
#     ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
#     net.train()
#
#     # train and update
#     optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
#     epoch_loss = []
#
#     if args.model == 'ResNet':
#         if base_layers == 216:
#             tempt = 108
#         elif base_layers == 204:
#             tempt = 102
#         elif base_layers == 192:
#             tempt = 96
#         elif base_layers == 174:
#             tempt = 87
#     elif args.model == 'MobileNet':
#         if base_layers == 162:
#             tempt = 81
#         elif base_layers == 150:
#             tempt = 75
#         elif base_layers == 138:
#             tempt = 69
#     elif args.model == 'ResNet50':
#         if base_layers == 318:
#             tempt = 159
#         elif base_layers == 300:
#             tempt = 150
#         elif base_layers == 282:
#             tempt = 141
#         elif base_layers == 258:
#             tempt = 129
#     elif args.model == 'cnn':
#         if args.dataset == 'cifar':
#             if base_layers == 8:
#                 tempt = 8
#             elif base_layers == 6:
#                 tempt = 6
#         elif args.dataset == 'mnist':
#             if base_layers == 6:
#                 tempt = 6
#             elif base_layers == 4:
#                 tempt = 4
#     elif args.model == 'ResNet18':
#         if base_layers == 120:
#             tempt = 60
#         elif base_layers == 108:
#             tempt = 54
#         elif base_layers == 90:
#             tempt = 45
#         elif base_layers == 78:
#             tempt = 39
#
#     start_net = copy.deepcopy(net)
#
#     for iter in range(args.local_ep):
#         batch_loss = []
#
#         for batch_idx, (images, labels) in enumerate(ldr_train):
#             images, labels = images.to(args.device), labels.to(args.device)
#             optimizer.zero_grad()
#             log_probs = net(images)
#             loss = loss_func(log_probs, labels)
#             loss.backward()
#             optimizer.step()
#
#             batch_loss.append(loss.item())
#         epoch_loss.append(sum(batch_loss) / len(batch_loss))
#
#     delta_net = copy.deepcopy(net)
#     w_start = start_net.state_dict()
#     w_delta = delta_net.state_dict()
#
#     for i in w_delta.keys():
#         w_delta[i] -= w_start[i]
#     delta_net.load_state_dict(w_delta)
#
#     # 打印增量的L2范数
#     norm = 0.0
#     t = 0
#     for name in w_delta.keys():
#         # if t > tempt:
#         #     break
#         if (
#                 "running_mean" not in name
#                 and "running_var" not in name
#                 and "num_batches_tracked" not in name
#         ):
#             norm += pow(w_delta[name].float().norm(2), 2)
#             t += 1
#     total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
#     # print("total_norm", total_norm)
#     avg_l2_norm = total_norm / t
#
#     # with torch.no_grad():
#     #     delta_net = clip_parameters(clip, delta_net, tempt, total_norm)
#     # delta_net = add_noise(args, clip, delta_net, tempt)
#
#     w_delta = delta_net.state_dict()
#     for i in w_start.keys():
#         w_start[i] += w_delta[i].to(w_start[i].dtype)
#
#     return w_start, sum(epoch_loss) / len(epoch_loss), avg_l2_norm, total_norm

def train_client_w1(args, clip, base_layers, dataset, train_idx, net):

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    eps = args.dp_epsilon

    if args.model == 'ResNet':
        if base_layers == 216:
            tempt = 108
        elif base_layers == 204:
            tempt = 102
        elif base_layers == 192:
            tempt = 96
        elif base_layers == 174:
            tempt = 87
    elif args.model == 'MobileNet':
        if base_layers == 162:
            tempt = 81
        elif base_layers == 150:
            tempt = 75
        elif base_layers == 138:
            tempt = 69
    elif args.model == 'ResNet50':
        if base_layers == 318:
            tempt = 159
        elif base_layers == 300:
            tempt = 150
        elif base_layers == 282:
            tempt = 141
        elif base_layers == 258:
            tempt = 129
    elif args.model == 'cnn':
        if args.dataset == 'cifar':
            if base_layers == 8:
                tempt = 8
            elif base_layers == 6:
                tempt = 6
        elif args.dataset == 'mnist':
            if base_layers == 6:
                tempt = 6
            elif base_layers == 4:
                tempt = 4
    elif args.model == 'ResNet18':
        if base_layers == 120:
            tempt = 60
        elif base_layers == 108:
            tempt = 54
        elif base_layers == 90:
            tempt = 45
        elif base_layers == 78:
            tempt = 39

    start_net = copy.deepcopy(net)

    clip = clip * torch.exp(-decay_rate * (iter))

    for iter in range(args.local_ep):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    delta_net = copy.deepcopy(net)
    w_start = start_net.state_dict()
    w_delta = delta_net.state_dict()

    for i in w_delta.keys():
        w_delta[i] -= w_start[i]

    # 打印增量的L2范数
    norm = 0.0
    t = 0
    for name in w_delta.keys():
        if (
                "running_mean" not in name
                and "running_var" not in name
                and "num_batches_tracked" not in name
        ):
            norm += pow(w_delta[name].float().norm(2), 2)
            t += 1
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)

    avg_l2_norm = total_norm / t
    # print(f"L2 Norm of the model: {total_norm[0]}")

    w_delta_vector = torch.cat([param.view(-1).abs() for param in w_delta.values()])
    count = torch.sum(w_delta_vector < clip).item()
    total = len(w_delta_vector)
    b = count / total  # 计算的是b

    delta_net.load_state_dict(w_delta)
    with torch.no_grad():
        delta_net = clip_parameters(clip, delta_net, total_norm)
    delta_net = add_noise(args, clip, eps, delta_net, tempt)

    w_delta = delta_net.state_dict()
    for i in w_start.keys():
        w_start[i] += w_delta[i].to(w_start[i].dtype)

    return w_start, sum(epoch_loss) / len(epoch_loss), avg_l2_norm, b, total_norm

def train_client_w2(args, clip, base_layers, dataset, train_idx, net):

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    eps = args.dp_epsilon

    if args.model == 'ResNet':
        if base_layers == 216:
            tempt = 108
        elif base_layers == 204:
            tempt = 102
        elif base_layers == 192:
            tempt = 96
        elif base_layers == 174:
            tempt = 87
    elif args.model == 'MobileNet':
        if base_layers == 162:
            tempt = 81
        elif base_layers == 150:
            tempt = 75
        elif base_layers == 138:
            tempt = 69
    elif args.model == 'ResNet50':
        if base_layers == 318:
            tempt = 159
        elif base_layers == 300:
            tempt = 150
        elif base_layers == 282:
            tempt = 141
        elif base_layers == 258:
            tempt = 129
    elif args.model == 'cnn':
        if args.dataset == 'cifar':
            if base_layers == 8:
                tempt = 8
            elif base_layers == 6:
                tempt = 6
        elif args.dataset == 'mnist':
            if base_layers == 6:
                tempt = 6
            elif base_layers == 4:
                tempt = 4
    elif args.model == 'ResNet18':
        if base_layers == 120:
            tempt = 60
        elif base_layers == 108:
            tempt = 54
        elif base_layers == 90:
            tempt = 45
        elif base_layers == 78:
            tempt = 39

    start_net = copy.deepcopy(net)

    for iter in range(args.local_ep):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    delta_net = copy.deepcopy(net)
    w_start = start_net.state_dict()
    w_delta = delta_net.state_dict()

    for i in w_delta.keys():
        w_delta[i] -= w_start[i]

    # 打印增量的L2范数
    norm = 0.0
    t = 0
    for name in w_delta.keys():
        if (
                "running_mean" not in name
                and "running_var" not in name
                and "num_batches_tracked" not in name
        ):
            norm += pow(w_delta[name].float().norm(2), 2)
            t += 1
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)

    delta_net.load_state_dict(w_delta)
    with torch.no_grad():
        delta_net = clip_parameters(clip, delta_net, total_norm)
    delta_net = add_noise(args, clip, eps, delta_net, tempt)

    w_delta = delta_net.state_dict()
    for i in w_start.keys():
        w_start[i] += w_delta[i].to(w_start[i].dtype)

    return w_start, sum(epoch_loss) / len(epoch_loss), total_norm

def train_client_w(args, clip, base_layers, dataset, train_idx, net):

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    start_net = copy.deepcopy(net)

    for iter in range(args.local_ep):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    delta_net = copy.deepcopy(net)
    w_start = start_net.state_dict()
    w_delta = delta_net.state_dict()

    for i in w_delta.keys():
        w_delta[i] -= w_start[i]

    # 打印增量的L2范数
    norm = 0.0
    t = 0
    for name in w_delta.keys():
        # if t > tempt:
        #     break
        if (
                "running_mean" not in name
                and "running_var" not in name
                and "num_batches_tracked" not in name
        ):
            norm += pow(w_delta[name].float().norm(2), 2)
            t += 1
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)

    return w_start, sum(epoch_loss) / len(epoch_loss), total_norm, w_delta

def client_w(args, avg_norm, w_start, w_delta, clip, t_e, base_layers, window, net):
    total_norm = window[-1]  # 获取最后一个值

    min_norm = min(avg_norm)  # 找到列表中的最小值
    min_v = min_norm.item()
    a = 0.9
    # 固定隐私预算
    # epsilon = args.dp_epsilon
    # 动态隐私预算
    if clip > total_norm:
        eps = args.dp_epsilon * a + (1-a) * args.dp_epsilon * (clip - total_norm) / (clip - min_v)
        epsilon = eps.item()
    else:
        eps = args.dp_epsilon * a
        epsilon = eps

    if args.model == 'ResNet':
        if base_layers == 216:
            tempt = 108
        elif base_layers == 204:
            tempt = 102
        elif base_layers == 192:
            tempt = 96
        elif base_layers == 174:
            tempt = 87
    elif args.model == 'MobileNet':
        if base_layers == 162:
            tempt = 81
        elif base_layers == 150:
            tempt = 75
        elif base_layers == 138:
            tempt = 69
    elif args.model == 'ResNet50':
        if base_layers == 318:
            tempt = 159
        elif base_layers == 300:
            tempt = 150
        elif base_layers == 282:
            tempt = 141
        elif base_layers == 258:
            tempt = 129
    elif args.model == 'cnn':
        if args.dataset == 'cifar':
            if base_layers == 8:
                tempt = 8
            elif base_layers == 6:
                tempt = 6
        elif args.dataset == 'mnist':
            if base_layers == 6:
                tempt = 6
            elif base_layers == 4:
                tempt = 4
    elif args.model == 'ResNet18':
        if base_layers == 120:
            tempt = 60
        elif base_layers == 108:
            tempt = 54
        elif base_layers == 90:
            tempt = 45
        elif base_layers == 78:
            tempt = 39
    # RDP
    # sampling_prob = 1
    # steps = args.local_ep
    # z = np.sqrt(2 * np.log(1.25 / args.delta)) / epsilon
    # sigma, eps = get_sigma(sampling_prob, steps, epsilon, args.delta, z, rgp=True)
    # s = 2 * clip * args.lr
    # noise_scale = s * sigma
    # t_e += eps

    print("epsilon:", epsilon)
    t_e += epsilon

    delta_net = copy.deepcopy(net)
    delta_net.load_state_dict(w_delta)

    with torch.no_grad():
        delta_net = clip_parameters(clip, delta_net, total_norm)
    # 原加噪
    delta_net = add_noise(args, clip, epsilon, delta_net, tempt)

    # RDP
    # count = 0
    # with torch.no_grad():
    #     for k, v in delta_net.named_parameters():
    #         if count > tempt:
    #             break
    #         v += torch.from_numpy(np.random.normal(0, noise_scale, size=v.shape)).to(args.device)
    #         count += 1

    w_delta = delta_net.state_dict()
    for i in w_start.keys():
        w_start[i] += w_delta[i].to(w_start[i].dtype)

    return w_start, t_e


def clip_parameters(clip, net, total_norm):
    L2 = total_norm[0]
    count = 0
    for k, v in net.named_parameters():
        # if count > tempt:
        #     break
        v /= max(1, L2 / clip)
        # count += 1
    return net
#
def add_noise(args, clip, epsilon, net, tempt):
    sensitivity = cal_sensitivity_up(args.lr, clip)
    count = 0
    with torch.no_grad():
        for k, v in net.named_parameters():
            if count > tempt:
                break
            noise = Gaussian_Simple(epsilon=epsilon, delta=args.delta, sensitivity=sensitivity, size=v.shape)
            noise = torch.from_numpy(noise).to(args.device)
            v += noise
            count += 1
    return net

def cal_sensitivity_up(lr, clip):
    return 2 * lr * clip

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    # sampling_prob = 1
    # steps = 1
    # delta = 1e-5
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    # noise_scale, eps = get_sigma(sampling_prob, steps, epsilon, delta, noise_scale, rgp=True)

    return np.random.normal(0, noise_scale, size=size)

def calculate_ave(lst):
    lst.sort()  # 对列表进行排序
    lst = lst[:]  # 去除最大的两个数和最小的两个数
    # 计算剩下数的平均值
    average = sum(lst) / len(lst)
    return average

def calculate_mean_std(vector):
    # 计算滑动窗口中所有值的平均值
    mean_value = statistics.mean(vector)
    # 计算滑动窗口中所有值的标准差
    std_dev = np.std(vector)

    return mean_value, std_dev

def finetune_client(args,dataset,train_idx,net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(1):
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss) / len(epoch_loss)


# function to test a client
def test_client(args,dataset,test_idx,net):

    '''

    Test the performance of the client models on their datasets

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : The data on which we want the performance of the model to be evaluated

        args (dictionary) : The list of arguments defined by the user

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''
    
    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.local_bs)  
    net.eval()
    #print (test_data)
    test_loss = 0
    correct = 0
    
    l = len(data_loader)
    
    with torch.no_grad():
                
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss

def Distilling(args, net_glob, Ensemble_model, iter):
    if args.pb_dataset == "pbMnist":
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载Mnist数据集
        trainset = torchvision.datasets.MNIST(root='..\data\pbMNIST', train=False, download=True, transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1000)
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1000)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    elif args.pb_dataset == "pbCifar10":
        # 加载CIFAR10数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        # 加载CIFAR-10数据集
        trainset = torchvision.datasets.CIFAR10(root='..\data\pbCifar10', train=False, download=True, transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1280)
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader1 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader1

    elif args.pb_dataset == "pbCifar100":
        # 加载CIFAR100数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        # 加载CIFAR-100数据集
        trainset = torchvision.datasets.CIFAR100(root='..\data\pbCifar100', train=False, download=True,transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1500) #500/1280/3200
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader2

    elif  args.pb_dataset == "SVHN":
        # 加载SVHN数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 加载SVHN数据集
        trainset = torchvision.datasets.SVHN(root='..\data\SVHN', split ="test", download=True,transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1280)  # 500/3200
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader2

    elif args.pb_dataset == "Usps":
        # 定义数据变换，将数据转换为Tensor并将像素值标准化为[0, 1]之间的范围
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 导入USPS数据集
        trainset = torchvision.datasets.USPS(root='..\\data\\Usps', train=True, download=True, transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1280)  # 500/3200
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader3 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader3

    elif args.pb_dataset == "FMnist":
        # 加载FashionMNIST数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 加载CIFAR-10数据集
        trainset = torchvision.datasets.FashionMNIST(root='..\data\FashionMNIST', train=False, download=True, transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 500)
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader4 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader1 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader4

    else:
        print("没有这个公共数据集")

    epochs = 2
    temp = 4
    alpha = 0.5
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(net_glob.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for i, (x, y)  in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to('cuda')
            else:
                x = x.to('cuda')
            y = y.to('cuda')
            # 教师模型预测
            with torch.no_grad():
                teacher_preds = Ensemble_model(x)
            # 学生模型预测
            student_preds = net_glob(x)
            student_loss = hard_loss(student_preds, y)
            # 计算蒸馏后的预测结果及soft_loss
            distillation_loss = soft_loss(
                F.softmax(student_preds / temp, dim=1),
                F.softmax(teacher_preds / temp, dim=1)
            ).to('cuda')
            # 将 hard_loss 和 soft_loss 加权求和
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
            # 反向传播,优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def resize_image(image):
    # 将PIL Image对象调整为28x28大小
    resized_image = image.resize((28, 28))
    return resized_image

def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average

def corresponding_base_layer(model, data_set):
    if model == 'ResNet':
        # values = [216, 204, 192, 174]
        values = [174, 192, 204, 216]
    elif model == 'ResNet18':
        values = [78, 90, 108, 120]
    elif model == 'ResNet50':
        values = [258, 282, 300, 318]
    elif model == 'cnn':
        if data_set == 'cifar':
            values = [6, 8]
        elif data_set == 'mnist':
            values = [4, 6]
    return values

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    previous_eps=eps
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if (rgp):
            rdp = compute_rdp(q, cur_sigma, steps,
                              orders) * 2  ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)  # 根据目标delta值计算对应的epsilon值，并获取最优的阶数。
        if (cur_eps < eps and cur_sigma > interval): # 判断当前epsilon值是否小于目标epsilon，并且当前的sigma值是否大于间隔值
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma, interval=1., rgp=True):
    cur_sigma = init_sigma

    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps
