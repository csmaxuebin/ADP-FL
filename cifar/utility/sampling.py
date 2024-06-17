import numpy as np
from torchvision import datasets, transforms
import random

# two functions for each type of dataset - one to divide data in iid manner and one in non-iid manner

# ================================================================病态非独立同分布==================================================================

def pathological_non_iid_split(args,dataset,num_users):
    data_idcs = list(range(len(dataset)))
    label2index = {k: [] for k in range(args.num_classes)}
    for idx in data_idcs:
        _, label = dataset[idx]
        label2index[label].append(idx)
    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]
    n_clients= args.num_users
    n_shards = n_clients * args.overlapping_classes
    # 一共分成n_shards个独立同分布的shards
    shards = iid_divide(sorted_idcs, n_shards)
    np.random.shuffle(shards)
    # 然后再将n_shards拆分为n_client份
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # 这里shard是一个shard的数据索引(一个列表)
            # += shard 实质上是在列表里并入列表
            clients_idcs[client_id] += shard

    return clients_idcs

def dirichlet_split_noniid(args,dataset_train,alpha):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    if args.dataset=='svhn':
        train_labels = np.array(dataset_train.labels)
    else:
        train_labels = np.array(dataset_train.targets)
    # train_labels = np.array(dataset_train.labels)
    n_classes = args.num_classes
    n_clients=args.num_users
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def iid_divide(l, g):
    """
    将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
    每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
    返回由不同的groups组成的列表
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

# ================================================================IID==================================================================

def mnist_iid(args,dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fmnist_iid(args,dataset, num_users):
    """
    Sample I.I.D. client data from FashionMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def svhn_iid(args,dataset, num_users):
    """
    Sample I.I.D. client data from SVHN dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(args,dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar100_iid(args,dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# ================================================================Non-IID==================================================================

# def mnist_noniid(args,dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users
#
# def cifar_noniid(args,dataset,num_users):
#
#     num_items = int(len(dataset))
#     dict_users = {}
#     labels = [i for i in range(10)]
#     idx = {i: np.array([], dtype='int64') for i in range(10)}
#
#     j = 0
#     # print((dataset[0][0]))
#     for i in dataset:
#         # print(i)
#         idx[i[1]] = np.append(idx[i[1]],j)
#         j += 1
#
#     # if k = 4, a particular user can have samples only from at max 4 classes
#     k = args.overlapping_classes
#     # print(idx)
#     num_examples = int(num_items/(k*num_users))
#
#     for i in range(num_users):
#         t = 0
#         while(t!=k):
#             j = random.randint(0,9)
#
#             if (len(idx[(i+j)%len(labels)]) >= num_examples):
#                 rand_set = set(np.random.choice(idx[(i+j)%len(labels)], num_examples, replace=False))
#                 idx[(i+j)%len(labels)] = list(set(idx[(i+j)%len(labels)]) - rand_set)
#                 rand_set = list(rand_set)
#                 if(t==0):
#                     dict_users[i] = rand_set
#                 else:
#                     dict_users[i] = np.append(dict_users[i],rand_set)
#                 t += 1
#     return dict_users
#
# def cifar100_noniid(args,dataset,num_users):
#
#     num_items = int(len(dataset))
#     dict_users = {}
#     labels = [i for i in range(100)]
#     idx = {i: np.array([], dtype='int64') for i in range(100) }
#
#     j = 0
#     for i in dataset:
#         # print(i[1])
#         idx[i[1]] = np.append(idx[i[1]],j)
#         j += 1
#     # print(idx.keys())
#     k = args.overlapping_classes
#
#     num_examples = int(num_items/(k*num_users))
#     # print(num_examples)
#
#     for i in range(num_users):
#         # print(i)
#         t = 0
#         while(t!=k):
#             j = random.randint(0,99)
#
#             if (len(idx[(i+j)%len(labels)]) >= num_examples):
#                 rand_set = set(np.random.choice(idx[(i+j)%len(labels)], num_examples, replace=False))
#                 idx[(i+j)%len(labels)] = list(set(idx[(i+j)%len(labels)]) - rand_set)
#                 rand_set = list(rand_set)
#                 if(t==0):
#                     dict_users[i] = rand_set
#                 else:
#                     dict_users[i] = np.append(dict_users[i],rand_set)
#                 t += 1
#     # print(dict_users[0])
#     return dict_users

if __name__ == '__main__':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    test_data = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    classes = train_data.classes
    n_classes = len(classes)