import statistics
from itertools import combinations

import matplotlib
from torch import nn

matplotlib.use('Agg')
import matplotlib.pyplot as pltz
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torchsummary import summary
import time
import random
import logging
import json
from hashlib import md5
import copy
import easydict
import os
import sys
from collections import defaultdict, deque, OrderedDict
from torch.utils.tensorboard import SummaryWriter
import pickle

import datetime
# Directory where the json file of arguments will be present    存放json的参数文件的目录
directory = './Parse_Files'

# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model, Load_Model1

from models.Update import train_client, train_client_w, test_client, finetune_client, Distilling, calculate_ave, \
    calculate_mean_std, corresponding_base_layer, calculate_average, client_w, train_client_w1, train_client_w2
from models.Fed import FedAvg, DiffPrivFedAvg
from models.test import test_img

torch.manual_seed(0)


if __name__ == '__main__':
    
    # Initialize argument dictionary 初始化参数字典
    args = {}

    # From Parse_Files folder, get the name of required parse file which is provided while running this script from bash
    f = directory+'/'+str(sys.argv[1])
    print(f)
    with open(f) as json_file:  
        args = json.load(json_file)

    # Taking hash of config values and using it as filename for storing model parameters and logs
    param_str = json.dumps(args)
    file_name1 = md5(param_str.encode()).hexdigest()

    # Converting args to easydict to access elements as args.device rather than args[device]
    args = easydict.EasyDict(args)
    print(args)

    # Save configurations by making a file using hash value 通过使用哈希值创建文件来保存配置
    with open('./config/parser_{}.txt'.format(file_name1),'w') as outfile:
        json.dump(args,outfile,indent=4)

    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # ================================================创建存放数据的文件==================================================
    # 定义一个.txt文件名
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    file_name = 'accuracy_{}.txt'.format(timestamp)

    # 指定文件路径和文件名
    file_path = './results.txt/'

    # 如果文件路径不存在，则创建它
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 将文件路径和文件名合并起来
    file_path_name = os.path.join(file_path, file_name)
    #=================================================导入数据集以及初始化模型=============================================

    # Load the training and testing datasets
    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)

    #splitting user data into training and testing parts 将用户数据拆分为训练和测试部分
    train_data_users = {}
    test_data_users = {}

    #=======计算第n项值=======
    def nth_term10(n, d):
        nth_term = 0.185 + (n - 1) * d
        return nth_term

    def nth_term5(n, d):
        nth_term = 0.082 + (n - 1) * d
        return nth_term

    def nth_term2(n, d):
        nth_term = 0.0219 + (n - 1) * d
        return nth_term

    for i in range(args.num_users):
        dict_users[i] = list(dict_users[i])
        train_data_users[i] = list(random.sample(dict_users[i],int(args.split_ratio*len(dict_users[i]))))
        test_data_users[i] = list(set(dict_users[i])-set(train_data_users[i]))
        # test_data_users[i] = random.sample(test_data_users[i], 500)
# =================================================初始化模型===========================================================
    # Initialize Global Server Model
    net_glob = Load_Model(args=args)

    # 打印每一层参数
    # for k,v in net_glob.state_dict().items():
    #     print(k)

    net_glob.train()

    # copy weights 复制权重
    w_glob = net_glob.state_dict()

    # local models for each client 每个客户端的本地模型
    local_nets = {}

    for i in range(0,args.num_users):
        local_nets[i] = Load_Model(args=args)      #创建新的网络模型
        local_nets[i].train()                      #进入训练模式
        local_nets[i].load_state_dict(w_glob)      #全局模型的参数加载到本地模型

    #创建集成模型
    Ensemble_model = Load_Model1(args=args)
# =================================================开始联邦学习训练========================================================

    # Start training 开始训练

    print("Start Training")
    print("网络模型：",args.model)
    print("本地数据集：",args.dataset)
    print("差分隐私机制：", args.dp_mechanism)

    start = time.time()
    all_accuracy = 0

    all_acc_list = []
    all_global_acc_list = []
    count = [0, 0, 0, 0]  # 统计个性化层数
    total_epsilon = [0,0,0,0,0,0,0,0,0,0] # 客户端消耗的总隐私预算

    # ================================Adc================================
    ada_clip = args.norm_clip
    # 定义滑动窗口
    window = {}
    for idx in range(0,args.num_users):
        window[idx] = deque(maxlen=3)
    # 标签
    lb = 0
    state = 0 # 隐私预算是否消耗完
    # =============================Adc=================================

    # ================================================动态调整个性化层=====================================================
    if args.model == "cnn":
        pers = [1, 2]
        base = corresponding_base_layer(args.model, args.dataset)
    else:
        pers = [1, 2, 3, 4]
        base = corresponding_base_layer(args.model, args.dataset)

    dictionary = dict(zip(pers, base))
    dy_per = 2
    # =====================================================动态调整个性化层================================================
    for iter in range(args.epochs):
        print("---------Round {}---------".format(iter))
        with open(file_path_name, 'a') as f:
            f.write("round: " + str(iter) + "  ")
#======================================================================客户端============================================
        w_locals, loss_locals = [], []
        acc_list = []  # 准确率
        std_list = []  # 标准差列表
        avg_norm = []  # 计算每轮的平均范数
        w_start = []   # 在其基础上加更新量
        w_delta = []   # 更新量
        delta_list = []  # 统计更新量占比
        if iter == 0:
            base_layers = dictionary.get(dy_per)

        print("裁剪阈值：", ada_clip)

        if args.dp_mechanism == 'no_dp':
            for idx in range(0,args.num_users):
                t_e = total_epsilon[idx]
                # 客户端模型进行训练（不加噪）
                w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                w_locals.append(w)
                loss_locals.append(copy.deepcopy(loss))
        else:
            if args.Ad_clip == "ada":
                for idx in range(0, args.num_users):
                    t_e = total_epsilon[idx]
                    # 客户端模型进行训练（在参数上加噪）
                    w_s, loss, total_norm, w_d = train_client_w(args, ada_clip, base_layers, dataset_train,train_data_users[idx], net=local_nets[idx])
                    w_start.append((w_s))
                    w_delta.append(w_d)

                    loss_locals.append(copy.deepcopy(loss))
                    window[idx].append(total_norm)
                    if args.Ad_clip == "ada":
                        gama = 0
                        if len(window[idx]) == 3:
                            values = [val[0] for val in list(window[idx])]
                            # values = list(window[idx])
                            mean_value, std_dev = calculate_mean_std(values)
                            if mean_value <= ada_clip:
                                std_dev = std_dev * (1 + gama)
                            else:
                                std_dev = -std_dev * (1 - gama)
                            std_list.append(std_dev)
                            lb = 1
                        else:
                            lb = 0
                    avg_norm.append(total_norm)
                # ==============================================诚实的服务器==================================================
                # 自适应裁剪阈值
                if args.Ad_clip == "ada":
                    # print(std_list)
                    if lb == 1:
                        # 计算列表中元素的平均值
                        list_avg = calculate_ave(std_list)
                        ada_clip = ada_clip - list_avg
                        # print("平均标准差", list_avg)
                        # print("衰减后的裁剪阈值为：", ada_clip)

                # ==============================================裁剪加载=================================================
                for idx in range(0, args.num_users):
                    t_e = total_epsilon[idx]
                    w, total_epsilon[idx] = client_w(args, avg_norm, w_start[idx], w_delta[idx], ada_clip, t_e, base_layers, window[idx], net=local_nets[idx])
                    w_locals.append(w)
                # with open(file_path_name, 'a') as f:
                #     f.write("client" + str(idx))
                #     f.write(': {:.14f}'.format(total_norm.item()) + "  ")
            elif args.Ad_clip == "adc":
                for idx in range(0, args.num_users):
                    w, loss, total_norm2, b, total_norm = train_client_w1(args, ada_clip, base_layers, dataset_train,train_data_users[idx], net=local_nets[idx])
                    w_locals.append(w)
                    loss_locals.append(copy.deepcopy(loss))
                    delta_list.append(b)
                    avg_norm.append(total_norm)

            elif args.Ad_clip == "ad":
                # 衰减
                ada_clip = ada_clip * torch.exp(torch.tensor(-0.02 * iter))
                for idx in range(0, args.num_users):
                    w, loss, total_norm = train_client_w2(args, ada_clip, base_layers, dataset_train,train_data_users[idx], net=local_nets[idx])
                    w_locals.append(w)
                    loss_locals.append(copy.deepcopy(loss))
                    avg_norm.append(total_norm)
            total_norm_avg = calculate_average(avg_norm)  # 所有客户端total_norm的平均值

        # 测试准确率
        s = 0
        for i in range(args.num_users):
            acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
            acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
            s += acc_test
            acc_list.append(acc_test)
        s /= args.num_users    # 平均准确率
        all_acc_list.append(s)    # 统计最高准确率
        iter_loss = sum(loss_locals) / len(loss_locals)

        # ================================保存文件=======================================================================
        with open(file_path_name, 'a') as f:
            # f.write("round: " + str(iter) + "  ")
            f.write('loss: {:.14f}'.format(iter_loss) + "  ")
            if args.Ad_clip == "ada":
                f.write('L2: {:.8f}'.format(total_norm_avg.item()) + "  ")
            f.write('Accuracy: {:.4f}  '.format(s))
        print("客户端的平均准确率: {: .3f}".format(s))
# ======================================================================服务器=====================================================================
        # 将本地模型的前base_layers层参数提取出来
        w_ens = {}  # 保存每一个客户端的基础层参数
        w_Ens = []  # 将所有客户端的基础层参数添加到列表里
        sim = []   # 保存这一轮客户端的相似度

        if args.Ad_clip == "adc":
            # 计算列表中元素的平均值
            adapt = 0.5
            lrate = 0.2
            b_avg = calculate_average(delta_list)
            lrate_tensor = torch.tensor(lrate)
            ada_clip = ada_clip * torch.exp(-lrate_tensor * (b_avg - adapt))

        for idx in range(args.num_users):
            for i in list(w_locals[idx].keys())[0:base_layers]:
                w_ens[i] = copy.deepcopy(w_locals[idx][i])
            w_Ens.append(w_ens)

        # 聚合基础层的参数
        w_Ensemble = FedAvg(w_Ens)

        if args.model == 'ResNet':
            Personal_layer = base_layers - 218
        elif args.model == 'cnn':
            if args.dataset == 'cifar':
                Personal_layer = base_layers - 10
            elif args.dataset == 'mnist':
                Personal_layer = base_layers - 8
        elif args.model == 'MobileNet':
            Personal_layer = base_layers - 164
        elif args.model == 'ResNet50':
            Personal_layer = base_layers - 320
        elif args.model == 'ResNet18':
            Personal_layer = base_layers - 122

        # 全局模型的个性化层
        new_personal_layer = {}                             # 全局模型的个性化层参数
        for i in list(w_glob.keys())[Personal_layer:]:
            new_personal_layer[i] = copy.deepcopy(w_glob[i])

        # 将个性化层参数添加到基础层参数后面
        w_Ensemble.update(new_personal_layer)

        # 集成模型的参数复制到集成模型上
        Ensemble_model.load_state_dict(w_Ensemble)

        #=======================知识蒸馏====================
        Distilling(args, net_glob, Ensemble_model, iter)

        # copy weights
        w_glob = net_glob.state_dict()

        # # ===============================================统计所有客户端的投票（选择个性化层）=================================
        # new_acc = []
        # if iter == 0:
        #     old_acc = acc_list
        # else:
        #     new_acc = acc_list
        #
        #     result = [x - y for x, y in zip(new_acc, old_acc)]
        #     positive_count = 0
        #     negative_count = 0
        #
        #     for num in result:
        #         if num > 0:
        #             positive_count += 1
        #         elif num < 0:
        #             negative_count += 1
        #     old_acc = new_acc
        #
        # # ======================================动态调整层=========================
        # if iter < args.epochs * 0.8:
        #     if iter != 0:
        #         if args.model == "cnn":
        #             if positive_count > negative_count:
        #                 dy_per = dy_per - 1
        #             elif positive_count < negative_count:
        #                 dy_per = dy_per + 1
        #
        #             if dy_per > 2:
        #                 dy_per = 2
        #             if dy_per < 1:
        #                 dy_per = 1
        #
        #             max_index = dy_per
        #             base_layers = dictionary.get(max_index)
        #
        #         else:
        #             if positive_count > negative_count:
        #                 dy_per = dy_per - 1
        #             elif positive_count < negative_count:
        #                 dy_per = dy_per + 1
        #
        #             if dy_per > 4:
        #                 dy_per = 4
        #             if dy_per < 1:
        #                 dy_per = 1
        #
        #             max_index = dy_per
        #             base_layers = dictionary.get(max_index)
        #     count[dy_per - 1] += 1
        # elif iter == args.epochs * 0.8:
        #     max_value = max(count)  # 找到列表中的最大值
        #     # max_index = [i for i, value in enumerate(count) if value == max_value]  # 查找最大值的索引
        #     max_index = count.index(max_value)
        #     base_layers = dictionary.get(max_index+1)
        #
        # print("动态选择后的基础层数:", base_layers)
        # ==============================================选择完毕=====================================================

# ======================================================================客户端=====================================================================
        # 更新客户端的基础层并保持个性化层不变
        # 将全局模型的前base_layers层参数复制到本地模型
        for idx in range(args.num_users):
            for i in list(w_glob.keys())[0:base_layers]:     #第一个循环遍历所有客户端，而第二个循环则遍历全局模型的前base_layers层的参数
                w_locals[idx][i] = copy.deepcopy(w_glob[i]) 
            local_nets[idx].load_state_dict(w_locals[idx])

        ### FineTuning
        if args.finetune:
            # print("FineTuning")
            personal_params=list(w_glob.keys())[base_layers:]
            for idx in range(0,args.num_users):
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=False
                w,loss = finetune_client(args,dataset_train,train_data_users[idx],net = local_nets[idx])
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=True

            s = 0
            for i in range(args.num_users):
                logging.info("Client {}:".format(i))
                acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
                acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
                s += acc_test
            s /= args.num_users
            all_global_acc_list.append(s) # 统计全局最高准确率

            print("全局精度: {: .3f}".format(s))
            with open(file_path_name, 'a') as f:
                f.write('Global_Accuracy: {:.4f}'.format(s) + "  ")
                f.write('clip: {:.8f}\n'.format(ada_clip))
                # f.write(f"累积消耗的隐私预算: {total_epsilon}")
        for value in total_epsilon:
            if value > args.dp_epsilon * 100:
                state = 1
        if state == 1:
            break
    end = time.time()
    print("Training Time: {}s".format(end-start))
    print("End of Training")
# ======================================================================联邦学习结束=====================================================================

    max_acc = max(all_acc_list)
    max_acc1 = max(all_global_acc_list)
    print("本地最高准确率: {: .4f}".format(max_acc))
    print("全局最高准确率: {: .4f}".format(max_acc1))
    print("累计消耗隐私预算：", total_epsilon)

    for i in range(4):
        print(f"个性化层为{i + 1}：{count[i]}次。")

    with open(file_path_name, 'a') as f:
        f.write('本地最高准确率: {: .4f}\n'.format(max_acc))
        f.write('全局最高准确率: {: .4f}\n'.format(max_acc1))
        f.write(f"累积消耗的隐私预算: {total_epsilon}\n")
        for i in range(4):
            f.write(f"个性化层为{i + 1}：{count[i]}次。")

