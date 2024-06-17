
# This code is the source code implementation for the paper "ADP-FL: Federated Learning Based on Adaptive Differential Privacy ."



## Abstract
![输入图片说明](/imgs/2024-06-17/Sl271dZxDPHU0bbt.png)
Federated Learning (FL) is a distributed machine learning framework where each participant achieves model training by exchanging model parameters. However, when the server is dishonest, it may lead to the leakage of the participants’ private data. In recent years, many research works have combined federated learning with differential privacy. To ensure user-level differential privacy in federated learning algorithms, the model updates transmitted by the clients need to be clipped before adding noise. However, a fixed clipping threshold may lead to loss of information or the addition of larger noise, especially in scenarios with data heterogeneity where the updates vary greatly among clients, which can impact the performance of the aggregated model. Therefore, in this paper, we propose a Federated Learning Based on Adaptive Differential Privacy (ADP-FL) that dynamically adjusts the clipping threshold based on changes in the norm of the update, aiming to align it as closely as possible with the trend of local updates. To address the problem of privacy protection imbalance among clients caused by a uniform clipping threshold, we introduce a privacy budget allocation strategy that allocates privacy budgets based on the influence of the clipping threshold on the norm of update, to better balance model performance and privacy protection. Finally, we evaluate the performance of the ADP-FL algorithm on the CIFAR-10, CIFAR-100, and MNIST datasets. The experimental results show that our proposed algorithm not only provides effective privacy protection but also exhibits better performance.


# Experimental Environment

**Operating environment:**
CPU: Intel (R) Xeon (R) Gold 6330 CPU @ 2.00GHz
GPU：NVIDIA Tesla A100 SXM4 80G GPU
```
- Pytorch==3.9
- scikit-learn==1.1.3
- numpy==1.23.4
— opacus==1.3.0
- matplotlib==3.7.1
- easydict==1.10
- dill==0.3.6
- pandas==1.5.3

```

## Experimental Setup

### Datasets and Models
1. **Datasets Used**: The experiments were conducted on three datasets:
   - **CIFAR-10**: Consists of 60,000 32×32 color images across 10 categories.
   - **CIFAR-100**: Contains 600 images per category from 100 different classes.
   - **MNIST**: Includes 60,000 training images of handwritten digits and 10,000 test images.

2. **Models**: 
   - **ResNet-18 and ResNet-34** are used, along with a **Convolutional Neural Network (CNN)**. ResNet models are utilized for their ability to handle deeper networks without performance degradation due to issues like vanishing gradients.

### Implementation Details
1. **Parameter Settings**:
   - **Local Epochs**: Set to 4.
   - **Communication Rounds**: 100 rounds.
   - **Clients**: 10 client participants.
   - **Batch Size**: Local batch size is set to 128.
   - **Learning Rate**: 0.01.
   - **Distillation Epochs and Batch Size**: Both set to 4 and 128, respectively.

2. **Personalization**: 
   - The last fully connected layer and basic blocks are considered personalized layers, with the number of these layers set to 2.

3. **Privacy and Clipping**:
   - Clipping threshold initially set with a specific value.
   - Adaptive clipping strategies are used to dynamically adjust the clipping threshold based on the L2 norm of parameter updates.

### Experimental Procedure
1. **Model Training**:
   - Each client trains their model locally using their respective datasets.
   - Parameters are then adjusted based on privacy-preserving mechanisms like adaptive clipping and noise addition.
   
2. **Evaluation**:
   - The effectiveness of the ADP-FL algorithm is assessed by comparing it with other federated learning algorithms like FedPer, FedAvg, and LG-Fed.
   - Model accuracy is evaluated on the CIFAR-10 and MNIST datasets.

3. **Privacy Techniques**:
   - The ADP-FL algorithm uses differential privacy techniques, including an adaptive strategy for applying clipping and noise based on data heterogeneity among clients.

This experimental setup allows the researchers to test the hypothesis that adaptive differential privacy can enhance both the privacy and accuracy of federated learning models, particularly in heterogeneous environments.


## Python Files
### 1. `Fed.py`

This file contain functions related to federated learning. It includes functions like `FedAvg` and `customFedAvg` for averaging weights of models from multiple clients. These functions are used to update the global model in a federated learning setting. Here's a brief on the key functions:

- `FedAvg(w)`: Averages the updated weights of client models, assuming equal number of samples per client.

- `customFedAvg(w, weight=1)`: Similar to `FedAvg` but might include a custom weighting scheme.

### 2. `Nets.py`

This file defines neural network models, possibly used in a machine learning context. It contains class definitions for various types of networks, such as a multi-layer perceptron (MLP) and potentially a convolutional neural network for the MNIST dataset.

### 3. `rdp_accountant.py`

This file related to differential privacy, specifically focused on accounting for privacy loss in a machine learning context using the Rényi Differential Privacy (RDP) framework. Functions like `compute_rdp` and `get_privacy_spent` might be defined here to help manage and calculate the privacy budget used during training.

### 4. `test.py`

This file contains code for testing models. It include functions to evaluate the performance of global models on test datasets, in the context of federated learning or privacy-preserving machine learning.

### 5. `Update.py`

This file manage data updates or processing in a learning system, in federated learning setups. It includes classes or functions that handle dataset splitting and possibly updating model parameters. The `DatasetSplit` class is used to handle parts of a dataset that are allocated to different clients in a federated learning system.


##  Experimental Results
The results demonstrate that the ADP-FL algorithm not only provides robust privacy protection but also enhances model performance across these datasets. It successfully balances privacy and accuracy by dynamically adjusting the clipping threshold based on local updates and employing a privacy budget allocation strategy. This strategy effectively addresses the privacy protection imbalance among clients, optimizing overall model performance while safeguarding user data.
![输入图片说明](/imgs/2024-06-17/YzmF779hahpoDV1T.png)![输入图片说明](/imgs/2024-06-17/UuzQUEkMkJiKBXYe.png)



```
## Update log

```
- {24.06.17} Uploaded overall framework code and readme file
```
