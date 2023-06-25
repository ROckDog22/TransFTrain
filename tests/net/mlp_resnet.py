import sys
sys.path.append('../python')
import TransFTrain as train
import TransFTrain.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(nn.Residual(fn),nn.ReLU())

def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    module_list = []
    module_list.append(nn.Linear(dim, hidden_dim))
    module_list.append(nn.ReLU())
    for _ in range(num_blocks):
        module_list.append(ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob))
    module_list.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*module_list)

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=train.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
