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
    loss_func = nn.SoftmaxLoss()
    correct_sum = 0
    loss_sum = 0.0
    batches = 0
    if opt:
        model.train()
    else:
        model.eval()

    for image,lable in dataloader:
        batches+=1
        if opt: opt.reset_grad()
        image =image.reshape((-1,784))
        predict = model(image)
        loss = loss_func(predict, lable)
        correct_sum += (predict.numpy().argmax(1) == lable.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.numpy()
        
    return 1-correct_sum/len(dataloader.dataset), loss_sum/batches
        


def train_mnist(batch_size=100, epochs=10, optimizer=train.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    training_accuracy = 0.0
    training_loss = 0.0
    test_accuracy = 0.0
    test_loss = 0.0
    train_dataset =train.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz",
                                         data_dir+"/train-labels-idx1-ubyte.gz")
    train_dataloader = train.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    test_dataset = train.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz",
                                         data_dir+"/t10k-labels-idx1-ubyte.gz")
    test_dataloader = train.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
   
    model = MLPResNet(784, hidden_dim)
    model.train()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        training_accuracy, training_loss = epoch(train_dataloader, model, opt)
    test_accuracy, test_loss = epoch(test_dataloader, model)
    return training_accuracy, training_loss, test_accuracy, test_loss

if __name__ == "__main__":
    train_mnist(data_dir="../data")
