import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.nn as nn
import math
import numpy as np
np.random.seed(0)


class BasicBlock(train.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device, dtype):
        self.conv1 = nn.Conv(in_channels=in_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride,
                             device=device,
                             dtype=dtype)
        self.batchnorm1 = nn.BatchNorm2d(dim=out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        return x

class ResNet9(train.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.BasicBlock1 = BasicBlock(3, 16, 7, 4, device, dtype)
        self.BasicBlock2 = BasicBlock(16, 32, 3, 2, device, dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                BasicBlock(32, 32, 3, 1, device, dtype), 
                BasicBlock(32, 32, 3, 1, device, dtype)
            )
        )
        self.BasicBlock3 = BasicBlock(32, 64, 3, 2, device, dtype)
        self.BasicBlock4 = BasicBlock(64, 128, 3, 2, device, dtype)
        self.res2 = nn.Residual(
            nn.Sequential(
                BasicBlock(128, 128, 3, 1, device, dtype),
                BasicBlock(128, 128, 3, 1, device, dtype)
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )

    def forward(self, x):
        x = self.BasicBlock1(x)
        x = self.BasicBlock2(x)
        x = self.res1(x)
        x = self.BasicBlock3(x)
        x = self.BasicBlock4(x)
        x = self.res2(x)
        x = self.classifier(x)
        return x

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.seq_model = seq_model

        PARAMS = dict(
            device = device, 
            dtype = dtype
        )

        self.embedding = nn.Embedding(output_size, embedding_size, **PARAMS)
        if seq_model == "rnn":
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, **PARAMS)
        elif seq_model == "lstm":
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, **PARAMS)
        else:
            raise NotImplementedError(
                f"Model {seq_model} is not supported. Use rnn or lstm instead."
            )
        self.fc = nn.Linear(hidden_size, output_size, **PARAMS)


    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        out, h = self.model(self.embedding(x), h)
        seq_len, bs, hidden_size = out.shape
        out = self.fc(out.reshape((seq_len*bs, hidden_size)))
        return out, h


if __name__ == "__main__":
    model = ResNet9()
    x = train.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = train.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = train.data.DataLoader(cifar10_train_dataset, 128, train.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)