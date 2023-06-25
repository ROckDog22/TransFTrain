# TransFTrain ![](https://img.shields.io/badge/version-v0.1.0-brightgreen)
a simple train framework for personal pytorch learning

# use TransFTrain in main.ipynb
## Train a machine learning model
### import dependencies
```bash
import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.nn as nn
from TransFTrain.data import Dataset, DataLoader
import numpy as np
import time
import os
import struct, gzip
from typing import Optional, List
```

```bash
data_dir = "data"
batch_size = 100
hidden_dim=100
lr =0.01
weight_decay=0.001
epochs = 10
```

### explore the MNIST dataset
```bash
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):  
        self.images, self.labels = self.parse_mnist(image_filename, label_filename)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        data= self.images[index]
        label= self.labels[index]
        if isinstance(index, int):
            data = self.apply_transforms(data.reshape(28, 28, 1))
        else:
            data = np.array([self.apply_transforms(i.reshape(28, 28, 1)) for i in data])
        return data, label

    def __len__(self) -> int:
        return len(self.images)

    def parse_mnist(self, image_filename, label_filename):
        image_file = gzip.open(image_filename, 'rb')
        magic, num_images, rows, cols = struct.unpack(">IIII", image_file.read(16))
        image_data = image_file.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = (images.reshape(num_images, rows * cols)/255).astype(np.float32)
        label_file = gzip.open(label_filename, 'rb')
        magic, num_labels = struct.unpack(">II", label_file.read(8))
        label_data = label_file.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        image_file.close()
        label_file.close()
        return (images, labels)
```

``` bash 
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
```

```bash
train_dataset =MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz",data_dir+"/train-labels-idx1-ubyte.gz")
train_dataloader = train.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz",data_dir+"/t10k-labels-idx1-ubyte.gz")
test_dataloader = train.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

### Train a TransFTrain model to classify digit images
#### define model
```bash
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
        super(ResidualBlock).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.norm1 = norm(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        ret = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = x+ret
        x = self.relu2(x)
        return x

class MLPResNet(nn.Module):
    def __init__(self, dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
        super(MLPResNet).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.module_list = self._model_list(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)

    def _model_list(self, dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
        module_list = []
        for _ in range(num_blocks):
            module_list.append(ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob))
        module_list.append(nn.Linear(hidden_dim, num_classes))
        return nn.Sequential(*module_list)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.module_list(x)
        return x
```

#### train and eval model
```bash
model = MLPResNet(784, hidden_dim)
model.train()
opt = train.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

```bash
loss_func = nn.SoftmaxLoss()
for i in range(epochs):
    correct_sum = 0
    loss_sum = 0.0
    batches = 0
    for image,lable in train_dataloader:
        batches+=1
        opt.reset_grad()
        image =image.reshape((-1,784))
        predict = model(image)
        loss = loss_func(predict, lable)
        correct_sum += (predict.numpy().argmax(1) == lable.numpy()).sum()
        loss.backward()
        loss_sum += loss.numpy()
    print(f"Epoch[{i}] train average error rate", 1-correct_sum/len(train_dataloader.dataset))
    print(f"Epoch[{i}] train average loss", loss_sum/batches)
```

```bash
model.eval()
for image,lable in test_dataloader:
    batches+=1
    image =image.reshape((-1,784))
    predict = model(image)
    loss = loss_func(predict, lable)
    correct_sum += (predict.numpy().argmax(1) == lable.numpy()).sum()
    loss_sum += loss.numpy()
print(f"Epoch[{i}] test average error rate", 1-correct_sum/len(train_dataloader.dataset))
print(f"Epoch[{i}] test average loss", loss_sum/batches)
```