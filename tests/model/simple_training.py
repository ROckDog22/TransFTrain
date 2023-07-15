import sys
sys.path.append('../python')
import TransFTrain as train
import TransFTrain.nn as nn
from TransFTrain import backend_ndarray as nd
from models import *
import time
from tqdm import trange
from tqdm.auto import tqdm

device = train.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.train()
    if opt is None:
        model.eval()

    count = loss_sum = accuracy = 0
    for batch in tqdm(dataloader, total=len(dataloader.dataset) // dataloader.batch_size):
        images, labels = batch
        images, labels = train.Tensor(images, device=model.device), train.Tensor(labels, device=model.device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if opt is not None:
            loss.backward()
            opt.step()

        count += images.shape[0]
        loss_sum += loss.detach().numpy() * images.shape[0]
        accuracy += (logits.detach().numpy().argmax(-1) == labels.detach().numpy()).sum()
        
    avg_loss = loss_sum / count
    avg_accuracy = accuracy / count

    return avg_accuracy, avg_loss  


def train_cifar10(model, dataloader, n_epochs=1, optimizer=train.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(n_epochs):
        start = time.time()
        train_acc, train_loss = epoch_general_cifar10(dataloader, model, loss_fn(), optim)
        end = time.time()
        print(
            f"Epoch {i}/{n_epochs - 1}", 
            f"accuracy: {train_acc}", 
            f"loss: {train_loss}",
            f"time: {end - start:.3f}s",
        )
    
    return train_acc, train_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    test_acc, test_loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt=None)
    print(f"accuracy: {test_acc}", f"loss: {test_loss}")

    return test_acc, test_loss




### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.train()
    if opt is None:
        model.eval()

    count = loss_sum = accuracy = 0
    last = None
    for i in trange(0, data.shape[0] - 1, seq_len):
        texts, labels = train.data.get_batch(data, i, seq_len, device=device, dtype=dtype)

        logits, last = model(texts, last)
        loss = loss_fn(logits, labels)

        # need to "break" autograd graph to avoid super deep recursion
        if isinstance(last, tuple):
            last = tuple(
                last_part.detach() for last_part in last
            )
        else:
            last = last.detach()

        if opt is not None:
            loss.backward()
            opt.step()

            if clip is not None:
                for param in opt.params:
                    param.grad = train.Tensor(
                        clip * param.grad.data / np.linalg.norm(param.grad.data),
                        device=param.device,
                        dtype=param.dtype,
                    )

        count += labels.shape[0]
        loss_sum += loss.detach().numpy().squeeze() * labels.shape[0]
        accuracy += (logits.detach().numpy().argmax(-1) == labels.detach().numpy()).sum()
        
    avg_loss = loss_sum / count
    avg_accuracy = accuracy / count

    return avg_accuracy, avg_loss   


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=train.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(n_epochs):
        start = time.time()
        train_acc, train_loss = epoch_general_ptb(
            data, 
            model,
            loss_fn=loss_fn(), 
            opt=optim, 
            clip=clip,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        end = time.time()
        print(
            f"Epoch {i}/{n_epochs - 1}", 
            f"accuracy: {train_acc}", 
            f"loss: {train_loss}",
            f"time: {end - start:.3f}s",
        )
    
    return train_acc, train_loss


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    test_acc, test_loss = epoch_general_ptb(
            data, 
            model,
            loss_fn=loss_fn(), 
            opt=None,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
    print(f"accuracy: {test_acc}", f"loss: {test_loss}")
    
    return test_acc, test_loss


if __name__ == "__main__":
    ### For testing purposes
    device = train.cpu()
    #dataset = train.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = train.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=train.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = train.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = train.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
