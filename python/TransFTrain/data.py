import numpy as np
import struct
import gzip
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        return np.flip(img, axis=1) if flip_img else img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        pad = self.padding
        ret = np.pad(img, pad)
        H, W, C = ret.shape
        ret = ret[pad+shift_x:H-(pad-shift_x),pad+shift_y:W-(pad-shift_y),pad:C-pad]
        return ret


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            ordering = np.arange(len(self.dataset))
            np.random.shuffle(ordering)
            self.ordering = np.array_split(ordering, range(self.batch_size, len(self.dataset), self.batch_size))
        self.__iterator = iter(self.ordering)
        return self

    def __next__(self):
        return [Tensor(x) for x in self.dataset[self.__iterator.__next__()]]


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

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
