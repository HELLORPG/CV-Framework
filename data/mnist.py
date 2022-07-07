# @Author       : Ruopeng Gao
# @Date         : 2022/7/6
# @Description  : MNIST data API.
import gzip
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader


class MNIST:
    """
    MNIST data API.
    """
    def __init__(self, root: str):
        """
        Init a MNIST data API by data root and data split you need.

        Args:
            root: Data root path.
        """
        self.root = root

    def get_split(self, split: str) -> [np.ndarray, np.ndarray]:
        """
        Get a data split, train or test.

        Args:
            split: Data split, 'train' or 'test'

        Returns:
            [Images, Labels]
        """
        if split == "train":
            path = os.path.join(self.root, "train-")
        elif split == 'test':
            path = os.path.join(self.root, "t10k-")
        else:
            raise RuntimeError("MNIST Data don't support split '%s'." % split)

        with gzip.open(path + "labels-idx1-ubyte.gz", "rb") as labels_file:
            labels = np.frombuffer(
                labels_file.read(),
                np.uint8,
                offset=8
            )

        with gzip.open(path + "images-idx3-ubyte.gz", "rb") as images_file:
            images = np.frombuffer(
                images_file.read(),
                np.uint8,
                offset=16
            ).reshape((len(labels), 28, 28))

        return images, labels


class MNISTDataset(Dataset):
    """
    A Dataset class for MNIST dataset.
    """
    def __init__(self, root: str, split: str, transforms=None):
        """
        Init a MNIST dataset class.

        Args:
            root: Data path root.
            split: 'train' or 'test'.
            transforms: How to transform data.
        """
        super(MNISTDataset, self).__init__()

        mnist = MNIST(root=root)
        self.images, self.labels = mnist.get_split(split=split)
        self.transforms = transforms

        return

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    pass
