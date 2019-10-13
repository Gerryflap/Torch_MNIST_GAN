import keras.datasets as ds
import torch

# Load MNIST data from Keras and normalize
from torch.utils.data import Dataset

((images, _), _) = ds.mnist.load_data()
images = torch.from_numpy(images).view(-1, 1, 28, 28).float()/255.0
images = images * 2 - 1


class MnistImageDataset(Dataset):
    def __getitem__(self, item):
        return images[item]

    def __len__(self):
        return images.shape[0]
