import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image, ImageFilter
import cv2


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _is_image_file(name):
    return os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS


def _list_class_images(class_dir):
    if not os.path.isdir(class_dir):
        return []
    out = []
    for name in sorted(os.listdir(class_dir)):
        if _is_image_file(name):
            out.append(os.path.join(class_dir, name))
    return out


def _split_indices(n):
    """70% / 10% / (remainder test) using integer tenths."""
    if n == 0:
        return 0, 0, 0
    n_train = (7 * n) // 10
    n_val = (1 * n) // 10
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _read_rgb_numpy(path):
    arr = cv2.imread(path, cv2.IMREAD_COLOR)
    if arr is not None and arr.size > 0:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class NCT_CRC_HE_Dataset(data.Dataset):
    """
    CRC-style 9-class folder layout under ``data_path``:
      ADI/, BACK/, ... TUM/

    Train/val/test are drawn from the same folders using a per-class 70/10/20 split
    (remainder to test) with ``numpy.random.RandomState(42)``.
    """

    def __init__(self, data_path, split="train", crop_size=(224, 224)):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")
        if split == "train":
            self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(size=crop_size),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=crop_size),
                    transforms.ToTensor(),
                ]
            )
        self.data_path = data_path
        self.classes = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

        self.train_samples = []
        self.train_labels = []

        rng = np.random.RandomState(42)

        for class_name, index in self.classes.items():
            class_dir = os.path.join(data_path, class_name)
            paths = _list_class_images(class_dir)
            n = len(paths)
            if n == 0:
                print(class_dir, class_name, 0)
                continue
            order = rng.permutation(n)
            paths = [paths[i] for i in order]
            n_train, n_val, n_test = _split_indices(n)
            if split == "train":
                chosen = paths[:n_train]
            elif split == "val":
                chosen = paths[n_train:n_train + n_val]
            else:
                chosen = paths[n_train + n_val:]
            print(class_dir, class_name, n, split, len(chosen))
            self.train_samples.extend(chosen)
            self.train_labels.extend([index] * len(chosen))

        print(split, f'dataset samples: {len(self.train_samples)}, labels: {len(self.train_labels)}')

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, index):
        img_path = self.train_samples[index]
        label = float(self.train_labels[index])
        label = torch.tensor([label])
        image2D = _read_rgb_numpy(img_path)
        image2D = self.transform(image2D)
        return image2D, label
