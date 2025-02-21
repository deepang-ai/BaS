import os
import pathlib
import random
from functools import partial
from typing import Callable, Hashable, Mapping

import cv2
import monai
import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.utils import cutmix, path_sort


def generate_dataset_BUV2022(root: str):
    import os

    image_path = root + "/images"
    label_path = root + "/labels"
    videos = os.listdir(image_path)
    dataset = []
    for video in videos:
        i_path = os.path.join(image_path, video)
        imgs = path_sort([os.path.join(i_path, x) for x in os.listdir(i_path)])
        for img in imgs:
            frame = img.split("/")[-1][:-4]
            label = os.path.join(label_path, video, frame + ".png")
            if os.path.isfile(label):
                dataset.append({"image": img, "label": label})
    return dataset


def generate_dataset_CVC612(root: str):
    import os

    img_root = root + "/Ground Truth"
    label_root = root + "/Original"
    images = path_sort(os.listdir(img_root))

    dataset = []
    for img in images:
        img_path = os.path.join(img_root, img)
        label_path = os.path.join(label_root, img)
        dataset.append({"image": img_path, "label": label_path})
    return dataset


class ConvertToMultiChannelBasedOnClasses(monai.transforms.transform.Transform):
    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self):
        super().__init__()

    def __call__(
        self, img: monai.config.NdarrayOrTensor
    ) -> monai.config.NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img != 0)]

        return (
            torch.stack(result, dim=0).long().squeeze(0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0).squeeze(0)
        )


class RandomMapTransformWrapper(monai.transforms.MapTransform):
    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        converter: Callable,
        keys: monai.config.KeysCollection = ("image", "label"),  # type: ignore
        allow_missing_keys: bool = False,
        p=0.5,
    ):
        super().__init__(keys, allow_missing_keys)
        self.converter = converter
        self.p = p

    def __call__(self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]):
        if torch.rand(1) <= self.p:
            for key in self.key_iterator(data):
                data[key] = self.converter(data[key])
        return data


class BDataset(data.Dataset):
    def __init__(self, datalist, img_size, transform, train: bool):
        super(BDataset, self).__init__()
        self.datalist = datalist
        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader
        self.resize = transforms.Resize(size=(img_size, img_size))
        self.train = train

    def __getitem__(self, index):
        img = self.resize(self.loader(self.datalist[index]["image"]))
        label = self.resize(Image.open(self.datalist[index]["label"]))
        if self.train:
            rand_index = random.randint(0, len(self.datalist) - 1)
            rand_img = self.resize(self.loader(self.datalist[rand_index]["image"]))
            rand_label = self.resize(Image.open(self.datalist[rand_index]["label"]))
            img, label = cutmix(img, label, rand_img, rand_label)

        sample = self.transform({"image": img, "label": label})
        return sample["image"].float(), sample["label"].float()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.datalist)


class CDataset(data.Dataset):
    def __init__(self, datalist, img_size, transform, train: bool):
        super(CDataset, self).__init__()
        self.datalist = datalist
        self.to_image = transforms.ToPILImage()
        self.transform = transform
        self.size = (img_size, img_size)
        self.train = train

    def __getitem__(self, index):
        img = cv2.resize(
            cv2.imread(self.datalist[index]["image"], cv2.IMREAD_COLOR), self.size
        )
        label = cv2.resize(
            cv2.imread(self.datalist[index]["label"], cv2.IMREAD_GRAYSCALE), self.size
        )
        label[label < 30] = 0
        label[label >= 30] = 1
        sample = self.transform(
            {"image": self.to_image(img), "label": self.to_image(label)}
        )
        return sample["image"].float(), sample["label"].float()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.datalist)


def imagesPath(root, folder, name):
    images = list(pathlib.Path(os.path.join(root, folder)).glob("*{}.*".format(name)))
    return images


def load_data(root):
    labels = ["benign", "malignant"]
    datas = []
    for label in labels:
        data = []
        for img, label in zip(
            imagesPath(root, label, name=")"), imagesPath(root, label, name="mask")
        ):
            data.append({"image": img, "label": label})
        datas.append(data)
    return datas[0], datas[1]


def get_transforms():
    finetune_train_transform = monai.transforms.Compose(
        [
            RandomMapTransformWrapper(transforms.functional.hflip),
            RandomMapTransformWrapper(transforms.functional.vflip),
            RandomMapTransformWrapper(partial(transforms.functional.rotate, angle=90)),
            RandomMapTransformWrapper(partial(transforms.functional.rotate, angle=180)),
            RandomMapTransformWrapper(
                partial(transforms.functional.gaussian_blur, kernel_size=3)
            ),
            RandomMapTransformWrapper(
                partial(transforms.functional.adjust_brightness, brightness_factor=0.3),
                keys="image",
            ),
            RandomMapTransformWrapper(
                partial(transforms.functional.adjust_contrast, contrast_factor=0.3),
                keys="image",
            ),
            RandomMapTransformWrapper(
                partial(transforms.functional.adjust_saturation, saturation_factor=0.3),
                keys="image",
            ),
            RandomMapTransformWrapper(
                partial(transforms.functional.adjust_hue, hue_factor=0.3), keys="image"
            ),
            RandomMapTransformWrapper(transforms.ToTensor(), p=1),
            RandomMapTransformWrapper(
                ConvertToMultiChannelBasedOnClasses(), keys=["label"], p=1
            ),
        ]
    )
    finetune_val_transform = monai.transforms.Compose(
        [
            RandomMapTransformWrapper(transforms.ToTensor(), p=1),
            RandomMapTransformWrapper(
                ConvertToMultiChannelBasedOnClasses(), keys=["label"], p=1
            ),
        ]
    )
    return finetune_train_transform, finetune_val_transform


def get_data_loader(
    path, train_ratio, img_size, in_channels, batch_size, num_workers, dataset=None
):
    train_transform, val_transform = get_transforms()

    if path.BUV2022[0]:
        if dataset is None:
            datasets = generate_dataset_BUV2022(
                path.BUV2022[1]
            )  # [{'image':[frames], 'label':[frames]} * videos]
            training_set = BDataset(
                datalist=datasets[: int(len(datasets) * train_ratio)],
                img_size=img_size,
                transform=train_transform,
                train=True,
            )
            val_set = BDataset(
                datalist=datasets[int(len(datasets) * train_ratio) :],
                img_size=img_size,
                transform=val_transform,
                train=False,
            )
        else:
            training_set = BDataset(
                datalist=dataset,
                img_size=img_size,
                transform=train_transform,
                train=True,
            )
            val_set = BDataset(
                datalist=dataset,
                img_size=img_size,
                transform=val_transform,
                train=False,
            )
    elif path.CVC612[0]:
        datasets = generate_dataset_CVC612(path.CVC612[1])
        training_set = CDataset(
            datalist=datasets[: int(len(datasets) * train_ratio)],
            img_size=img_size,
            transform=train_transform,
            train=True,
        )
        val_set = CDataset(
            datalist=datasets[int(len(datasets) * train_ratio) :],
            img_size=img_size,
            transform=val_transform,
            train=False,
        )
    assert datasets is not None, "ERROR: dataset can not be None!"

    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
