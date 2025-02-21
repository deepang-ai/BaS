import copy
import logging
import os

import accelerate
import numpy as np
import torch
from torchvision import transforms


def path_sort(paths: list, reverse=False):
    paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]), reverse=reverse)
    return paths


def rand_bbox(size, lam):
    if len(size) == 3:
        W = size[0]
        H = size[1]
    elif len(size) == 4:
        W = size[2]
        H = size[3]
    else:
        raise Exception

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(image, label, rand_image, rand_label):
    to_image = transforms.ToPILImage()
    # Image 2 array
    image = np.array(image)
    label = np.array(label)
    rand_image = np.array(rand_image)
    rand_label = np.array(rand_label)
    # select border
    lam = np.random.beta(1, 1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)
    # cat image
    image[bbx1:bbx2, bby1:bby2, :] = rand_image[bbx1:bbx2, bby1:bby2, :]
    label[bbx1:bbx2, bby1:bby2] = rand_label[bbx1:bbx2, bby1:bby2]

    return to_image(image), to_image(label)


class MetricSaver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.best_acc = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


def resume_train_state(
    path: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    logger: logging.Logger,
    accelerator: accelerate.Accelerator,
):
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + "/" + path
        dirs = [
            base_path + "/" + f.name
            for f in os.scandir(base_path)
            if (f.is_dir() and f.name.startswith("epoch_"))
        ]
        dirs.sort(
            key=os.path.getctime
        )  # Sorts folders by date modified, most recent checkpoint is the last
        logger.info(f"Try to load epoch {dirs[-1]} train state")
        accelerator.load_state(dirs[-1])
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        if val_loader is not None:
            val_step = starting_epoch * len(val_loader)
        else:
            val_step = 0
        logger.info(f"Load train state success! Start from epoch {starting_epoch}")
        return starting_epoch, step, val_step
    except Exception as e:
        logger.error(e)
        logger.error(f"Load train state fail!")
        return 0, 0, 0
