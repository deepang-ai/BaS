import logging
import os
from collections import defaultdict
from datetime import datetime

import monai
import pytz
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils.random import set_seed
from easydict import EasyDict
from objprint import objstr
from timm.optim import optim_factory

from src.bas import Bas
from src.utils import utils
from src.utils.image_loader import get_data_loader


class Trainer(object):
    def __init__(self, config: EasyDict):
        set_seed(42)
        torch.multiprocessing.set_sharing_strategy("file_system")
        time = str(
            datetime.now(tz=pytz.timezone("Asia/Shanghai")).strftime(
                "%Y-%m-%d-%H-%M-%S"
            )
        )
        logging_dir = os.path.join(os.getcwd(), "logs", time)

        self.accelerator = Accelerator(
            log_with="tensorboard",  # type:ignore
            project_dir=logging_dir,
            mixed_precision=config.trainer.mixed_precision,  # TODO
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,  # TODO
        )
        self.accelerator.init_trackers(__name__)
        if self.accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d-%H:%M:%S",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(logging_dir + "/log.txt"),
                ],
                force=True,
            )
        self.logger = get_logger(__name__)
        self.logger.info(objstr(config))
        self.config = config
        self.train_step = 0
        self.val_step = 0
        self.starting_epoch = 0
        self.metric_saver = utils.MetricSaver()
        self.logger.info("Load dataset")
        self.train_loader, self.val_loader = get_data_loader(
            config.dataset.path,
            config.trainer.train_ratio,
            config.trainer.formal_size,
            config.trainer.in_channels,
            config.trainer.batch_size,
            config.trainer.num_workers,
        )

        self.model = Bas(**config.model.bas)

        self.metrics = {
            "Jaccard": monai.metrics.MeanIoU(include_background=False),
            "f1": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="f1 score"
            ),
            "precision": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="precision"
            ),
            "recall": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="recall"
            ),
            "dice_metric": monai.metrics.DiceMetric(
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=False,
            ),
        }
        # loss functions
        self.loss_functions = {
            "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
            "BCE_with_logits_loss": torch.nn.BCEWithLogitsLoss(),
            "dice_loss": monai.losses.DiceLoss(
                smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
            ),
        }
        self.post_trans = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.optimizer = optim_factory.create_optimizer_v2(
            self.model,
            opt=config.trainer.optimizer.opt,
            weight_decay=config.trainer.optimizer.weight_decay,  # TODO
            lr=config.trainer.optimizer.lr,
            betas=(0.9, 0.95),
        )
        self.scheduler = transformers.get_scheduler(
            "cosine_with_restarts",
            self.optimizer,
            num_warmup_steps=config.trainer.scheduler.warmup_epochs
            * len(self.train_loader),
            num_training_steps=config.trainer.num_epochs * len(self.train_loader),
        )

    def _run(self):
        self.accelerator.wait_for_everyone()
        self.logger.info("Register metric saver")
        self.accelerator.register_for_checkpointing(self.metric_saver)
        self.metric_saver.to(self.accelerator.device)
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,  # type:ignore
        )

        self.starting_epoch, self.train_step, self.val_step = utils.resume_train_state(
            self.config.trainer.finetune_save_dir,
            self.train_loader,
            self.val_loader,
            self.logger,
            self.accelerator,
        )

        self.logger.info("Start training!")
        best_acc = 0
        for epoch in range(self.starting_epoch, self.config.trainer.num_epochs):
            # 训练
            self.train_one_epoch(epoch)
            # 验证
            self.logger.info(
                "----------Epoch [{}/{}] Start verification----------".format(
                    epoch + 1, self.config.trainer.num_epochs
                )
            )
            mean_acc = self.val_one_epoch(epoch)

            self.logger.info(
                "----------Epoch [{}/{}] Verification results-----mean acc :{}-----lr = {}----------".format(
                    epoch + 1,
                    self.config.trainer.num_epochs,
                    mean_acc,
                    self.scheduler.get_last_lr(),
                )
            )
            self.accelerator.log({"lr": self.scheduler.get_last_lr()}, step=epoch)
            if mean_acc > self.metric_saver.best_acc:
                best_acc = mean_acc
                self.metric_saver.best_acc.data = mean_acc
                self.accelerator.save_state(
                    output_dir=f"{os.getcwd()}/{self.config.trainer.finetune_save_dir}/best"
                )

            self.accelerator.save_state(
                output_dir=f"{os.getcwd()}/{self.config.trainer.finetune_save_dir}/epoch_{epoch}"
            )
            self.logger.info(
                "----------Epoch [{}/{}] Verification results-----best acc :{}-----lr = {}----------".format(
                    epoch + 1,
                    self.config.trainer.num_epochs,
                    best_acc,
                    self.scheduler.get_last_lr(),
                )
            )
        self.logger.info(f"----------End of all training----------/n")
        self.logger.info(f"Best acc: {best_acc}")
        exit(1)

    def _run_test(self):
        self.accelerator.wait_for_everyone()
        self.logger.info("Register metric saver")
        self.accelerator.register_for_checkpointing(self.metric_saver)
        self.metric_saver.to(self.accelerator.device)
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,  # type:ignore
        )

        self.model.load_state_dict(
            torch.load(
                f"./{self.config.trainer.finetune_save_dir}/best/pytorch_model.bin"
            )
        )

        self.logger.info("Start Testing!")

        # 验证
        self.logger.info("----------Start verification----------")
        Jaccard, f1, precision, recall, dice = self.val_one_epoch(epoch=-1)
        self.logger.info(f"----------End verification----------/n")

        self.logger.info(f"Best dice: {dice}")
        self.logger.info(f"Best Jaccard: {Jaccard}")
        self.logger.info(f"Best f1: {f1}")
        self.logger.info(f"Best precision: {precision}")
        self.logger.info(f"Best recall: {recall}")
        exit(1)

    def train_one_epoch(self, epoch):
        self.model.train()
        for i, (image, label) in enumerate(self.train_loader):
            with self.accelerator.accumulate(self.model):
                # total_loss, log = self.update_epoch(images=value['rgb'], labels=value['cls_gt'], is_train=True)
                total_loss, log = self.update_epoch(
                    image=image, label=label, is_train=True
                )
                self.accelerator.backward(total_loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.accelerator.log(
                    {
                        "Train/Total Loss": float(total_loss),
                    },
                    step=self.train_step,
                )
                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Training [{i + 1}/{len(self.train_loader)}] Loss: {float(total_loss):1.5f} {log}"
                )
                self.train_step += 1

        _metric = {}
        for metric_name, metric_fn in self.metrics.items():
            batch_acc = metric_fn.aggregate()
            if isinstance(batch_acc, list):
                batch_acc = torch.Tensor(batch_acc).mean()
            metric_fn.reset()
            _metric.update(
                {
                    f"Train/mean {metric_name}": float(batch_acc),
                }
            )
        self.logger.info(
            f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Training metric {_metric}"
        )
        self.accelerator.log(_metric, step=epoch)

    @torch.no_grad()
    def val_one_epoch(self, epoch=-1):
        self.model.eval()
        if epoch == -1:
            for i, (image, label) in enumerate(self.val_loader):
                total_loss, log = self.update_epoch(
                    image=image, label=label, is_train=False
                )
                self.logger.info(f"Validation [{i + 1}/{len(self.val_loader)}] {log}")
                self.val_step += 1
            _metric = {}
            for metric_name, metric_fn in self.metrics.items():
                batch_acc = metric_fn.aggregate()
                if isinstance(batch_acc, list):
                    batch_acc = torch.Tensor(batch_acc).mean()
                metric_fn.reset()
                _metric.update(
                    {
                        f"Val/mean {metric_name}": float(batch_acc),
                    }
                )
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Validation metric {_metric}"
            )
            self.accelerator.log(_metric, step=epoch)
            Jaccard = (
                torch.Tensor([_metric["Val/mean Jaccard"]])
                .float()
                .to(self.accelerator.device)
            )
            f1 = (
                torch.Tensor([_metric["Val/mean f1"]])
                .float()
                .to(self.accelerator.device)
            )
            precision = (
                torch.Tensor([_metric["Val/mean precision"]])
                .float()
                .to(self.accelerator.device)
            )
            recall = (
                torch.Tensor([_metric["Val/mean recall"]])
                .float()
                .to(self.accelerator.device)
            )
            dice = (
                torch.Tensor([_metric["Val/mean dice_metric"]])
                .float()
                .to(self.accelerator.device)
            )
            return Jaccard, f1, precision, recall, dice
        else:
            for i, (image, label) in enumerate(self.val_loader):
                total_loss, log = self.update_epoch(
                    image=image, label=label, is_train=False
                )
                self.accelerator.log(
                    {
                        "Val/Total Loss": float(total_loss),
                    },
                    step=self.val_step,
                )
                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Validation [{i + 1}/{len(self.val_loader)}] Loss: {(float(total_loss)):1.5f} {log}"
                )
                self.val_step += 1

            _metric = {}
            for metric_name, metric_fn in self.metrics.items():
                batch_acc = metric_fn.aggregate()
                if isinstance(batch_acc, list):
                    batch_acc = torch.Tensor(batch_acc).mean()
                metric_fn.reset()
                _metric.update(
                    {
                        f"Val/mean {metric_name}": float(batch_acc),
                    }
                )
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Validation metric {_metric}"
            )
            self.accelerator.log(_metric, step=epoch)
            return torch.Tensor([_metric["Val/mean dice_metric"]]).to(
                self.accelerator.device
            )

    def update_epoch(self, image, label, is_train: bool):
        action = "Train" if is_train else "Val"
        loss = defaultdict(int)
        log = ""
        # 计算模型fps

        logits = self.model(image)
        label = label
        for loss_name, loss_fn in self.loss_functions.items():
            with self.accelerator.autocast():
                l = loss_fn(logits, label)
            loss[loss_name] += l
        logits = self.post_trans(logits)

        # Compute acc
        logits, label = self.accelerator.gather_for_metrics((logits, label))

        for metric_name, metric_fn in self.metrics.items():
            metric_fn(y_pred=logits, y=label)

        total_loss = 0
        for name in loss.keys():
            mean_loss = loss[name]
            self.accelerator.log(
                {action + "/" + name: float(mean_loss)}, step=self.train_step
            )
            log += f" {name} {float(mean_loss):1.5f} "
            total_loss += mean_loss
        return total_loss, log

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Train error!")
            self.accelerator.end_training()
            exit(-1)

    def run_val(self):
        try:
            self._run_test()
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Val error!")
            self.accelerator.end_training()
            exit(-1)


if __name__ == "__main__":
    cfg = EasyDict(
        yaml.load(open("config.yaml", "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    )
    Trainer(cfg).run()
    # Trainer(cfg).run_val()
