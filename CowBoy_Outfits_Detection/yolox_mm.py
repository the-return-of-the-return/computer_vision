#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.distributed as dist
import torch.nn as nn

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "/content/drive/MyDrive/CowBoy/working/"
        self.train_ann = "new_train.json"
        self.val_ann = "new_train.json"
        self.dataset_name = "images"

        self.num_classes = 5
        self.max_epoch = 10
        #self.data_num_workers = 2
        self.eval_interval = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            COCODataset,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler
        )

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            name=self.dataset_name,
            preproc=TrainTransform(
                # rgb_means=(0.485, 0.456, 0.406),
                # std=(0.229, 0.224, 0.225),
                max_labels=50
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                # rgb_means=(0.485, 0.456, 0.406),
                # std=(0.229, 0.224, 0.225),
                max_labels=120
            ),
            degrees=self.degrees,
            translate=self.translate,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.dataset_name,
            img_size=self.test_size,
            preproc=ValTransform(
                # rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader