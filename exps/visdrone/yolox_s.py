#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        # self.num_classes = 80
        self.num_classes = 10
        # self.depth = 1.00
        # self.width = 1.00
        self.depth = 0.33
        self.width = 0.50
        self.act = 'silu'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        #self.data_num_workers = 4
        self.data_num_workers = 12
        # /home/caoliwei/miniconda3/envs/yolov5/lib/python3.7/site-packages/torch/utils/data/dataloader.py:481:
        # UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker
        # in current system is 12, which is smaller than what this DataLoader is going to create.
        # Please be aware that excessive worker creation might get DataLoader running slow or even freeze,
        # lower the worker number to avoid potential slowness/freeze if necessary. cpuset_checked))

        self.input_size = (640, 640)  # (height, width)
        # self.input_size = (1504, 1504)  # (height, width)


        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # self.data_dir = None
        self.data_dir = '/home/caoliwei/Dataset/visdrone'

        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"
        self.train_ann = "train.json"
        self.val_ann = "test.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        # self.warmup_epochs = 5
        self.warmup_epochs = 2
        #self.max_epoch = 300
        #self.max_epoch = 40
        self.max_epoch = 100
        # self.max_epoch = 16
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0

        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        #self.eval_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        #self.test_size = (640, 640)
        self.test_size = self.input_size  # clw modify
        self.test_conf = 0.01
        self.nmsthre = 0.65