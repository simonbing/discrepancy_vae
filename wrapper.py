import os
import argparse
from argparse import Namespace
import time
import json
import pickle

import torch
import numpy as np
import random

from crc.baselines.discrepancy_vae.src.train import train
from crc.baselines.discrepancy_vae.src.utils import get_chamber_data

from crc.wrappers import TrainModel, EvalModel
from crc.utils import get_device


class TrainCMVAE(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        """
        Adapted from source code for "Identifiability Guarantees for Causal Disentanglement
        from Soft Interventions".
        """
        device = get_device()
        print(f'using device: {device}')

        opts = Namespace(
            batch_size=self.batch_size,
            mode='train',
            lr=1e-3,
            epochs=self.epochs,
            grad_clip=False,
            mxAlpha=10,
            mxBeta=2,
            mxTemp=5,
            lmbda=1e-3,
            MMD_sigma=1000,
            kernel_num=10,
            matched_IO=False,
            latdim=self.lat_dim,
            seed=self.seed
        )

        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        dataloader_train, dataloader_test, dim, cdim, ptb_targets = get_chamber_data(
            chamber_data=self.dataset, experiment=self.experiment,
            batch_size=opts.batch_size, mode=opts.mode)

        opts.dim = dim
        if opts.latdim is None:
            opts.latdim = cdim
        opts.cdim = cdim

        # Save training metadata
        train_data_path = os.path.join(self.model_dir, 'train_data.pkl')
        if not os.path.exists(train_data_path):
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataloader_train, f, protocol=pickle.HIGHEST_PROTOCOL)

        test_data_path = os.path.join(self.model_dir, 'test_data.pkl')
        if not os.path.exists(test_data_path):
            with open(test_data_path, 'wb') as f:
                pickle.dump(dataloader_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.train_dir, 'config.json'), 'w') as f:
            json.dump(opts.__dict__, f, indent=4)


        train(dataloader_train, opts, device, self.train_dir, image_data=True, log=True)
