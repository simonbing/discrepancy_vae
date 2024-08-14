import os
import argparse
from argparse import Namespace
import time
import json
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from crc.baselines.discrepancy_vae.src.train import train
from crc.baselines.discrepancy_vae.src.utils import get_chamber_data, SCDATA_sampler

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
        # Need this to prevent overflow during training
        torch.multiprocessing.set_sharing_strategy('file_system')

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

        dataset_train, dataset_test, dim, cdim, ptb_targets, \
            iv_name_train, iv_name_test = get_chamber_data(chamber_data=self.dataset,
                                                           experiment=self.experiment,
                                                           batch_size=opts.batch_size,
                                                           mode=opts.mode)

        dataloader_train = DataLoader(dataset_train,
                                      batch_sampler=SCDATA_sampler(
                                          dataset_train,
                                          opts.batch_size,
                                          iv_name_train),
                                      num_workers=0)
        # TODO: move this to eval code
        dataloader_test = DataLoader(dataset_test,
                                     batch_sampler=SCDATA_sampler(
                                         dataset_test,
                                         opts.batch_size,
                                         iv_name_test),
                                     num_workers=0
                                     )

        opts.dim = dim
        if opts.latdim is None:
            opts.latdim = cdim
        opts.cdim = cdim

        # Save training metadata
        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path):
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)

        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        if not os.path.exists(test_data_path):
            with open(test_data_path, 'wb') as f:
                pickle.dump(dataset_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.train_dir, 'config.json'), 'w') as f:
            json.dump(opts.__dict__, f, indent=4)

        train(dataloader_train, opts, device, self.train_dir, image_data=True, log=True)


class EvalCMVAE(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        G = dataset_test.dataset.G
        G_hat = self.trained_model.G.cpu().detach().numpy()

        return G, G_hat

    def get_encodings(self, dataset_test, batch_size=500):
        # Make dataloader for test data
        dataloader_test = DataLoader(dataset_test,
                                     batch_sampler=SCDATA_sampler(
                                         dataset_test,
                                         batch_size,
                                         dataset_test.dataset.iv_names[dataset_test.indices]),
                                     num_workers=0,
                                     shuffle=False)

        self.trained_model.eval()

        z_list = []
        z_hat_list = []
        # Iterate over test dataloader and encode all samples and save gt data
        for X in dataloader_test:
            x = X[0]
            z = X[3]

            x = x.to(self.device)

            mu, var = self.trained_model.encode(x)
            z_hat = self.trained_model.reparametrize(mu, var)

            z_list.append(z)
            z_hat_list.append(z_hat)

        z = torch.cat(z_list).cpu().detach().numpy()
        z_hat = torch.cat(z_hat_list).cpu().detach().numpy()

        return z, z_hat
