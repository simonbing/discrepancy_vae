import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# import scanpy as sc

from causalchamber.datasets import Dataset as ChamberData
from skimage import io, transform

from crc.utils import get_task_environments


# read the norman dataset.
# map the target genes of each cell to a binary vector, using a target gene list "perturb_targets".
# "perturb_type" specifies whether the returned object contains single trarget-gene samples, double target-gene samples, or both.
# class SCDataset(Dataset):
#     def __init__(self, datafile='/home/jzhang/discrepancy_vae/identifiable_causal_vae/data/datasets/Norman2019_raw.h5ad', perturb_type='single', perturb_targets=None):
#         super(Dataset, self).__init__()
#         assert perturb_type in ['single', 'double', 'both'], 'perturb_type not supported!'
#
#         adata = sc.read_h5ad(datafile)
#
#         if perturb_targets is None:
#             ptb_targets = list(set().union(*[set(i.split(',')) for i in adata.obs['guide_ids'].value_counts().index]))
#             ptb_targets.remove('')
#         else:
#             ptb_targets = perturb_targets
#         self.ptb_targets = ptb_targets
#
#         if perturb_type == 'single':
#             ptb_adata = adata[(~adata.obs['guide_ids'].str.contains(',')) & (adata.obs['guide_ids']!='')].copy()
#             self.ptb_samples = ptb_adata.X
#             self.ptb_names = ptb_adata.obs['guide_ids'].values
#             self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
#             del ptb_adata
#         elif perturb_type == 'double':
#             ptb_adata = adata[adata.obs['guide_ids'].str.contains(',')].copy()
#             self.ptb_samples = ptb_adata.X
#             self.ptb_names = ptb_adata.obs['guide_ids'].values
#             self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
#             del ptb_adata
#         else:
#             ptb_adata = adata[adata.obs['guide_ids']!=''].copy()
#             self.ptb_samples = ptb_adata.X
#             self.ptb_names = ptb_adata.obs['guide_ids'].values
#             self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
#             del ptb_adata
#
#         self.ctrl_samples = adata[adata.obs['guide_ids']==''].X.copy()
#         self.rand_ctrl_samples = self.ctrl_samples[
#             np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
#             ]
#         del adata
#
#     def __getitem__(self, item):
#         x = torch.from_numpy(self.rand_ctrl_samples[item].toarray().flatten()).double()
#         y = torch.from_numpy(self.ptb_samples[item].toarray().flatten()).double()
#         c = torch.from_numpy(self.ptb_ids[item]).double()
#         return x, y, c
#
#     def __len__(self):
#         return self.ptb_samples.shape[0]


# read simulation dataset
class SimuDataset(Dataset):
    def __init__(self, datafile='../data/simulation/data_1.pkl', perturb_type='single', perturb_targets=None):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double'], 'perturb_type not supported!'

        with open(datafile, 'rb') as f:
            dataset = pickle.load(f)

        if perturb_targets is None:
            ptb_targets = dataset['ptb_targets']
        else:
            ptb_targets = perturb_targets
        self.ptb_targets = ptb_targets

        
        ptb_data = dataset[perturb_type]
        self.ctrl_samples = ptb_data['X']
        self.ptb_samples = ptb_data['Xc']
        self.ptb_names = np.array(ptb_data['ptbs'])
        self.ptb_ids = map_ptb_features(ptb_targets, ptb_data['ptbs'])
        del ptb_data 

        self.nonlinear = dataset['nonlinear']
        del dataset

    def __getitem__(self, item):
        x = torch.from_numpy(self.ctrl_samples[item].flatten()).double()
        y = torch.from_numpy(self.ptb_samples[item].flatten()).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        return x, y, c
    
    def __len__(self):
        return self.ptb_samples.shape[0]


class ChamberDataset(Dataset):
    def __init__(self, dataset, task,
                 data_root='/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/data/chamber_downloads',
                 transform=None,
                 eval=False):
        super(Dataset, self).__init__()
        self.eval = eval

        self.transform = transform

        self.data_root = data_root
        self.chamber_data_name = dataset
        self.exp, self.env_list, self.features = get_task_environments(task)
        chamber_data = ChamberData(self.chamber_data_name, root=self.data_root, download=True)
        # Observational data
        obs_data = chamber_data.get_experiment(name=f'{self.exp}_reference').as_pandas_dataframe()
        # Interventional data
        iv_data_list = [chamber_data.get_experiment(name=f'{self.exp}_{env}').as_pandas_dataframe() for env in self.env_list]

        self.iv_data = pd.concat(iv_data_list)

        # Generate intervention index list
        iv_names = []
        for idx, iv_data in enumerate(iv_data_list):
            iv_names.append(np.repeat(f'{idx}', len(iv_data)))
        self.iv_names = np.concatenate(iv_names)

        # Resample observational data to have same nr of samples as iv_data
        self.obs_data = obs_data.loc[np.random.choice(len(obs_data),
                                                      size=len(self.iv_data),
                                                      replace=True), :]

        # Get one-hot encoding of iv environments
        self.iv_targets = [str(elem) for elem in np.arange(len(self.env_list))] # hardcoded for now, need this name
        self.iv_ids = map_ptb_features(self.iv_targets, self.iv_names)

        # Get ground truth adjacency matrix
        if self.exp in ['scm_1', 'scm_2']:
            # TODO: probably need to follow some convention of making this upper triang
            self.G = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Observational sample
        obs_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                    f'{self.exp}_reference',
                                    'images_64',
                                    self.obs_data['image_file'].iloc[item])
        obs_sample = io.imread(obs_img_name)
        # Interventional sample
        iv_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                   _map_iv_envs(self.iv_names[item], self.exp, self.env_list),
                                   'images_64',
                                   self.iv_data['image_file'].iloc[item])
        iv_sample = io.imread(iv_img_name)

        # Normalize inputs
        obs_sample = obs_sample / 255.0
        iv_sample = iv_sample / 255.0

        # One-hot intervention label
        c = self.iv_ids[item]

        if self.transform:
            obs_sample = self.transform(obs_sample)
            iv_sample = self.transform(iv_sample)

        if not self.eval:
            return torch.as_tensor(obs_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32),  \
                torch.as_tensor(iv_sample.transpose((2, 0, 1)),
                                dtype=torch.float32), \
                torch.as_tensor(c, dtype=torch.float32)
        else:  # also return the ground truth variables
            Z_obs = self.obs_data[self.features].iloc[item].to_numpy()
            Z_iv = self.iv_data[self.features].iloc[item].to_numpy()
            return torch.as_tensor(obs_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32),  \
                torch.as_tensor(iv_sample.transpose((2, 0, 1)),
                                dtype=torch.float32), \
                torch.as_tensor(c, dtype=torch.float32), \
                Z_obs, Z_iv

    def __len__(self):
        return len(self.obs_data)


def map_ptb_features(all_ptb_targets, ptb_ids):
    ptb_features = []
    for id in ptb_ids:
        feature = np.zeros(all_ptb_targets.__len__())
        feature[[all_ptb_targets.index(i) for i in id.split(',')]] = 1
        ptb_features.append(feature)
    return np.vstack(ptb_features)


def _map_iv_envs(idx, exp, env_list):
    idx = int(idx)
    map = [f'{exp}_{env}' for env in env_list]

    return map[idx]
