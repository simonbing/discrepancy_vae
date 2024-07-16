import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# import scanpy as sc

from causalchamber.datasets import Dataset as ChamberData
from skimage import io


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
    def __init__(self, data_root='/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/data/chamber_downloads'):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.chamber_data_name = 'lt_camera_v1'
        chamber_data = ChamberData(self.chamber_data_name, root=self.data_root, download=True)
        # Observational data
        obs_data = chamber_data.get_experiment(name='scm_1_reference').as_pandas_dataframe()
        # Interventional data
        iv_data_1 = chamber_data.get_experiment(name='scm_1_red').as_pandas_dataframe()
        iv_data_2 = chamber_data.get_experiment(name='scm_1_green').as_pandas_dataframe()
        iv_data_3 = chamber_data.get_experiment(name='scm_1_blue').as_pandas_dataframe()
        iv_data_4 = chamber_data.get_experiment(name='scm_1_pol_1').as_pandas_dataframe()
        iv_data_5 = chamber_data.get_experiment(name='scm_1_pol_2').as_pandas_dataframe()
        iv_data_list = [iv_data_1, iv_data_2, iv_data_3, iv_data_4, iv_data_5]
        # Get one big df for all iv data
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
        self.iv_targets = ['0','1','2','3','4'] # hardcoded for now, need this name
        self.iv_ids = map_ptb_features(self.iv_targets, self.iv_names)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Observational sample
        obs_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                    'scm_1_reference',
                                    'images_100',
                                    self.obs_data['image_file'].iloc[item])
        obs_sample = io.imread(obs_img_name)
        # Interventional sample
        iv_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                   _map_iv_envs(self.iv_names[item]),
                                   'images_100',
                                   self.iv_data['image_file'].iloc[item])
        iv_sample = io.imread(iv_img_name)
        # One-hot intervention label
        c = self.iv_ids[item]

        return obs_sample, iv_sample, c

    def __len__(self):
        return len(self.obs_data)


def map_ptb_features(all_ptb_targets, ptb_ids):
    ptb_features = []
    for id in ptb_ids:
        feature = np.zeros(all_ptb_targets.__len__())
        feature[[all_ptb_targets.index(i) for i in id.split(',')]] = 1
        ptb_features.append(feature)
    return np.vstack(ptb_features)


def _map_iv_envs(idx):
    idx = int(idx)
    map = ['scm_1_red', 'scm_1_green', 'scm_1_blue', 'scm_1_pol_1', 'scm_1_pol_2']

    return map[idx]

