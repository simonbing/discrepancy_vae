import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler
from skimage import transform

from dataset import SimuDataset, ChamberDataset


## MMD LOSS
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # Flatten source and targets
        source = torch.flatten(source, start_dim=1)
        target = torch.flatten(target, start_dim=1)

        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


# def get_data(batch_size=32, mode='train', perturb_targets=None):
#     assert mode in ['train', 'test'], 'mode not supported!'
#
#     if mode == 'train':
#         dataset = SCDataset(perturb_type='single', perturb_targets=perturb_targets)
#         train_idx, test_idx = split_scdata(
#             dataset,
#             split_ptbs=['KLF1', 'BAK1', 'CEBPE', 'UBASH3B', 'ETS2', 'OSR2', 'SLC4A1','SET', 'ELMSAN1', 'MAP2K6', 'FOXF1', 'C19orf26', 'FOXA1','UBASH3A'],
#             batch_size=batch_size
#         ) # leave out some cells from the top 14 single target-gene interventions
#     elif mode == 'test':
#         assert perturb_targets is not None, 'perturb_targets has to be specified during testing, otherwise the index might be mismatched!'
#         dataset = SCDataset(perturb_type='double', perturb_targets=perturb_targets)
#
#     ptb_genes = dataset.ptb_targets
#
#     if mode == 'train':
#         dataset1 = Subset(dataset, train_idx)
#         ptb_name = dataset.ptb_names[train_idx]
#         dataloader = DataLoader(
#             dataset1,
#             batch_sampler=SCDATA_sampler(dataset1, batch_size, ptb_name),
#             num_workers=0
#         )
#
#         dim = dataset[0][0].shape[0]
#         cdim = dataset[0][2].shape[0]
#
#         dataset2 = Subset(dataset, test_idx)
#         ptb_name = dataset.ptb_names[test_idx]
#         dataloader2 = DataLoader(
#             dataset2,
#             batch_sampler=SCDATA_sampler(dataset2, batch_size, ptb_name),
#             num_workers=0
#         )
#         return dataloader, dataloader2, dim, cdim, ptb_genes
#     else:
#         dataloader = DataLoader(
#             dataset,
#             batch_sampler=SCDATA_sampler(dataset, batch_size),
#             num_workers=0
#         )
#
#         dim = dataset[0][0].shape[0]
#         cdim = dataset[0][2].shape[0]
#
#         return dataloader, dim, cdim, ptb_genes


def get_simu_data(batch_size=32, mode='train', perturb_targets=None):
    assert mode in ['train', 'test'], 'mode not supported!'

    if mode == 'train':
        dataset = SimuDataset(perturb_type='single', perturb_targets=perturb_targets)
        train_idx, test_idx = split_simudata(
            dataset,
            batch_size=batch_size
        ) # leave out some cells from the top 14 single target-gene interventions
    elif mode == 'test':
        assert perturb_targets is not None, 'perturb_targets has to be specified during testing, otherwise the index might be mismatched!'
        dataset = SimuDataset(perturb_type='double', perturb_targets=perturb_targets)

    ptb_genes = dataset.ptb_targets
        
    if mode == 'train':
        dataset1 = Subset(dataset, train_idx)
        ptb_name = dataset.ptb_names[train_idx]
        dataloader = DataLoader(
            dataset1,
            batch_sampler=SCDATA_sampler(dataset1, batch_size, ptb_name),
            num_workers=0
        )

        dim = dataset[0][0].shape[0]
        cdim = dataset[0][2].shape[0]

        dataset2 = Subset(dataset, test_idx)
        ptb_name = dataset.ptb_names[test_idx]
        dataloader2 = DataLoader(
            dataset2,
            batch_sampler=SCDATA_sampler(dataset2, batch_size, ptb_name),
            num_workers=0
        )
        return dataloader, dataloader2, dim, cdim, ptb_genes, dataset.nonlinear
    else:	
        dataloader = DataLoader(
            dataset,
            batch_sampler=SCDATA_sampler(dataset, batch_size),
            num_workers=0
        )
        
        dim = dataset[0][0].shape[0]
        cdim = dataset[0][2].shape[0]

        return dataloader, dim, cdim, ptb_genes, dataset.nonlinear


def get_chamber_data(batch_size=32, mode='train'):
    assert mode in ['train', 'test'], 'mode not supported!'

    if mode == 'train':
        # Define custom torch dataset for chambers data
        dataset = ChamberDataset(transform=Rescale(64))
        # Split train test
        train_idx, test_idx = split_chamberdata(dataset, batch_size=batch_size)
        # TODO take care of the rest of splitting as above
        dataset_train = Subset(dataset, train_idx)
        iv_name_train = dataset.iv_names[train_idx]
        dataloader_train = DataLoader(dataset_train,
                                      batch_sampler=SCDATA_sampler(dataset_train,
                                                                   batch_size,
                                                                   iv_name_train),
                                      num_workers=0)

        dataset_test = Subset(dataset, test_idx)
        iv_name_test = dataset.iv_names[test_idx]
        dataloader_test = DataLoader(dataset_test,
                                     batch_sampler=SCDATA_sampler(dataset_test,
                                                                  batch_size,
                                                                  iv_name_test),
                                     num_workers=0)

        dim = dataset[0][0].shape # TODO: see where this is used in training
        cdim = dataset[0][2].shape[0]

        return dataloader_train, dataloader_test, dim, cdim, dataset.iv_targets


def split_simudata(simudataset, batch_size=32):
    num_sample = 1024

    test_idx = []
    train_idx = []
    from tqdm import tqdm
    for ptb in tqdm(simudataset.ptb_targets):
        idx = list(np.where(simudataset.ptb_names == ptb)[0])
        test_idx.append(idx[0:num_sample])
        train_idx.append(idx[num_sample:2*num_sample])

    test_idx = list(np.hstack(test_idx))
    train_idx = list(np.hstack(train_idx)) 
    
    return train_idx, test_idx


def split_chamberdata(dataset, batch_size=32):
    num_sample = 4000 # seems to be train_samples PER environment

    # test_idx = []
    train_idx = []
    from tqdm import tqdm
    for iv in tqdm(dataset.iv_targets):
        idx = list(np.where(dataset.iv_names == iv)[0])
        # test_idx.append(idx[0:num_sample])
        train_idx.append(idx[0:num_sample])

    # test_idx = list(np.hstack(test_idx))
    train_idx = list(np.hstack(train_idx))
    test_idx = [l for l in range(len(dataset)) if l not in train_idx]

    return train_idx, test_idx


# leave out some cells from the split_ptbs
def split_scdata(scdataset, split_ptbs, batch_size=32):
    num_batch = 96 // batch_size
    num_sample = num_batch * batch_size

    test_idx = []
    for ptb in split_ptbs:
        idx = list(np.where(scdataset.ptb_names == ptb)[0])
        test_idx.append(idx[0:num_sample])

    test_idx = list(np.hstack(test_idx))
    train_idx = [l for l in range(len(scdataset)) if l not in test_idx]
    
    return train_idx, test_idx


# a special batch sampler that groups only cells from the same interventional distribution into a batch
class SCDATA_sampler(Sampler):
    def __init__(self, scdataset, batchsize, ptb_name=None):
        self.intervindices = []
        self.len = 0
        if ptb_name is None:
            ptb_name = scdataset.ptb_names
        for ptb in set(ptb_name):
            idx = np.where(ptb_name == ptb)[0]
            self.intervindices.append(idx)
            self.len += len(idx) // batchsize
        self.batchsize = batchsize
    
    def __iter__(self):
        comb = []
        for i in range(len(self.intervindices)):
            random.shuffle(self.intervindices[i])
        
            interv_batches = chunk(self.intervindices[i], self.batchsize)
            if interv_batches:
                comb += interv_batches

        combined = [batch.tolist() for batch in comb]
        random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return self.len


def chunk(indices, chunk_size):
    split = torch.split(torch.tensor(indices), chunk_size)
    
    if len(indices) % chunk_size == 0:
        return split
    elif len(split) > 0:
        return split[:-1]
    else:
        return None


class Rescale(object):
    """
    Adapted from pytorch tutorial.

    Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        # image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
