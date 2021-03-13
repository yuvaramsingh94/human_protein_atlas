import random 
import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

class hpa_dataset(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8):
        self.main_df = main_df
        self.aug_per = aug_per
        self.augmentation = augmentation
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]

        target_vec = info[self.label_col].values

        hdf5_path = os.path.join(self.path,f'{ids}.hdf5')
        print(hdf5_path)
        hdf5_file = h5py.File(hdf5_path,"r")
        train_x = hdf5_file['train_img'][...]
        print(train_x.shape)
        hdf5_file.close()
        print(train_x.shape)
        #check for the cell count 
        cell_count = train_x.shape[0]

        if cell_count == self.cells_used:
            train_img = train_x
        elif cell_count > self.cells_used:#random downsample
            rand_idx = [i for i in range(0,cell_count)]
            print('random idx ', rand_idx)
            random.shuffle(rand_idx)
            print('random idx ', rand_idx)
            train_img = train_x[rand_idx[:self.cells_used],:,:,:]
            print(train_img.shape)

        elif cell_count < self.cells_used:# add zero images
            shape = (self.cells_used - cell_count, train_x.shape[1], train_x.shape[2], train_x.shape[3])
            zero_arr = np.zeros(shape, dtype=float)
            train_img = np.concatenate([train_x, zero_arr], axis=0)

        return {'image' : train_img, 'label' : target_vec}





