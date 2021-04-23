import random 
import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
from sklearn.metrics import f1_score, roc_auc_score
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import math
#from torchvision import transforms
import albumentations as albu

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

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

class hpa_dataset_v1(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, label_smoothing = False, l_alp = 0.3, is_validation = False, size = 224, cell_repetition = True):
        self.main_df = main_df
        self.aug_percent = aug_per
        self.augmentation = augmentation
        self.float_conv = albu.Compose([albu.ToFloat(max_value=255.,always_apply=True)])
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path
        self.size = size
        self.is_validation = is_validation
        self.label_smoothing = label_smoothing
        self.l_alp = l_alp
        self.cell_repetition = cell_repetition
        #self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]

        target_vec = info[self.label_col].values.astype(np.int)

        # lets begin
        cell_count = len(os.listdir(os.path.join(self.path,f'{ids}')))
        
        if not self.is_validation:

            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    #print('this is ',i)
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    #print(hdf5_path)
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample

                rand_idx = [i for i in range(1,cell_count)]
                #print('random idx ', rand_idx)
                random.shuffle(rand_idx)
                #print('random idx ', rand_idx, self.cells_used)
                cell_list = []
                for i in rand_idx[:self.cells_used ]:
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            ##print('augmentation')
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                ##print('in the less class')
                cell_list = []
                for i in range(1, cell_count):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)

                if self.cell_repetition:
                    cell_count_lis = [i for i in range(1, cell_count)]
                    for i in range(self.cells_used - cell_count + 1):
                        cell_choice = random.choice(cell_count_lis)
                        hdf5_path = os.path.join(self.path,ids,f'{ids}_{cell_choice}.hdf5')
                        with h5py.File(hdf5_path,"r") as h:
                            vv = h['train_img'][...]
                            if random.random() < self.aug_percent:
                                vv = self.augmentation(image= vv)["image"]
                            else:
                                vv = self.float_conv(image= vv)["image"]
                            rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                            #print('this is rf ', rf)
                            rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                            vv = np.dstack([vv,rf_np])
                            #print('this is vv shape ',vv.shape)
                            cell_list.append(vv)

                    train_img = np.array(cell_list)
                else:
                    train_img = np.array(cell_list)
                    shape = (self.cells_used - cell_count + 1, self.size, self.size, 5)
                    zero_arr = np.zeros(shape, dtype=float)
                    train_img = np.concatenate([train_img, zero_arr], axis=0)
                    target_vec[-1] = 1# as we are adding black img . negative = 1 also
                #print('black ',target_vec)

            #### label smoothening
            if self.label_smoothing:
                #print('sm')
                #if target_vec.sum() == 1:
                    #print('sum is one ',target_vec)
                #    target_vec[-1] = 1
                target_vec = (1. - self.l_alp) * target_vec + self.l_alp / 19
                #target_vec[-1] = 1
                #print(target_vec)
        else:
            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                ##print('in the less class')
                cell_list = []
                for i in range(1, cell_count):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)

                if self.cell_repetition:
                    cell_count_lis = [i for i in range(1, cell_count)]
                    for i in range(self.cells_used - cell_count + 1):
                        cell_choice = random.choice(cell_count_lis)
                        hdf5_path = os.path.join(self.path,ids,f'{ids}_{cell_choice}.hdf5')
                        with h5py.File(hdf5_path,"r") as h:
                            vv = h['train_img'][...]
                            if random.random() < self.aug_percent:
                                vv = self.augmentation(image= vv)["image"]
                            else:
                                vv = self.float_conv(image= vv)["image"]
                            rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                            #print('this is rf ', rf)
                            rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                            vv = np.dstack([vv,rf_np])
                            #print('this is vv shape ',vv.shape)
                            cell_list.append(vv)

                    train_img = np.array(cell_list)
                else:
                    train_img = np.array(cell_list)
                    shape = (self.cells_used - cell_count + 1, self.size, self.size, 5)
                    zero_arr = np.zeros(shape, dtype=float)
                    train_img = np.concatenate([train_img, zero_arr], axis=0)
                    target_vec[-1] = 1# as we are adding black img . negative = 1 also
                #print('black ',target_vec)


            #if self.label_smoothing:
                #print('sm')
            #    if target_vec.sum() == 1:
                    #print('sum is one ',target_vec)
            #        target_vec[-1] = 0.7
        #print('this is the shape ', train_img.shape)
        #print("{} seconds".format(end_time-start_time))
        #return {'image' : torch.from_numpy(train_img), 'label' : torch.from_numpy(target_vec)}
        return {'image' : train_img, 'label' : target_vec}

#here we are going to resample cells if the cell count is higher thatn the requirment
class hpa_dataset_v2(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, label_smoothing = False, l_alp = 0.3, is_validation = False):
        self.main_df = main_df
        self.aug_percent = aug_per
        self.augmentation = augmentation
        self.float_conv = albu.Compose([albu.ToFloat(max_value=255.,always_apply=True)])
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path
        self.is_validation = is_validation
        self.label_smoothing = label_smoothing
        self.l_alp = l_alp

        #self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]

        target_vec = info[self.label_col].values.astype(np.int)

        # lets begin
        cell_count = len(os.listdir(os.path.join(self.path,f'{ids}')))
        
        if not self.is_validation:

            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                #print('lead')
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            ##print('augmentation')
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                ##print('in the less class')
                cell_list = []
                cell_count_list = [i for i in range(1,cell_count+1)]
                resampled_cell_list = [random.choice(cell_count_list) for i in range(self.cells_used)]
                #print(resampled_cell_list)
                for i in resampled_cell_list:
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                
                #print('black ',target_vec)

            #### label smoothening
            if self.label_smoothing:
                #print('sm')
                #if target_vec.sum() == 1:
                    #print('sum is one ',target_vec)
                #    target_vec[-1] = 1
                target_vec = (1. - self.l_alp) * target_vec + self.l_alp / 19
                #target_vec[-1] = 1
                #print(target_vec)
        else:
            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                ##print('in the less class')
                cell_list = []
                
            
                count = 1
                for i in range (1,self.cells_used+1):
                    #print(i,count)
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{count}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)

                    if i%cell_count ==0:
                        count = 1
                    else:
                        count += 1
                train_img = np.array(cell_list)

            #if self.label_smoothing:
                #print('sm')
            #    if target_vec.sum() == 1:
                    #print('sum is one ',target_vec)
            #        target_vec[-1] = 0.7
        #print('this is the shape ', train_img.shape)
        #print("{} seconds".format(end_time-start_time))
        #return {'image' : torch.from_numpy(train_img), 'label' : torch.from_numpy(target_vec)}
        return {'image' : train_img, 'label' : target_vec}


class hpa_dataset_v3(data.Dataset):
    def __init__(self, main_df, axe_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, label_smoothing = False, l_alp = 0.3, is_validation = False):
        self.main_df = main_df
        self.axe_df = axe_df
        self.aug_percent = aug_per
        self.augmentation = augmentation
        self.float_conv = albu.Compose([albu.ToFloat(max_value=255.,always_apply=True)])
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path
        self.is_validation = is_validation
        self.label_smoothing = label_smoothing
        self.l_alp = l_alp

        #self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]

        target_vec = info[self.label_col].values.astype(np.int)

        # lets begin
        cell_count = len(os.listdir(os.path.join(self.path,f'{ids}')))
        
        if not self.is_validation:

            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                #print('lead')

                #if its multilabel
                #there is still a problem with thos approach . with replace True . lower samples gets repeated
                if target_vec.sum() > 1:
                    #print('filtered')
                    count_list = self.axe_df[self.axe_df['ID'].isin([ids])].groupby('cluster').apply(lambda x: x.sample(self.cells_used,replace=True))['count'].values
                    #print(count_list)
                    random.shuffle(count_list) 
                    count_list = count_list[:self.cells_used] 
                else:
                    count_list = [i for i in range(1, self.cells_used + 1)]
                    random.shuffle(count_list) 
                    count_list = count_list[:self.cells_used]                 

                
                #print(len(count_list))
                cell_list = []
                for i in count_list:
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            ##print('augmentation')
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                ##print('in the less class')
                cell_list = []
                cell_count_list = [i for i in range(1,cell_count+1)]
                resampled_cell_list = [random.choice(cell_count_list) for i in range(self.cells_used)]
                #print(resampled_cell_list)
                for i in resampled_cell_list:
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                
                #print('black ',target_vec)

            #### label smoothening
            if self.label_smoothing:
                #print('sm')
                #if target_vec.sum() == 1:
                    #print('sum is one ',target_vec)
                #    target_vec[-1] = 1
                target_vec = (1. - self.l_alp) * target_vec + self.l_alp / 19
                #target_vec[-1] = 1
                #print(target_vec)
        else:
            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                ##print('in the less class')
                cell_list = []
                
            
                count = 1
                for i in range (1,self.cells_used+1):
                    #print(i,count)
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{count}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        else:
                            vv = self.float_conv(image= vv)["image"]
                        rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                        #print('this is rf ', rf)
                        rf_np = np.full(shape = (224,224), fill_value = rf)
                        vv = np.dstack([vv,rf_np])
                        #print('this is vv shape ',vv.shape)
                        cell_list.append(vv)

                    if i%cell_count ==0:
                        count = 1
                    else:
                        count += 1
                train_img = np.array(cell_list)

            #if self.label_smoothing:
                #print('sm')
            #    if target_vec.sum() == 1:
                    #print('sum is one ',target_vec)
            #        target_vec[-1] = 0.7
        #print('this is the shape ', train_img.shape)
        #print("{} seconds".format(end_time-start_time))
        #return {'image' : torch.from_numpy(train_img), 'label' : torch.from_numpy(target_vec)}
        return {'image' : train_img, 'label' : target_vec}



def score_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    ROC_AUC_score = roc_auc_score(labels, preds, average='micro')
    preds = preds > 0.5
    F1_score = f1_score(labels, preds, average='micro')
    

    return {'AUROC':ROC_AUC_score,'F1_score':F1_score}



def lsep_loss_stable(input, target, average=True):

    n = input.size(0)

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)

    max_difference, index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower

    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

    if average:
        return lsep.mean()
    else:
        return lsep



class ImprovedPANNsLoss(nn.Module):
    def __init__(self, device,  weights=[1, 0.5]):
        super().__init__()

        self.normal_loss = focal_loss(alpha=0.1, gamma=5).cuda(device=device)

        self.bce = focal_loss(alpha=0.1, gamma=5).cuda(device=device)
        self.weights = weights

    def forward(self, input, target):
        input_ = input['final_output']
        target = target.float()

        framewise_output = input["cell_pred"]
        #print('fw ', framewise_output.shape)
        clipwise_output_with_max, _ = framewise_output.max(dim=-1)

        normal_loss = self.normal_loss(input_, target)
        auxiliary_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction = 'mean'):
        super(focal_loss, self).__init__()
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        preds,
        targets,):
        bce_loss = self.loss_fct(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5, self.alpha * (1. - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            pass

        return loss

#https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr








