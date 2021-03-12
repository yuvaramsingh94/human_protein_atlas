import random 
import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
from sklearn.metrics import f1_score, roc_auc_score
import time
#from torchvision import transforms

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
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, is_validation = False):
        self.main_df = main_df
        self.aug_percent = aug_per
        self.augmentation = augmentation
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path
        self.is_validation = is_validation

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
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample

                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            #print('augmentation')
                            vv = self.augmentation(image= vv)["image"]
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                #print('in the less class')
                cell_list = []
                for i in range(1, cell_count):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                shape = (self.cells_used - cell_count + 1, 224, 224, 4)
                zero_arr = np.zeros(shape, dtype=float)
                #print('zero_arr ',zero_arr.shape)
                #print('train_img ',train_img.shape)
                train_img = np.concatenate([train_img, zero_arr], axis=0)
                target_vec[-1] = 1# as we are adding black img . negative = 1 also
        else:
            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        cell_list.append(h['train_img'][...])
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        cell_list.append(h['train_img'][...])
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                #print('in the less class')
                cell_list = []
                for i in range(1, cell_count):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        cell_list.append(h['train_img'][...])
                train_img = np.array(cell_list)
                shape = (self.cells_used - cell_count + 1, 224, 224, 4)
                zero_arr = np.zeros(shape, dtype=float)
                #print('zero_arr ',zero_arr.shape)
                #print('train_img ',train_img.shape)
                train_img = np.concatenate([train_img, zero_arr], axis=0)
                target_vec[-1] = 1# as we are adding black img . negative = 1 also
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





