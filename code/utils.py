import random 
import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
from sklearn.metrics import f1_score, roc_auc_score
#from torchvision import transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

class hpa_dataset_v1(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, is_validation = False):
        self.main_df = main_df
        self.aug_per = aug_per
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

        hdf5_path = os.path.join(self.path,f'{ids}.hdf5')
        #print(hdf5_path)
        hdf5_file = h5py.File(hdf5_path,"r")
        train_x = hdf5_file['train_img'][...]
        #print(train_x.shape)
        hdf5_file.close()
        #print(train_x.shape)
        #check for the cell count 
        cell_count = train_x.shape[0]

        if cell_count == self.cells_used:
            train_img = train_x
        elif cell_count > self.cells_used:#random downsample
            if not self.is_validation:
                rand_idx = [i for i in range(0,cell_count)]
                #print('random idx ', rand_idx)
                random.shuffle(rand_idx)
                #print('random idx ', rand_idx)
                train_img = train_x[rand_idx[:self.cells_used],:,:,:]
                #print(train_img.shape)
            else:
                train_img = train_x[:self.cells_used,:,:,:]#for now just taking top counts.

        elif cell_count < self.cells_used:# add zero images
            shape = (self.cells_used - cell_count, train_x.shape[1], train_x.shape[2], train_x.shape[3])
            zero_arr = np.zeros(shape, dtype=float)
            train_img = np.concatenate([train_x, zero_arr], axis=0)
        #print(f'train img details: type {type(train_img)} shape {train_img.shape}')
        #print(f'target_vec details: type {type(target_vec)} shape {target_vec.shape}')

        #v = torch.from_numpy(target_vec)


        return {'image' : torch.from_numpy(train_img), 'label' : torch.from_numpy(target_vec)}

def score_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    ROC_AUC_score = roc_auc_score(labels, preds, average='micro')
    preds = preds > 0.5
    F1_score = f1_score(labels, preds, average='micro')
    

    return {'AUROC':ROC_AUC_score,'F1_score':F1_score}





