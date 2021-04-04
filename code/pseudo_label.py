#main idea here is to create a per cell pseudo label 
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import imageio

import imageio
import os
import h5py
from skimage.transform import resize
from model import HpaModel, HpaModel_1, HpaModel_2
import torch
import torch.utils.data as data
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from pycocotools import _mask as coco_mask


### we need to create a Dataframe which will have all cell list to make iteration easy
#note here we are using the extracted train df . but there is a big one also

FOLDS = 3
n_classes = 19
train_base_df = pd.read_csv('data/train_fold_v6.csv')

class hpa_dataset(data.Dataset):
    def __init__(self, cell_list, path):
        self.cell_list = cell_list
        self.path = path
    def __len__(self):
        return len(self.cell_list)
    
    def __getitem__(self, idx):
        cell = self.cell_list[idx]
        

        hdf5_path = os.path.join(self.path,cell)
        with h5py.File(hdf5_path,"r") as h:
            vv = h['test_img'][...]
            rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
            #print('this is rf ', rf)
            rf_np = np.full(shape = (224,224), fill_value = rf)
            vv = np.dstack([vv,rf_np])
        return { 'image':vv}

def model_prediction(model, X):
    #print(X.shape)
    pred_0 = model(X)['sigmoid_output']
    
    #torch.Size([1, 8, 5, 224, 224])
    X_up_down = torch.flip(X,[3])
    #print('X_up_down ',X_up_down.shape)
    pred_3 = model(X_up_down)['sigmoid_output']
    #torch.Size([1, 8, 5, 224, 224])
    X_right_left = torch.flip(X,[4])
    #print('X_right_left ',X_right_left.shape)
    pred_6 = model(X_right_left)['sigmoid_output']
    #torch.Size([1, 8, 5, 224, 224])
    X_right_left_up_down = torch.flip(X_right_left,[3])
    #print('X_right_left_up_down ',X_right_left_up_down.shape)
    pred_9 = model(X_right_left_up_down)['sigmoid_output']
    pred = (pred_3 + pred_6 + pred_9 )/3.
    #print('pred ',pred.shape)
    #pred = torch.clamp(pred * 1.5, min=0.0, max = 1.0)
    return pred



def processor(unique_id):
    ID_list = []
    prediction_list = []
    cell_full_list = []
    for uniq in tqdm(unique_id):
        cell_list = os.listdir(f'data/train_h5_224_30000_v3/{uniq}')
        ID_list.extend([uniq]*len(cell_list))
        cell_full_list.extend(cell_list)
        cell_dataset = hpa_dataset(cell_list = cell_list, 
                            path = f'data/train_h5_224_30000_v3/{uniq}')
        cell_dataloader = data.DataLoader(
            cell_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            drop_last=False,
            pin_memory=True, 
        )
        
        with torch.no_grad():
            for data_t in cell_dataloader:
                X = data_t['image']
                X = X.to(device, dtype=torch.float) #(cell_count , 224, 224, 4)
                X = X.unsqueeze(0).permute(0,1,4,2,3)
                pred = model_prediction(X)
                prediction_list.append(pred.detach().squeeze(0).cpu().numpy()) 
    predictions = np.concatenate(prediction_list, axis=0)

    pred_df = pd.DataFrame.from_dict({'ID':ID_list, 'cell':cell_full_list,})
    pred_df[[str(i) for i in range(n_classes)]] = predictions

    return pred_df
    
                



fold_df = []
for fold in FOLDS:

    valid_df = train_base_df[train_base_df['fold'] == fold]

    unique_id = valid_df["ID"].unique()

    fold_df.append(processor(unique_id))


