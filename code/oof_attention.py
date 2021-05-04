import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import imageio

import imageio
import os
import h5py
from skimage.transform import resize
from model import HpaModel, HpaModel_1
import torch
import torch.utils.data as data
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from pycocotools import _mask as coco_mask
from utils import hpa_dataset_v1
from torch.backends import cudnn
import albumentations as albu
cudnn.benchmarks = True

fold = 2
DATA_PATH = 'data/train_h5_256_40000_v6'
train_base_df = pd.read_csv('data/train_fold_v11.csv')
valid_df = train_base_df[train_base_df['fold'] == fold]
BATCH_SIZE = 16
WORKERS = 2
cells_used = 8
device = torch.device("cuda:0")
vees = 'v6_6_3_2'# actually v6_5_2_1_1
MODEL_PATH = f'weights/version_{vees}'
n_classes = 19
SIZE = 256
metric_use = 'loss'

# config_v1.ini
model_fold_0 = HpaModel_1(classes = n_classes, device = device, 
                        base_model_name = 'efficientnet-b4', features = 1792, feature_red = 512, pretrained = False)

model_fold_0.load_state_dict(torch.load(f"{MODEL_PATH}/fold_{fold}_seed_2/model_{metric_use}_{fold}.pth",map_location = device))
model_fold_0.to(device)
model_fold_0.eval()
float_conv = albu.Compose([albu.ToFloat(max_value=255.,always_apply=True)])
class hpa_dataset(data.Dataset):
    def __init__(self, hdf5_path, path, ids ):
        self.hdf5_path = hdf5_path
        self.path = path
        self.ids = ids

    def __len__(self):
        return len(self.hdf5_path)
    
    def __getitem__(self, idx):
        h5_path = self.hdf5_path[idx]
        ids = self.ids[idx]
        hdf5_path = os.path.join(self.path,ids,h5_path)
        with h5py.File(hdf5_path,"r") as h:
            vv = h['train_img'][...]
            rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
            #print('this is rf ', rf)
            rf_np = np.full(shape = (256, 256), fill_value = rf)
            vv = np.dstack([vv,rf_np])
        vv = float_conv(image= vv)["image"]
        return { 'image':vv, 'ids':ids,'count':hdf5_path.split('_')[-1].split('.')[0]}

'''
valid_dataset = hpa_dataset_v1(main_df = valid_df, path = DATA_PATH, cells_used = cells_used, is_validation = True, 
                                size = SIZE, cell_repetition = True)

valid_dataloader = data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )
'''


id_list = valid_df['ID'].values
ids_list = []
hdf5_path_list = []
for ids in id_list:

    hdf5_path = os.listdir(os.path.join(DATA_PATH,ids))
    ids_list.extend([ids]*len(hdf5_path))
    hdf5_path_list.extend(hdf5_path)
#print(hdf5_path)
valid_dataset = hpa_dataset(hdf5_path = hdf5_path_list, path = DATA_PATH, ids = ids_list)
valid_dataloader = data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    drop_last=False,
    pin_memory=True,
    
)

ids_final_list = []
count_final_list = []
argmax_list = []
attention_list = []
with torch.no_grad():
    for data_t in tqdm(valid_dataloader):
        
        X = data_t['image']#,data_t['label']
        ids = data_t['ids']
        count = data_t['count']
        X = X.to(device, dtype=torch.float)#shape (1,256,256,5)
        #print(X.shape)
        X = X.unsqueeze(0).permute(0,1,4,2,3)
        #print(X.shape)
        #print(X[:,:,:4,:,:].min(), X.max())
        
        with torch.cuda.amp.autocast():
            prediction = model_fold_0(X)
            attentions = prediction['norm_att'].permute(0,2,1)
        cell_pred = torch.sigmoid(prediction['cell_pred']).permute(0,2,1)
        #print(attentions.shape)
        #print('norm_att ',attentions )
        #print('norm_att classes',torch.argmax(attentions, dim=-1) )
        #print('cell_pred ',cell_pred)


        ## lets take the markers
        #print('ids ',ids.detach().cpu().numpy())
        #print('count ',count.detach().cpu().numpy())
        #print(ids)
        ids_final_list.extend(ids)
        count_final_list.extend(count)
        #print()
        argmax_list.extend(torch.argmax(attentions, dim=-1).detach().cpu().numpy()[0])
        #print('appr ',attentions.detach().cpu().numpy())
        #print('shape ',attentions.detach().cpu().numpy()[0].shape)
        attention_list.extend(attentions.detach().cpu().numpy()[0])
        #print('actual ',Y)
        #valid_loss = criterion(prediction, Y)

value_dict = {'ids':ids_final_list, 'count':count_final_list, 'argmax_list':argmax_list}

final_df = pd.DataFrame.from_dict(value_dict)
final_df[[str(i) for i in range(19)]] = attention_list

final_df.to_csv(f'data/oof/fold_{fold}.csv')



