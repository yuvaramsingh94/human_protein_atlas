import pandas as pd
import numpy as np
from model import HpaModel
import torch
import torch.utils.data as data
import h5py
from tqdm import tqdm
import os
import albumentations as albu
import matplotlib.pyplot as plt

main_df = pd.read_csv('data/cell_mask_study_30000.csv')
main_df['is_single'] = main_df[[str(i) for i in range(0,19)]].apply(np.sum, axis=1)

#lets take images with more than 10 cells and have classes greater than 2
feature_extractor_interest = main_df[(main_df['selected_cells'] >10) & (main_df['is_single'] == 1)] 

print('this is main ',main_df.shape)
print('this is FE ',feature_extractor_interest.shape)

class hpa_dataset(data.Dataset):
    def __init__(self, ids, selected_cells, path):
        self.ids = ids
        self.selected_cells = selected_cells
        self.path = path
        self.float_conv = albu.Compose([albu.ToFloat(max_value=255.,always_apply=True)])
    def __len__(self):
        return len(self.selected_cells)
    
    def __getitem__(self, idx):
        i = selected_cells[idx]
        hdf5_path = os.path.join(self.path,self.ids,f'{self.ids}_{i}.hdf5')
        with h5py.File(hdf5_path,"r") as h:
            vv = h['train_img'][...]
            vv = self.float_conv(image= vv)["image"]
            rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
        #print('this is rf ', rf)
        rf_np = np.full(shape = (224,224), fill_value = rf)
        vv = np.dstack([vv,rf_np])
                    
        return { 'image':vv}

vees = 'v6_2_1'
n_classes = 19
device = torch.device("cuda:1")
MODEL_PATH = f'weights/version_{vees}'
metric_use = 'loss'
model_fold_0 = HpaModel(classes = n_classes, device = device, 
                        base_model_name = 'resnest50', features = 2048, pretrained = False)

model_fold_0.load_state_dict(torch.load(f"{MODEL_PATH}/st_fold_{0}_seed_1/model_st_{metric_use}_{0}.pth",map_location = device))
model_fold_0.to(device)
model_fold_0.eval()


### lets try with our  resnet50 v6_2_1 model for now. lets see what it can do for us

total_sample = len(feature_extractor_interest)
path = 'data/train_h5_224_30000_v4'
BATCH_SIZE = 64
WORKERS = 10

ids_list = []
count_list = []
feature_vec = []
label_list = []

for idx in range(total_sample):
    info = feature_extractor_interest.iloc[idx]
    ids = info["ID"]
    is_single = info["is_single"]
    label = info["Label"]
    selected_cells = [i for i in range(1,info["selected_cells"]+1)]
    
    test_dataset = hpa_dataset(ids, selected_cells, path)
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )
    with torch.no_grad():
        for data_t in tqdm(test_dataloader):
            X = data_t['image'].to(device, dtype=torch.float)
            X = X.unsqueeze(0).permute(0,1,4,2,3)
            spe = model_fold_0.extract_features(X)
            #print('X ',X.min(),X.max())
            #print('this is spe ',spe.shape)
            feature_vec.extend(spe.squeeze(0).detach().cpu().numpy())
            
    ids_list.extend([ids]*len(selected_cells))
    count_list.extend(selected_cells)
    label_list.extend([label]*len(selected_cells))
    
    
    
cell_feature_extract = pd.DataFrame.from_dict({'ID':ids_list, 'count':count_list, 'Label':label_list})
cell_feature_extract[[str(i) for i in range(1,2048+1)]] = np.array(feature_vec)
cell_feature_extract.to_csv('data/cell_feature_extracted_single_v1.csv',index=False)


'''
cell_feature_extracted_v1.csv

main_df = pd.read_csv('data/cell_mask_study_30000.csv')
feature_extractor_interest = main_df[(main_df['selected_cells'] >10) & (main_df['is_single'] == 1)] 

vees = 'v6_2_1'
n_classes = 19
device = torch.device("cuda:0")
MODEL_PATH = f'weights/version_{vees}'
metric_use = 'loss'
model_fold_0 = HpaModel(classes = n_classes, device = device, 
                        base_model_name = 'resnest50', features = 2048, pretrained = False)

model_fold_0.load_state_dict(torch.load(f"{MODEL_PATH}/st_fold_{0}_seed_1/model_st_{metric_use}_{0}.pth",map_location = device))
model_fold_0.to(device)
model_fold_0.eval()
'''


