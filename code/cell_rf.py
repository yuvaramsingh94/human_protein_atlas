### the aim here is to get a dataset which will have relative freq of protein presence per cell
### this will be useful for finding about where to add 18 to a clas

import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import os
import pandas as pd
#import h5py
from skimage.transform import resize
from tqdm import tqdm
import h5py

train_df = pd.read_csv('data/train.csv')
print(train_df.shape)

def img_splitter(im_tok):
    #img_filename = img_token+'.png'
    
    
    #img_red    = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_red.png')), axis = -1)
    #img_yellow = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_yellow.png')), axis = -1)
    #img_green  = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_green.png')), axis = -1)
    #img_blue   = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_blue.png')), axis = -1)
    #image = np.concatenate([img_red, img_yellow, img_green, img_blue], axis=-1)

    try:
        cell_count = len(os.listdir(os.path.join('data/train_h5_224_30000_v3',f'{im_tok}')))
        #print(cell_count)
        for i in range(1, cell_count + 1):
            tok_name.append(im_tok)
            file_name.append(im_tok+f'_{i}')
            hdf5_path = f'data/train_h5_224_30000_v3/{im_tok}/{im_tok}_{i}.hdf5'
            with h5py.File(hdf5_path,"r") as h:
                rf = h['protein_rf'][...]
                #print(rf)
                rf_list.append(rf)
    except:
        print('issue ',os.path.join('data/train_h5_224_30000_v3',f'{im_tok}'))

    return None #crop_img_list
    


file_name = []
tok_name = []
rf_list = []

img_token_list = train_df['ID'].values#[:5]

for i in  tqdm(img_token_list):
    crop_img_list = img_splitter(i)
    
v = {'ID':tok_name,'file_name':file_name,'rf_list':rf_list}
v_df = pd.DataFrame.from_dict(v)
v_df.to_csv('data/cell_rf_v1.csv')
