import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import os
import pandas as pd
#import h5py
from skimage.transform import resize
from tqdm import tqdm

train_df = pd.read_csv('data/train.csv')
print(train_df.shape)

def img_splitter(im_tok):
    #img_filename = img_token+'.png'
    
    file_name.append(im_tok)
    #img_red    = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_red.png')), axis = -1)
    #img_yellow = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_yellow.png')), axis = -1)
    #img_green  = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_green.png')), axis = -1)
    #img_blue   = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_blue.png')), axis = -1)
    #image = np.concatenate([img_red, img_yellow, img_green, img_blue], axis=-1)
    img_mask = np.load(os.path.join('data/hpa_cell_mask',f'{im_tok}.npz'))['arr_0'].astype(np.uint8)

    #area_list = []
    crop_img_list = []
    
    selected_cell_count = 0
    total_cell_mask.append(img_mask.max())
    masked_img = np.zeros(img_mask.shape)
    #print(masked_img.shape)
    for i in range(1, img_mask.max() + 1):
        bmask = img_mask == i
        bmask = np.expand_dims(bmask, axis = -1)
        #bmask = np.concatenate([bmask, bmask, bmask, bmask], axis=-1)
        #masked_img = image * bmask

        true_points = np.argwhere(bmask)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
        #print(cropped_arr.shape)
        #print('Area: ',cropped_arr.shape[0] * cropped_arr.shape[1])
        if cropped_arr.shape[0] * cropped_arr.shape[1] > 100000:
            selected_cell_count += 1
            #print(cropped_arr.min(), cropped_arr.max())
            cropped_arr = resize(cropped_arr, (224, 224))
            #print(cropped_arr.min(), cropped_arr.max())
            #area_list.append(cropped_arr.shape[0] * cropped_arr.shape[1]
            #crop_img_list.append(cropped_arr)
            
            #pass
    crop_img_arr = np.array(crop_img_list)

    selected_cells.append(selected_cell_count)
    return None #crop_img_list
    


file_name = []
total_cell_mask = []
selected_cells = []
img_token_list = train_df['ID'].values

for i in  tqdm(img_token_list):
    crop_img_list = img_splitter(i)
    
v = {'token':file_name,'total_cell_mask':total_cell_mask,'selected_cells':selected_cells}
v_df = pd.DataFrame.from_dict(v)
v_df.to_csv('data/cell_mask_study_v3.csv')

#v1 = 30000
#v2 = 70000
#v3 = 100000