## create H5 for cell patch using segmented mask
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import pandas as pd
import h5py
from skimage.transform import resize
from joblib import delayed, Parallel

train_df = pd.read_csv('data/train.csv')

def img_splitter(im_tok):
    #img_filename = img_token+'.png'
    
    
    img_red    = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_red.png')), axis = -1)
    img_yellow = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_yellow.png')), axis = -1)
    img_green  = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_green.png')), axis = -1)
    img_blue   = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_blue.png')), axis = -1)
    image = np.concatenate([img_red, img_yellow, img_green, img_blue], axis=-1)
    img_mask = np.load(os.path.join('data/hpa_cell_mask',f'{im_tok}.npz'))['arr_0'].astype(np.uint8)
    #print(image.shape,img_mask.shape)
    #area_list = []
    crop_img_list = []
    
    for i in range(1, img_mask.max() + 1):
        bmask = img_mask == i
        bmask = np.expand_dims(bmask, axis = -1)
        bmask = np.concatenate([bmask, bmask, bmask, bmask], axis=-1)
        masked_img = image * bmask

        true_points = np.argwhere(bmask)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
        #print(cropped_arr.shape)
        #print('Area: ',cropped_arr.shape[0] * cropped_arr.shape[1])
        if cropped_arr.shape[0] * cropped_arr.shape[1] > 100000:
            cropped_arr = resize(cropped_arr, (224, 224))
            #area_list.append(cropped_arr.shape[0] * cropped_arr.shape[1]
            crop_img_list.append(cropped_arr)
            
            #pass
    crop_img_arr = np.array(crop_img_list)
    hdf5_path = os.path.join('data/train_h5_224',f'{im_tok}.hdf5')
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("train_img",crop_img_arr.shape,np.float)
    hdf5_file["train_img"][...] = crop_img_arr
    hdf5_file.close()

img_token_list = train_df['ID'].values

#for i in  img_token_list[:3]:
    #print(i)
#    img_splitter(i)

Parallel(n_jobs=-1, verbose=5)(
        delayed(img_splitter)(
            im_tok,
        )
        for im_tok in img_token_list#[:5]
    )