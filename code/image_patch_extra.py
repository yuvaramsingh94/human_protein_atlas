## create H5 for cell patch using segmented mask
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
import cv2
#from skimage.transform import resize
from joblib import delayed, Parallel
import cv2

train_df = pd.read_csv('data/train_ext/extra_data_full.csv')

AREA = 40000
version = 'v6'
SIZE = 256
BUFFER = 20#pix

def img_splitter(im_tok):

    img_red    = np.expand_dims(cv2.imread(os.path.join('data/train_ext/HPA-Challenge-2021-trainset-extra',f'{im_tok}_red.png'), cv2.IMREAD_GRAYSCALE), axis = -1)
    img_yellow = np.expand_dims(cv2.imread(os.path.join('data/train_ext/HPA-Challenge-2021-trainset-extra',f'{im_tok}_yellow.png'), cv2.IMREAD_GRAYSCALE), axis = -1)
    img_green  = np.expand_dims(cv2.imread(os.path.join('data/train_ext/HPA-Challenge-2021-trainset-extra',f'{im_tok}_green.png'), cv2.IMREAD_GRAYSCALE), axis = -1)
    img_blue   = np.expand_dims(cv2.imread(os.path.join('data/train_ext/HPA-Challenge-2021-trainset-extra',f'{im_tok}_blue.png'), cv2.IMREAD_GRAYSCALE), axis = -1)
    image = np.concatenate([img_red, img_yellow, img_green, img_blue], axis=-1)
    #print(image.max())
    img_mask = np.load(os.path.join('data/train_ext/mask',f'{im_tok}_cell.npz'))['arr_0'].astype(np.uint8)
    img_nu_mask = np.load(os.path.join('data/train_ext/mask',f'{im_tok}_nu.npz'))['arr_0'].astype(np.uint8)
    Shape = img_mask.shape
    crop_img_list = []
    count = 0
    for i in range(1, img_mask.max() + 1):
        bmask = img_mask == i
        masked_nuclieus = img_nu_mask * bmask

        true_nu_points = np.argwhere(masked_nuclieus)
        try:
            top_nu_left = true_nu_points.min(axis=0)
            bottom_nu_right = true_nu_points.max(axis=0)

            #height , width
            y1, x1 = top_nu_left
            y2, x2 = bottom_nu_right

            bmask = np.expand_dims(bmask, axis = -1)
            bmask = np.concatenate([bmask, bmask, bmask, bmask], axis=-1)
            masked_img = image * bmask

            true_points = np.argwhere(bmask)
            top_left = true_points.min(axis=0)
            bottom_right = true_points.max(axis=0)
            cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
            #print(cropped_arr.shape)
            #print('Area: ',cropped_arr.shape[0] * cropped_arr.shape[1])
            if cropped_arr.shape[0] * cropped_arr.shape[1] > AREA:

                if not (y1 < BUFFER or x1 < BUFFER or y2 > Shape[0] - BUFFER or x2 > Shape[1] - BUFFER):
                    count += 1

                    #lets calcualte teh green channel stats
                    non_zero_count = np.count_nonzero(cropped_arr[:,:,2])
                    relative_freq = np.array(non_zero_count/(cropped_arr.shape[0] * cropped_arr.shape[1]))
                    #print('relative freq ', relative_freq)
                    actual_h = cropped_arr.shape[0]
                    actual_w = cropped_arr.shape[1]
                    cropped_arr = cv2.resize(cropped_arr, (SIZE, SIZE))
                    #print('after resize ',cropped_arr.min(),cropped_arr.max())
                    if not os.path.exists(f"data/train_h5_{SIZE}_{AREA}_{version}/{im_tok}"):
                        #print('creating')
                        os.mkdir(f"data/train_h5_{SIZE}_{AREA}_{version}/{im_tok}")

                    hdf5_path = os.path.join(f"data/train_h5_{SIZE}_{AREA}_{version}/{im_tok}",f'{im_tok}_{count}.hdf5')
                    hdf5_file = h5py.File(hdf5_path, mode='w')
                    hdf5_file.create_dataset("train_img",cropped_arr.shape,np.uint8)
                    hdf5_file.create_dataset("protein_rf",relative_freq.shape,np.float)
                    hdf5_file["train_img"][...] = cropped_arr
                    hdf5_file["protein_rf"][...] = relative_freq
                    hdf5_file.close()
                #else:
                #    print('the nuclius is tourching the edge ',im_tok)
            #else:
            #    print('area ')
        except :
            pass
            #print('no nuce ',im_tok)
            #print(true_nu_points)

img_token_list = train_df['ID'].values

#for i in  img_token_list[:3]:
    #print(i)
#    img_splitter(i)
if not os.path.exists(f"data/train_h5_{SIZE}_{AREA}_{version}"):
    os.mkdir(f"data/train_h5_{SIZE}_{AREA}_{version}")
Parallel(n_jobs=10, verbose=5)(# job -1
        delayed(img_splitter)(
            im_tok,
        )
        for im_tok in img_token_list#[:20]
    )

'''
v6 = we are going to remove nuclieus that are tourching the edge of the image,

'''