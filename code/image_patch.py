## create H5 for cell patch using segmented mask
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import pandas as pd
import h5py
#from skimage.transform import resize
from joblib import delayed, Parallel
import cv2

train_df = pd.read_csv('data/train.csv')

SIZE = 512
version = 'v1'

def img_splitter(im_tok):
    
    img_red    = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_red.png')), axis = -1)
    img_yellow = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_yellow.png')), axis = -1)
    img_green  = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_green.png')), axis = -1)
    img_blue   = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_blue.png')), axis = -1)
    image = np.concatenate([img_red, img_yellow, img_green, img_blue], axis=-1)

    img_resize = cv2.resize(image, (SIZE, SIZE))
        
    #if not os.path.exists(f"data/train_h5_irn_{SIZE}_{version}/{im_tok}"):
    #        os.mkdir(f"data/train_h5_irn_{SIZE}_{version}/{im_tok}")
    hdf5_path = os.path.join(f"data/train_h5_irn_{SIZE}_{version}/",f'{im_tok}.hdf5')
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("train_img",img_resize.shape,np.uint8)
    hdf5_file["train_img"][...] = img_resize
    hdf5_file.close()



img_token_list = train_df['ID'].values

#for i in  img_token_list[:3]:
    #print(i)
#    img_splitter(i)

if not os.path.exists(f"data/train_h5_irn_{SIZE}_{version}"):
        os.mkdir(f"data/train_h5_irn_{SIZE}_{version}")

Parallel(n_jobs=-1, verbose=5)(
        delayed(img_splitter)(
            im_tok,
        )
        for im_tok in img_token_list#[:5]
    )