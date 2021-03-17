## create H5 for cell patch using segmented mask
import numpy as np
import imageio
import os
import pandas as pd
import cv2
from joblib import delayed, Parallel

train_df = pd.read_csv('data/filtered_single.csv')
PATH = 'data/visualization/'

def img_viz(im_tok, label):
    #img_filename = img_token+'.png'
    
    
    img_red    = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_red.png')), axis = -1)
    img_yellow = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_yellow.png')), axis = -1)
    img_green  = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_green.png')), axis = -1)
    img_blue   = np.expand_dims(imageio.imread(os.path.join('data/train',f'{im_tok}_blue.png')), axis = -1)
    
    image_viz = cv2.resize(np.hstack([np.dstack([img_red, img_yellow, img_blue]), 
                            np.dstack([img_red, img_green, img_blue])]),(1024,1024))
    imageio.imwrite(os.path.join(PATH,f'{im_tok}_{str(label)}.jpg'), image_viz)


img_token_list = train_df[['ID','Label']].values


Parallel(n_jobs=20, verbose=5)(
        delayed(img_viz)(
            im_tok,label
        )
        for im_tok, label in img_token_list#[:5]
    )