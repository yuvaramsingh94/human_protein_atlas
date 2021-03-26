# pip install pycocotools
# for hpa related . install it from the folder
#!pip install -q "../input/hpapytorchzoozip/pytorch_zoo-master"
#!pip install -q "../input/hpacellsegmentatormaster/HPA-Cell-Segmentation-master"
#
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import imageio
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
import imageio
import os
import h5py
from skimage.transform import resize
from model import HpaModel, HpaModel_1, HpaModel_2
import torch
import torch.utils.data as data

def build_image_names(image_id: str) -> list:
    # mt is the mitchondria
    mt = f'data/test/{image_id}_red.png'    
    # er is the endoplasmic reticulum
    er = f'data/test/{image_id}_yellow.png'    
    # nu is the nuclei
    nu = f'data/test/{image_id}_blue.png'    
    return [[mt], [er], [nu]]

import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str.decode()#('ascii')

sub = pd.read_csv('data/sample_submission.csv')
'''

NUC_MODEL = '../weights/dpn_unet_nuclei_v1.pth'
CELL_MODEL = '../weights/dpn_unet_cell_3ch_v1.pth'

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device='cuda:2',
    padding=True,
    multi_channel_model=True
)



sub_dfs = []
for dim in sub.ImageWidth.unique():
    print(dim)
    df = sub[sub['ImageWidth'] == dim].copy().reset_index(drop=True)
    sub_dfs.append(df)



import time
start = time.time()
bs = 32
img_token_sub_list = []
img_encoder_sub_list = []
img_prediction_sub_list = []
for sub in sub_dfs:
    print(f'Starting prediction for image size: {sub.ImageWidth.loc[0]}')
    for start in range(0, len(sub), bs):
    #for start in range(0, 2, bs):
        if start + bs > len(sub): end = len(sub)
        else: end = start + bs
            
        images = []
        img_id_list = []
        for row in range(start, end):
            image_id = sub['ID'].loc[row]
            img_id_list.append(image_id)
            img = build_image_names(image_id=image_id)
            #print('img shape ',img.shape)
            images.append(img)
        images = np.stack(images).squeeze()
        images = np.transpose(images).tolist()

        try: 
            nuc_segmentations = segmentator.pred_nuclei(images[2])
            cell_segmentations = segmentator.pred_cells(images)
            predstrings = []
            for i in tqdm(range(len(cell_segmentations))):
                _, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
                np.savez(f'data/test_mask/{img_id_list[i]}', cell_mask)




        except Exception as e: 
            print('hitting except ',e)
            continue
'''
'''
def img_splitter(im_tok):

    img_red    = np.expand_dims(imageio.imread(os.path.join('data/test',f'{im_tok}_red.png')), axis = -1)
    img_yellow = np.expand_dims(imageio.imread(os.path.join('data/test',f'{im_tok}_yellow.png')), axis = -1)
    img_green  = np.expand_dims(imageio.imread(os.path.join('data/test',f'{im_tok}_green.png')), axis = -1)
    img_blue   = np.expand_dims(imageio.imread(os.path.join('data/test',f'{im_tok}_blue.png')), axis = -1)
    image = np.concatenate([img_red, img_yellow, img_green, img_blue], axis=-1)
    img_mask = np.load(os.path.join('data/test_mask',f'{im_tok}.npz'))['arr_0'].astype(np.uint8)

    #print('img shape ',image.shape)
    #print('mask shape ',img_mask.shape)

    if image.shape[:-1] != img_mask.shape:
        print('they are not same ',im_tok)
        print('img shape ',image.shape)
        print('mask shape ',img_mask.shape)

    if not os.path.exists(f"data/test_h5_224_30000/{im_tok}"):
                os.mkdir(f"data/test_h5_224_30000/{im_tok}")
    
    for i in range(1, img_mask.max() + 1):
        bmask = img_mask == i

        token_list.append(im_tok)
        token_count.append(i)
        token_enc.append('' + encode_binary_mask(bmask))

        bmask = np.expand_dims(bmask, axis = -1)
        bmask = np.concatenate([bmask, bmask, bmask, bmask], axis=-1)
        masked_img = image * bmask

        true_points = np.argwhere(bmask)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
        non_zero_count = np.count_nonzero(cropped_arr[:,:,2])
        relative_freq = np.array(non_zero_count/(cropped_arr.shape[0] * cropped_arr.shape[1]))
        #print(relative_freq)
        cropped_arr = resize(cropped_arr, (224, 224))

        hdf5_path = os.path.join(f'data/test_h5_224_30000/{im_tok}',f'{im_tok}_{i}.hdf5')
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("test_img",cropped_arr.shape,np.float)
        hdf5_file.create_dataset("protein_rf",relative_freq.shape,np.float)
        hdf5_file["test_img"][...] = cropped_arr
        hdf5_file["protein_rf"][...] = relative_freq
        hdf5_file.close()

img_token_list = sub['ID'].values#[:5]

token_list = []
token_count = []
token_enc = []

if not os.path.exists(f"data/test_h5_224_30000"):
                os.mkdir(f"data/test_h5_224_30000")

for i in tqdm(img_token_list):
    img_splitter(i)

test_enc_df = pd.DataFrame.from_dict({'ID': token_list, 'count': token_count, 'encoding': token_enc})
test_enc_df.to_csv('data/test_enc_v3.csv',index=False)
'''
BATCH_SIZE = 64
WORKERS = 15
n_classes = 19
metric_use = 'loss'
vees = 'v2_3_7'
WORK_LOCATION = f'data/submissions/test_{vees}_{metric_use}/'




if not os.path.exists(WORK_LOCATION):
        os.mkdir(WORK_LOCATION)

device = torch.device("cuda:2")
MODEL_PATH = f'weights/version_{vees}'
n_classes = 19
# config_v1.ini
model_fold_0 = HpaModel_2(classes = n_classes, device = device, 
                        base_model_name = 'densenet121', features = 1024, pretrained = False, init_linear_comb = False)

model_fold_0.load_state_dict(torch.load(f"{MODEL_PATH}/fold_{0}_seed_1/model_{metric_use}_{0}.pth",map_location = device))
model_fold_0.to(device)
model_fold_0.eval()

model_fold_1 = HpaModel_2(classes = n_classes, device = device, 
                        base_model_name = 'densenet121', features = 1024, pretrained = False, init_linear_comb = False)

model_fold_1.load_state_dict(torch.load(f"{MODEL_PATH}/fold_{1}_seed_1/model_{metric_use}_{1}.pth",map_location = device))
model_fold_1.to(device)
model_fold_1.eval()

model_fold_2 = HpaModel_2(classes = n_classes, device = device, 
                        base_model_name = 'densenet121', features = 1024, pretrained = False, init_linear_comb = False)

model_fold_2.load_state_dict(torch.load(f"{MODEL_PATH}/fold_{2}_seed_1/model_{metric_use}_{2}.pth",map_location = device))
model_fold_2.to(device)
model_fold_2.eval()


def model_prediction(X):
    #print(X.shape)
    pred_0 = model_fold_0(X)['sigmoid_output']
    pred_1 = model_fold_1(X)['sigmoid_output']
    pred_2 = model_fold_2(X)['sigmoid_output']
    #torch.Size([1, 8, 5, 224, 224])
    X_up_down = torch.flip(X,[3])
    #print('X_up_down ',X_up_down.shape)
    pred_3 = model_fold_0(X_up_down)['sigmoid_output']
    pred_4 = model_fold_1(X_up_down)['sigmoid_output']
    pred_5 = model_fold_2(X_up_down)['sigmoid_output']
    #torch.Size([1, 8, 5, 224, 224])
    X_right_left = torch.flip(X,[4])
    #print('X_right_left ',X_right_left.shape)
    pred_6 = model_fold_0(X_right_left)['sigmoid_output']
    pred_7 = model_fold_1(X_right_left)['sigmoid_output']
    pred_8 = model_fold_2(X_right_left)['sigmoid_output']
    #torch.Size([1, 8, 5, 224, 224])
    X_right_left_up_down = torch.flip(X_right_left,[3])
    #print('X_right_left_up_down ',X_right_left_up_down.shape)
    pred_9 = model_fold_0(X_right_left_up_down)['sigmoid_output']
    pred_10 = model_fold_1(X_right_left_up_down)['sigmoid_output']
    pred_11 = model_fold_2(X_right_left_up_down)['sigmoid_output']

    pred = (pred_0 + pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7 + pred_8 + pred_9 + pred_10 + pred_11)/12.
    #print('pred ',pred.shape)
    #pred = torch.clamp(pred * 1.5, min=0.0, max = 1.0)
    return pred

class hpa_dataset(data.Dataset):
    def __init__(self, main_df, path):
        self.main_df = main_df
        self.path = path
    def __len__(self):
        return len(self.main_df)
    
    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]
        count = info["count"]

        hdf5_path = os.path.join(self.path,ids,f'{ids}_{count}.hdf5')
        with h5py.File(hdf5_path,"r") as h:
            vv = h['test_img'][...]
            rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
            #print('this is rf ', rf)
            rf_np = np.full(shape = (224,224), fill_value = rf)
            vv = np.dstack([vv,rf_np])
        return { 'image':vv}

test_enc_df = pd.read_csv('data/test_enc_v3.csv')#[:10]

test_dataset = hpa_dataset(main_df = test_enc_df, path = 'data/test_h5_224_30000/')
test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True, 
    )

prediction_list = []
with torch.no_grad():
    for data_t in tqdm(test_dataloader):
        X = data_t['image']
        X = X.to(device, dtype=torch.float) #(cell_count , 224, 224, 4)
        X = X.unsqueeze(0).permute(0,1,4,2,3)
        
        ## your model prediction goes here
        pred = model_prediction(X)
        #print('this is pred shape ',pred.shape)# my guess(1, cell count , 19)
        prediction_list.append(pred.detach().squeeze(0).cpu().numpy()) 
predictions = np.concatenate(prediction_list, axis=0)
#print('prediction shape ',predictions.shape)


for i in range(n_classes):
    test_enc_df[str(i)] = -1

#print(predictions.max(axis=1))

test_enc_df[[str(i) for i in range(n_classes)]] = predictions

test_enc_df.to_csv(os.path.join(WORK_LOCATION,'stage_1.csv'),index=False)
tokens_list = test_enc_df.ID.unique()

prediction_string_list = []
token_list = []
for tok in tokens_list:
    prediction_str = ''
    sub_d = test_enc_df[test_enc_df['ID'] == tok]
    for i in range(len(sub_d)):
        info = sub_d.iloc[i]
        encoding = info['encoding']
        class_pred = info[[str(j) for j in range(n_classes)]].values
        for count, k in enumerate(class_pred):
            prediction_str += f'{count} {k} ' + encoding + ' '
    #here we might have to check if the string has len > 0 . maybe we might get '' also ......
    prediction_str = prediction_str.strip()# hopefuly removes the final space
    token_list.append(tok)  
    prediction_string_list.append(prediction_str)

sub_stage_2_df = pd.DataFrame.from_dict({'ID':token_list,"PredictionString":prediction_string_list })

sub = pd.read_csv('data/sample_submission.csv')
sub = sub.drop(['PredictionString'],axis=1)
sub = sub.merge(sub_stage_2_df, on='ID')
sub.to_csv(os.path.join(WORK_LOCATION,'submission.csv'), index=False)
#'''