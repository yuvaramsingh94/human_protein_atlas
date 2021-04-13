import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import imageio

import imageio
import os
import h5py
from skimage.transform import resize
import torch
import torch.utils.data as data
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from pycocotools import _mask as coco_mask
#work/ddsm/hpa/human_protein_atlas/data/train_ext/HPA-Challenge-2021-trainset-extra
def build_image_names(image_id: str) -> list:
    # mt is the mitchondria
    mt = f'data/train_ext/HPA-Challenge-2021-trainset-extra/{image_id}_red.png'    
    # er is the endoplasmic reticulum
    er = f'data/train_ext/HPA-Challenge-2021-trainset-extra/{image_id}_yellow.png'    
    # nu is the nuclei
    nu = f'data/train_ext/HPA-Challenge-2021-trainset-extra/{image_id}_blue.png'    
    return [[mt], [er], [nu]]

import base64
import numpy as np

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

sub = pd.read_csv('data/train_ext/extra_cutdown_data_part2.csv')

import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from pycocotools import _mask as coco_mask

NUC_MODEL = '../weights/dpn_unet_nuclei_v1.pth'
CELL_MODEL = '../weights/dpn_unet_cell_3ch_v1.pth'

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device='cuda:0',
    padding=True,
    multi_channel_model=True
)


import time
start = time.time()
bs = 32
img_token_sub_list = []


sub_dfs = [sub]

for sub in sub_dfs:
    #print(f'Starting prediction for image size: {sub.ImageWidth.loc[0]}')
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
                np.savez_compressed(f'data/train_ext/mask/{img_id_list[i]}', cell_mask)




        except Exception as e: 
            print('hitting except ',e)
            continue