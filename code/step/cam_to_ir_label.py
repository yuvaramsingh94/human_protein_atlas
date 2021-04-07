
import os
import numpy as np
import imageio
import pandas as pd
from torch import multiprocessing
from torch.utils.data import DataLoader
import torch.utils.data as data
#import voc12.dataloader
from misc import torchutils, imutils
import h5py
class hpa_dataset_cam(data.Dataset):

    def __init__(self, main_df,  path=None, scales=(1.0,)):
        self.main_df = main_df
        self.label_col = [str(i) for i in range(19)]
        self.path = path
        self.scales = scales
    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]
        hdf5_path = os.path.join(self.path,f'{ids}.hdf5')
        target_vec = info[self.label_col].values.astype(np.int)
        with h5py.File(hdf5_path,"r") as h:
            img = h['train_img'][...]
        #name = self.img_name_list[idx]
        #name_str = decode_int_filename(name)
        if img.shape[-1] == 4:
            #remove the protein layer , we will add it back agter augmentation
            img = np.dstack([img[:,:,0], img[:,:,1], img[:,:,-1]])
        #img = imageio.imread(get_img_path(name_str, self.voc12_root))

        ## we will remove 
        out = {"name": ids, "img": img, "label": target_vec}
        return out


def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = pack['name'][0]
        img = pack['img'][0].numpy()
        
        cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                        conf.astype(np.uint8))


        if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):
    train_base_df = pd.read_csv('data/train_fold_v6.csv')
    fold = 0
    train_df = train_base_df[train_base_df['fold'] != fold]
    valid_df = train_base_df[train_base_df['fold'] == fold]
    PATH = 'data/train_h5_irn_512_v1'
    dataset = hpa_dataset_cam(main_df = train_df, path=PATH, scales=None)
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')
