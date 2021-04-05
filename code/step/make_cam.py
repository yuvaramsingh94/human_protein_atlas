import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True

class hpa_dataset(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,is_validation = False):
        self.main_df = main_df
        self.label_col = [str(i) for i in range(19)]
        self.path = path
        self.is_validation = is_validation
    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]
        hdf5_path = os.path.join(self.path,f'{ids}.hdf5')
        target_vec = info[self.label_col].values.astype(np.int)

        if not self.is_validation:
            with h5py.File(hdf5_path,"r") as h:
                vv = h['train_img'][...]
                vv = vv/255.
        else:
            with h5py.File(hdf5_path,"r") as h:
                vv = h['train_img'][...]
                vv = vv/255.
        return {'image' : vv, 'label' : target_vec}



def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    print('loading tyhis cam : ',args.cam_weights_name + '.pth')
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = 1#torch.cuda.device_count()

    train_base_df = pd.read_csv('data/train_fold_v6.csv')
    fold = 0
    train_df = train_base_df[train_base_df['fold'] != fold]
    valid_df = train_base_df[train_base_df['fold'] == fold]
    PATH = 'data/train_h5_irn_512_v1'
    dataset = hpa_dataset(main_df = train_df, path=PATH, is_validation = False)
    
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()