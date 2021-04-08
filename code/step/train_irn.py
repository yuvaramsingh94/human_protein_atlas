
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from misc import pyutils, torchutils, indexing
import importlib
import pandas as pd
import h5py
import imageio
import os
from misc import imutils
import numpy as np

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


class VOC12SegmentationDataset(data.Dataset):

    def __init__(self, main_df, path, label_dir, crop_size,
                 rescale=None, img_normal=None, hor_flip=False,
                 crop_method = 'random'):

        self.main_df = main_df

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.path = path

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]
        hdf5_path = os.path.join(self.path,f'{ids}.hdf5')
        with h5py.File(hdf5_path,"r") as h:
            img = h['train_img'][...]
        label = imageio.imread(os.path.join(self.label_dir, ids + '.png'))


        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        #if self.img_normal:
        #    img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)

        return {'name': ids, 'img': img, 'label': label}

class VOC12AffinityDataset(VOC12SegmentationDataset):
    def __init__(self, main_df, path, label_dir, crop_size, 
                 indices_from, indices_to,
                 rescale=None, img_normal=None, hor_flip=False, crop_method=None):
        super().__init__(main_df, path, label_dir, crop_size,  rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out




def run(args):

    path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    model = getattr(importlib.import_module(args.irn_network), 'AffinityDisplacementLoss')(
        path_index)
    device = torch.device("cuda:0")

    train_base_df = pd.read_csv('data/train_fold_v6.csv')
    fold = 0
    train_df = train_base_df#[train_base_df['fold'] != fold]
    valid_df = train_base_df[train_base_df['fold'] == fold][:100]
    PATH = 'data/train_h5_irn_512_v1'


    train_dataset = VOC12AffinityDataset(train_df,
                                        path = PATH,
                                        label_dir=args.ir_label_out_dir,
                                        indices_from=path_index.src_indices,
                                        indices_to=path_index.dst_indices,
                                        hor_flip=True,
                                        crop_size=args.irn_crop_size,
                                        crop_method="random",
                                        rescale=(0.5, 1.5)
                                        )
    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[1], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[2], 'lr': 10*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay}
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = model.to(device)
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img'].to(device,non_blocking=True).float()/255.
            bg_pos_label = pack['aff_bg_pos_label'].to(device,non_blocking=True).float()
            fg_pos_label = pack['aff_fg_pos_label'].to(device,non_blocking=True).float()
            neg_label = pack['aff_neg_label'].to(device,non_blocking=True).float()

            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)

            bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

            dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
            dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})

            total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (
                      avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'), avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter + 1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()
    '''
    infer_dataset = voc12.dataloader.VOC12ImageDataset(args.infer_list,
                                                       voc12_root=args.voc12_root,
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.irn_batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            img = pack['img'].to(device,non_blocking=True)

            aff, dp = model(img, False)

            dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

        model.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
    print('done.')
    '''
    torch.save(model.state_dict(), args.irn_weights_name)
    #torch.to.empty_cache()
