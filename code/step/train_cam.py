import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import importlib
from misc import pyutils, torchutils
import torch.utils.data as data
import pandas as pd
import os
import numpy as np
import h5py


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

def validate(model, data_loader,device):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['image'].permute(0,3,1,2).to(device).float()

            label = pack['label'].to(device)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    device = torch.device("cuda:0")

    train_base_df = pd.read_csv('data/train_fold_v6.csv')
    fold = 0
    train_df = train_base_df[train_base_df['fold'] != fold]
    valid_df = train_base_df[train_base_df['fold'] == fold]
    PATH = 'data/train_h5_irn_512_v1'
    train_dataset = hpa_dataset(main_df = train_df, path=PATH, is_validation = False)
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = hpa_dataset(main_df = valid_df, path=PATH, is_validation = True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = model.to(device)
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['image'].permute(0,3,1,2).to(device).float()
            label = pack['label'].to(device)

            x = model(img)
            #print('vi ',x.shape)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader,device)
            timer.reset_stage()

    torch.save(model.state_dict(), args.cam_weights_name + '.pth')