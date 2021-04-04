import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import importlib
from misc import pyutils, torchutils


class hpa_dataset(data.Dataset):
    class hpa_dataset_v1(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, label_smoothing = False, l_alp = 0.3, is_validation = False):
        self.main_df = main_df
        self.label_col = [str(i) for i in range(19)]
        self.path = path
        self.is_validation = is_validation
    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]
        hdf5_path = os.path.join(self.path,ids,f'{ids}.hdf5')
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
            img = pack['img'].to(device,non_blocking=True)

            label = pack['label'].to(device,non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    device = torch.device("cuda:0")

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

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

            img = pack['img'].to(device,non_blocking=True)
            label = pack['label'].to(device,non_blocking=True)

            x = model(img)
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