import torch
from utils import set_seed, focal_loss, CosineAnnealingWarmupRestarts
from model import HpaModel_2
import pandas as pd
import os
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import math
import albumentations as albu
import argparse
import math
import h5py
import matplotlib.pyplot as plt
from torch.backends import cudnn
#cudnn.benchmark = True

#https://www.kaggle.com/hirune924/o2unet-loss-aggregate

max_LR = 0.001
min_LR = 0.00001
CELL_COUNT = 12
BATCH_SIZE = 4
WORKERS = 10
EPOCH = 75#60
O2U_save = 'O2U_v2'
if not os.path.exists(f"weights/{O2U_save}"):
    os.mkdir(f"weights/{O2U_save}")

parser = argparse.ArgumentParser(description='set seed.')
parser.add_argument('seed', metavar='N', type=int, nargs='+',
                    help='seed to use')
parser.add_argument('cuda', metavar='N', type=int, nargs='+',
                    help='cuda device to use')


def create_df():
    train_df = pd.read_csv('data/cell_mask_study_30000.csv')
    filtered_df_10_greater = train_df[(train_df['selected_cells'] >12) ] 
    actual_cells = []
    PATH = 'data/train_h5_256_40000_v5/'
    for i in filtered_df_10_greater['ID'].values:
        actual_cells.append(len(os.listdir(os.path.join(PATH, i))))
    filtered_df_10_greater['actual_cells'] = actual_cells

    cells_count = CELL_COUNT
    total_samples = filtered_df_10_greater.shape[0]

    ID_list = []
    count_list = []
    count = []
    Label_list = []
    for i in range(total_samples):
    #for i in range(2):
        info = filtered_df_10_greater.iloc[i]
        ids = info["ID"]
        actual_c = info["actual_cells"]
        label_c = info["Label"]
        cells_c = [v for v in range(1,actual_c+1)]
        for v in range(1, actual_c+1, cells_count):
            #print(v)
            start = v
            end = min(start+cells_count, actual_c+1)
            #print(f'start {start} end {end}')
            ID_list.append(ids)
            count_list.append([j for j in range(start,end)])
            count.append(len([j for j in range(start,end)]))
            Label_list.append(label_c)
    ou_df = pd.DataFrame.from_dict({'ID':ID_list,"count_list":count_list, 'count': count, 'Label' : Label_list})
    labels = [str(i) for i in range(19)]
    for x in labels: ou_df[x] = ou_df['Label'].apply(lambda r: int(x in r.split('|')))
    print(ou_df.head())
    return ou_df



class hpa_dataset_v1(data.Dataset):
    def __init__(self, main_df,  path=None,  cells_used = 8,  size = 224):
        self.main_df = main_df
        self.main_df['count_list'].tolist()
        self.float_conv = albu.Compose([albu.ToFloat(max_value=255.,always_apply=True)])
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path
        self.size = size

        #self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]

        target_vec = info[self.label_col].values.astype(np.int)

        # lets begin
        count_list = info['count_list']
        
        if len(count_list) == self.cells_used:
            cell_list = []
            for i in count_list:
                hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                with h5py.File(hdf5_path,"r") as h:
                    vv = h['train_img'][...]
                    vv = self.float_conv(image= vv)["image"]
                    rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                    #print('this is rf ', rf)
                    rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                    vv = np.dstack([vv,rf_np])
                    #print('this is vv shape ',vv.shape)
                    cell_list.append(vv)
            train_img = np.array(cell_list)

        elif len(count_list) < self.cells_used:# add zero images
            ##print('in the less class')
            cell_list = []
            for i in count_list:
                hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                with h5py.File(hdf5_path,"r") as h:
                    vv = h['train_img'][...]
                    vv = self.float_conv(image= vv)["image"]
                    rf = h['protein_rf'][...] - 0.5 ##this 0.5 is to zero center the values
                    #print('this is rf ', rf)
                    rf_np = np.full(shape = (self.size,self.size), fill_value = rf)
                    vv = np.dstack([vv,rf_np])
                    #print('this is vv shape ',vv.shape)
                    cell_list.append(vv)
            train_img = np.array(cell_list)
            shape = (self.cells_used - len(count_list), self.size, self.size, 5)
            zero_arr = np.zeros(shape, dtype=float)
            ##print('zero_arr ',zero_arr.shape)
            ##print('train_img ',train_img.shape)
            train_img = np.concatenate([train_img, zero_arr], axis=0)
            target_vec[-1] = 1# as we are adding black img . negative = 1 also
            #print('black ',target_vec)
        mask = len(count_list)

        return {'image' : train_img, 'label' : target_vec, 'mask':mask, 'ids':idx}



def train(model,train_dataloader,optimizer,criterion,loss_df,epoch):

    model.train()
    #print('model.training = ',model.training)
    train_loss_loop_list = []
    for data_t in tqdm(train_dataloader):
        X, Y, ids, mask = data_t['image'],data_t['label'],data_t['ids'],data_t['mask'][0]
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device, dtype=torch.float)
        X = X.permute(0,1,4,2,3)
        #print(X[:,:,:4,:,:].min(), X.max())
        
        optimizer.zero_grad(set_to_none=True)#better mode
        with torch.cuda.amp.autocast():
            prediction = model(X, mask)
            train_loss = criterion(prediction['final_output'], Y) 
            train_loss = train_loss.mean(1)# this will give [bs,]
            #print('IDs ',ids)
            for count in range(len(train_loss)):
                #print(train_loss[count].item())
                loss_df.at[ids[count].item(),f'epoch_{epoch}'] = train_loss[count].item()
            #print('loss shape ',train_loss.shape)
            train_loss = train_loss.mean()  
        
        scaler.scale(train_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()

        train_loss_loop_list.append(train_loss.item())

    train_total_loss = np.array(train_loss_loop_list)
    train_total_loss = train_total_loss.sum() / len(train_total_loss)

    print(f" \n train loss : {train_total_loss}")
    return loss_df, train_total_loss


'''
    model.train()
    #print('model.training = ',model.training)
    train_loss_loop_list = []
    LWLRAP_loop_list = []
    for data_t in tqdm(train_dataloader):
        X, Y, ids = data_t['waveform'],data_t['targets'],data_t['ids']

        X = X.to(device, dtype=torch.float)
        Y = Y.to(device, dtype=torch.float)
        optimizer.zero_grad()

        prediction = model(X)
        #print(prediction["clipwise_output"].max())
        train_loss = criterion(prediction, Y)
        train_loss = train_loss.mean(1)
        for count in range(len(train_loss)):
            #print(train_loss[count])
            loss_df.at[ids[count],f'epoch_{epoch}'] = train_loss[count].item()

        train_loss = train_loss.mean()
        train_loss.backward()
        optimizer.step()
        train_loss_loop_list.append(train_loss.item())

    train_total_loss = np.array(train_loss_loop_list)
    train_total_loss = train_total_loss.sum() / len(train_total_loss)
    print(f'At Epoch {epoch} train loss {train_total_loss}')
    return loss_df, train_total_loss
'''


def run():


    better_df = create_df()
    print('shape of better before ',better_df.shape)
    better_df = better_df[better_df['count']>5]
    print('shape of better ',better_df.shape)
    train_dataset = hpa_dataset_v1(main_df = better_df, path = 'data/train_h5_256_40000_v5',  
                                    cells_used = CELL_COUNT,
                                    size = 256)

    train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False,
            pin_memory=True,
            
        )


    model = HpaModel_2(19, device = device, 
                            base_model_name = 'resnet34', 
                            features = 512, pretrained = True,)
    model = model.to(device)

    # Optimizer
    optimizer = optim._multi_tensor.AdamW(model.parameters(), lr= max_LR)

    # Scheduler
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=15,
                                          cycle_mult=1.0,
                                          max_lr=max_LR,
                                          min_lr=min_LR,
                                          warmup_steps=0,
                                          gamma=1.0)

    # Loss
    criterion = focal_loss(alpha=0.25, gamma=2, reduction = 'none').cuda(device=device)#PANNsLoss().to(device)

    data_loss_dict = {'ID':better_df['ID'].values, 
                      'count_list':better_df['count_list'].values,
                      'count':better_df['count'].values,'Label':better_df['Label'].values}
    loss_df = pd.DataFrame.from_dict(data_loss_dict)

    
    lr_list = []
    train_loss_list = []
    for epoch in range (EPOCH):
        loss_df[f'epoch_{epoch}']  = -1.
        
        loss_df, train_total_loss = train(model,train_dataloader,optimizer,criterion,loss_df,epoch)
        scheduler.step()
        print('EPOCH ',epoch)

        for param_group in optimizer.param_groups:
            lr_list.append(param_group["lr"])
        train_loss_list.append(train_total_loss)

    plt.plot(lr_list)
    plt.savefig(f"weights/{O2U_save}/lr.jpg")
    plt.clf()
    plt.plot(train_loss_list)
    plt.savefig(f"weights/{O2U_save}/train_loss.jpg")
    plt.clf()
    plt.close()

    loss_df.to_csv(f"weights/{O2U_save}/o2u.csv")
    print('Better df shape ',better_df.shape)
    print('loss df shape ',loss_df.shape)

    
    
if __name__ == "__main__":

    args = parser.parse_args()

    SEED = args.seed[0]

    scaler = torch.cuda.amp.GradScaler()

    if not os.path.exists(f"weights/{O2U_save}"):
        os.mkdir(f"weights/{O2U_save}")
    print('setting seed to ',SEED)
    set_seed(SEED)
    CUDA_DEVICE = args.cuda[0]
    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:0")
    run()