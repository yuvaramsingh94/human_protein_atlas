
import torch
from utils import set_seed, score_metrics
from model import HpaModel
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
import gc
import shutil
import argparse
import configparser



parser = argparse.ArgumentParser(description='set seed.')
parser.add_argument('seed', metavar='N', type=int, nargs='+',
                    help='seed to use')
parser.add_argument('cuda', metavar='N', type=int, nargs='+',
                    help='cuda device to use')
parser.add_argument('config_file', metavar='N', type=str, nargs='+',
                    help='configuration file path')


def train(model,train_dataloader,optimizer,criterion, MIXUP = False):
    model.train()
    #print('model.training = ',model.training)
    train_loss_loop_list = []
    LWLRAP_loop_list = []
    for data_t in tqdm(train_dataloader):
        X, Y = data_t['waveform'],data_t['targets']
        #print(X.shape)
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device, dtype=torch.float)
        mixup_lambda = None
        if MIXUP and X.shape[0]%2 == 0:
            mixup_lambda = mixup_values.get_lambda(int(X.shape[0]))
            mixup_lambda = mixup_lambda.to(device, dtype=torch.float)
            #Y = torch.logical_or(Y[0::2], Y[1::2])*1.0
            Y = (Y[0::2].transpose(0, -1) * mixup_lambda[0::2] + Y[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
            #print('Mix')
        #print("X shape ",X.shape)
        #print('Y ',Y.shape)
        optimizer.zero_grad()

        #with amp.autocast():
        prediction = model(X, mixup_lambda = mixup_lambda)
        #print(prediction["clipwise_output"].max())
        if prediction["clipwise_output"].min().item() < 0. or prediction["clipwise_output"].max().item() >1.:
            print(f'prediction broke {prediction["clipwise_output"].min().item()} {prediction["clipwise_output"].max().item()}',)
        train_loss = criterion(prediction, Y) + lsep_loss_stable(prediction["clipwise_output"], Y)
        #train_loss = lsep_loss_stable(prediction["clipwise_output"], Y)
        
        train_loss.backward()
        optimizer.step()
        train_loss_loop_list.append(train_loss.item())
        #lwlrap_metric = LWLRAP(torch.sigmoid(prediction["clipwise_output"]), Y, device= device)
        lwlrap_metric = LWLRAP(prediction["clipwise_output"], Y, device= device)
                
        LWLRAP_loop_list.append(lwlrap_metric if not math.isnan(lwlrap_metric) else 0.0 )


    train_total_loss = np.array(train_loss_loop_list)
    train_total_loss = train_total_loss.sum() / len(train_total_loss)
    
    LWLRAP_loop_list = np.array(LWLRAP_loop_list)
    LWLRAP_loop_list = LWLRAP_loop_list.sum() / len(LWLRAP_loop_list)



    print(f" \n train loss : {train_total_loss} LWLRAP {LWLRAP_loop_list}")
    return train_total_loss


def validation(model,valid_dataloader,criterion):
    model.eval()
    #print('model.training = ',model.training)
    valid_loss_loop_list = []
    LWLRAP_loop_list = []
    AUROC_loop_list = []
    F1_loop_list = []
    with torch.no_grad():
        for data_t in tqdm(valid_dataloader):

            X, Y = data_t['waveform'],data_t['targets']



            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, dtype=torch.float)
            #print("X shape ",X.shape)
            #print("X shape ",Y)

            prediction = model(X)

            valid_loss = criterion(prediction, Y) + lsep_loss_stable(prediction["clipwise_output"], Y)
            valid_loss_loop_list.append(valid_loss.detach().cpu().item())

            lwlrap_metric = LWLRAP(prediction["clipwise_output"], Y, device= device)
            scores = score_metrics(prediction["clipwise_output"], Y)
            #print(scores)
            LWLRAP_loop_list.append(lwlrap_metric if not math.isnan(lwlrap_metric) else 0.0 )
            AUROC_loop_list.append(scores['AUROC'] if not math.isnan(lwlrap_metric) else 0.0 )
            F1_loop_list.append(scores['F1_score'] if not math.isnan(lwlrap_metric) else 0.0 )


    valid_total_loss = np.array(valid_loss_loop_list)
    valid_total_loss = valid_total_loss.sum() / len(valid_total_loss)
    
    LWLRAP_loop_list = np.array(LWLRAP_loop_list)
    LWLRAP_loop_list = LWLRAP_loop_list.sum() / len(LWLRAP_loop_list)

    AUROC_loop_list = np.array(AUROC_loop_list)
    AUROC_loop_list = AUROC_loop_list.sum() / len(AUROC_loop_list)

    F1_loop_list = np.array(F1_loop_list)
    F1_loop_list = F1_loop_list.sum() / len(F1_loop_list)


    #valid_total_loss = 0.0
    print(f" \n valid loss : {valid_total_loss} LWLRAP {LWLRAP_loop_list}")
    return valid_total_loss, LWLRAP_loop_list, {'AUROC':AUROC_loop_list,'F1_score':F1_loop_list}



def master_validation(model,bet_df):
    model.eval()
    print('model.training = ',model.training)
    LWLRAP_loop_list = []
    
    master_dataset = master_val_dataset_v2(main_df = bet_df, 
                                              path = DATA_PATH)
    
    
    master_dataloader = data.DataLoader(
        master_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=False,
    )
    
    with torch.no_grad():
        
        LWLRAP_loop_list = []
        for data_t in tqdm(master_dataloader):

            X, Y = data_t['waveform'],data_t['targets']
            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, dtype=torch.float)
            prediction = model(X)
            #lwlrap_metric = LWLRAP(torch.sigmoid(prediction["clipwise_output"]), Y, device= device)
            lwlrap_metric = LWLRAP(prediction["clipwise_output"], Y, device= device)
            LWLRAP_loop_list.append(lwlrap_metric if not math.isnan(lwlrap_metric) else 0.0 ) 
    LWLRAP_loop_list = np.array(LWLRAP_loop_list)
    LWLRAP_loop_list = LWLRAP_loop_list.sum() / len(LWLRAP_loop_list)
    print(f"LWLRAP master {LWLRAP_loop_list}")
    return LWLRAP_loop_list




def run(fold):

    train_dataset = rfcx_dataset_v3(main_df = train_df, path = DATA_PATH, effective_sec= effective_sec, augmentation = transform, aug_per= 0.4, is_mixing = True, mix_per = 0.4)
    valid_dataset_segment = rfcx_dataset_v2(main_df = valid_df, path = DATA_PATH, effective_sec= effective_sec, augmentation = None, is_validation_full = False)

    if not os.path.exists(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}"):
        os.mkdir(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}")

    if  os.path.exists(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/log_dir"):
        shutil.rmtree(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/log_dir")

    
    writer = SummaryWriter(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/log_dir")

    writer.add_text('description',f'This is done using config {args.config_file[0]} ')


    train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False,
            pin_memory=True,
            
        )

    valid_dataloader = data.DataLoader(
        valid_dataset_segment,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )


    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr= LR)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    # Loss
    criterion = loss()

    best_val_AUROC = 0.0
    best_val_F1_score = 0.0
    for epoch in range (EPOCH):
        train_loss = train(model,train_dataloader,optimizer,criterion, MIXUP = False)
        val_loss, val_LWLRAP,val_scores = validation(model,valid_dataloader,criterion)
        scheduler.step()
        print('EPOCH ',epoch)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('LWLRAP/valid', val_LWLRAP, epoch)

        writer.add_scalar('AUROC/valid', val_scores['AUROC'], epoch)
        writer.add_scalar('F1_score/valid', val_scores['F1_score'], epoch)

        for param_group in optimizer.param_groups:
            writer.add_scalar('LR',param_group["lr"],epoch)

        if val_scores['AUROC'] > best_val_AUROC:
            print(f"saving as we have {val_scores['AUROC']} val_AUROC which is improvement over {best_val_AUROC}")
            best_val_AUROC = val_scores['AUROC']
            
            torch.save(
                        model.state_dict(),
                        f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/model_AUC_{fold}.pth",)

        if val_scores['F1_score'] > best_val_F1_score:
            print(f"saving as we have {val_scores['F1_score']} val_F1_score which is improvement over {best_val_F1_score}")
            best_val_F1_score = val_scores['F1_score']
            
            torch.save( model.state_dict(),
                        f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/model_F1_{fold}.pth",)

    ### now we do the master check once . it should be slow so we do it once
    print("### Training ended ###")
    del model
    gc.collect()

    writer.add_text('description',f'Here the  AUROC {best_val_AUROC} F1 {best_val_F1_score} ',EPOCH)
    writer.close()
    
    
    
if __name__ == "__main__":

    args = parser.parse_args()
    SEED = args.seed[0]
    print('setting seed to ',SEED)
    set_seed(SEED)
    CUDA_DEVICE = args.cuda[0]
    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

    config = configparser.ConfigParser()
    config.read(args.config_file[0])


    FOLDS = config['general']['fold']
    DATA_PATH = config['general']['data_path']
    BATCH_SIZE = config['general']['batch_size']
    WORKERS = config['general']['workers']
    EPOCH = config['general']['epoch']
    WEIGHT_SAVE = config['general']['weight_save_version']
    LR = config['general']['lr']
    train_base_df = config['general']['data_csv']

    if not os.path.exists(f"weights/{WEIGHT_SAVE}"):
        os.mkdir(f"weights/{WEIGHT_SAVE}")


    for fold in FOLDS:
        run(fold)