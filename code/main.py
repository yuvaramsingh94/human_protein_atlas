
import torch
from utils import set_seed, score_metrics, hpa_dataset_v1
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
import random



parser = argparse.ArgumentParser(description='set seed.')
parser.add_argument('seed', metavar='N', type=int, nargs='+',
                    help='seed to use')
parser.add_argument('cuda', metavar='N', type=int, nargs='+',
                    help='cuda device to use')
parser.add_argument('config_file', metavar='N', type=str, nargs='+',
                    help='configuration file path')


def train(model,train_dataloader,optimizer,criterion):
    model.train()
    #print('model.training = ',model.training)
    train_loss_loop_list = []
    for data_t in tqdm(train_dataloader):
        X, Y = data_t['image'],data_t['label']
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device, dtype=torch.float)
        X = X.permute(0,1,4,2,3)
        #print('Y shape ',Y)
        optimizer.zero_grad()
        prediction = model(X)
        train_loss = criterion(prediction['final_output'], Y) 
        train_loss.backward()
        optimizer.step()
        train_loss_loop_list.append(train_loss.item())

    train_total_loss = np.array(train_loss_loop_list)
    train_total_loss = train_total_loss.sum() / len(train_total_loss)

    print(f" \n train loss : {train_total_loss}")
    return train_total_loss


def validation(model,valid_dataloader,criterion):
    model.eval()
    #print('model.training = ',model.training)
    valid_loss_loop_list = []
    AUROC_loop_list = []
    F1_loop_list = []
    with torch.no_grad():
        for data_t in tqdm(valid_dataloader):

            X, Y = data_t['image'],data_t['label']

            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, dtype=torch.float)
            X = X.permute(0,1,4,2,3)
            prediction = model(X)

            valid_loss = criterion(prediction['final_output'], Y)
            valid_loss_loop_list.append(valid_loss.detach().cpu().item())

            scores = score_metrics(prediction['final_output'], Y)

            AUROC_loop_list.append(scores['AUROC']  )
            F1_loop_list.append(scores['F1_score']  )


    valid_total_loss = np.array(valid_loss_loop_list)
    valid_total_loss = valid_total_loss.sum() / len(valid_total_loss)

    AUROC_loop_list = np.array(AUROC_loop_list)
    AUROC_loop_list = AUROC_loop_list.sum() / len(AUROC_loop_list)

    F1_loop_list = np.array(F1_loop_list)
    F1_loop_list = F1_loop_list.sum() / len(F1_loop_list)


    #valid_total_loss = 0.0
    print(f" \n valid loss : {valid_total_loss}")
    return valid_total_loss, {'AUROC':AUROC_loop_list,'F1_score':F1_loop_list}


def run(fold):

    train_df = train_base_df[train_base_df['fold'] != fold]
    valid_df = train_base_df[train_base_df['fold'] == fold]

    train_dataset = hpa_dataset_v1(main_df = train_df[:20], path = DATA_PATH, augmentation = None, aug_per= 0.0, cells_used = 8)
    valid_dataset = hpa_dataset_v1(main_df = valid_df[:20], path = DATA_PATH, cells_used = 8, is_validation = True)

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
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )
    model = HpaModel(classes = int(config['general']['classes']), device = device, 
                        base_model_name = config['general']['pretrained_model'], 
                        features = int(config['general']['feature']), pretrained = True)
    model = model.to(device)
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr= LR)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    best_val_AUROC = 0.0
    best_val_F1_score = 0.0
    for epoch in range (EPOCH):
        train_loss = train(model,train_dataloader,optimizer,criterion)
        val_loss,val_scores = validation(model,valid_dataloader,criterion)
        scheduler.step()
        print('EPOCH ',epoch)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)

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
    print(os.getcwd())
    args = parser.parse_args()
    SEED = args.seed[0]
    print('setting seed to ',SEED)
    set_seed(SEED)
    CUDA_DEVICE = args.cuda[0]
    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

    config = configparser.ConfigParser()
    config.read(args.config_file[0])


    FOLDS = int(config['general']['folds'])
    DATA_PATH = config['general']['data_path']
    BATCH_SIZE = int(config['general']['batch_size'])
    WORKERS = int(config['general']['workers'])
    EPOCH = int(config['general']['epoch'])
    WEIGHT_SAVE = config['general']['weight_save_version']
    LR = float(config['general']['lr'])
    print('LR ',LR, type(LR))
    train_base_df = pd.read_csv(config['general']['data_csv'])
    #train_base_df = pd.read_csv('data/train_fold_v1.csv')
    if config['general']['loss'] == 'BCE':
        criterion = nn.BCELoss()

    if not os.path.exists(f"weights/{WEIGHT_SAVE}"):
        os.mkdir(f"weights/{WEIGHT_SAVE}")


    for fold in range(FOLDS):
        print('FOLD ',fold)
        print('This is the first rand no ',random.randint(2,50))
        print('This is the sec rand no ',random.randint(2,50))
        print('This is the 3 rand no ',random.randint(2,50))
        run(fold)