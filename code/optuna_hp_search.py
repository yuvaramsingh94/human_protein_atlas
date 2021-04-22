import torch
import torch.optim as optim
from utils import set_seed,  hpa_dataset_v1, focal_loss
import pandas as pd
from model import HpaModel_1
import os
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from torch.cuda import amp
import torch.nn as nn
import torch.nn.functional as F
import optuna
import albumentations as albu
from augmix import RandomAugMix
import math
from torch.backends import cudnn

cudnn.benchmark = True


### amp
scaler = amp.GradScaler()





def train(model,train_dataloader,optimizer,criterion):
    model.train()
    #print('model.training = ',model.training)
    train_loss_loop_list = []
    for data_t in tqdm(train_dataloader):
        X, Y = data_t['image'],data_t['label']
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device, dtype=torch.float)
        X = X.permute(0,1,4,2,3)
        #print(X[:,:,:4,:,:].min(), X.max())
        
        optimizer.zero_grad(set_to_none=True)#better mode
        with torch.cuda.amp.autocast():
            prediction = model(X)
            train_loss = criterion(prediction['final_output'], Y) 
        
        scaler.scale(train_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()

        #train_loss.backward()
        #optimizer.step()
        #print(model.init_layer.weight)
        train_loss_loop_list.append(train_loss.item())

    train_total_loss = np.array(train_loss_loop_list)
    train_total_loss = train_total_loss.sum() / len(train_total_loss)

    #print(f" \n train loss : {train_total_loss}")
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
            #print(X[:,:,:4,:,:].min(), X.max())
            
            with torch.cuda.amp.autocast():
                prediction = model(X)
                valid_loss = criterion(prediction['final_output'], Y)
                
            valid_loss_loop_list.append(valid_loss.detach().cpu().item())



    valid_total_loss = np.array(valid_loss_loop_list)
    valid_total_loss = valid_total_loss.sum() / len(valid_total_loss)

    return valid_total_loss


def run(trial):

    train_df = train_base_df[train_base_df['fold'] != fold]
    valid_df = train_base_df[train_base_df['fold'] == fold]

    train_dataset = hpa_dataset_v1(main_df = train_df, path = DATA_PATH, augmentation = aug_fn, aug_per= 0.8, 
                                    cells_used = cells_used,label_smoothing = False,
                                     l_alp = 0.0, size =  256, cell_repetition = False)
    valid_dataset = hpa_dataset_v1(main_df = valid_df, path = DATA_PATH, cells_used = cells_used, is_validation = True, size = 256, cell_repetition = False)

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
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.5, log=True)
    gamma = trial.suggest_int("gamma", 1, 5, log=True)
    spe_drop = trial.suggest_float("spe_drop", 0.2, 0.8, log=True)
    att_drop = trial.suggest_float("att_drop", 0.2, 0.8, log=True)
    hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.2, 0.8, log=True)
    model = HpaModel_1(classes = 19, device = device, 
                            base_model_name = 'resnet18', 
                            features = 512, pretrained = True, spe_drop = spe_drop, att_drop = att_drop, 
                            hidden_dropout_prob = hidden_dropout_prob)
    model = model.to(device)

    

    optimizer = optim._multi_tensor.AdamW(params=model.parameters(), lr= lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = focal_loss(alpha=alpha, gamma=gamma).cuda(device=device)
 
    for epoch in range(EPOCH):
        train_loss = train(model, train_dataloader, optimizer, criterion)
        valid_loss = validation(model, valid_dataloader, criterion)
        print(f'Training {train_loss} valid {valid_loss}')
        #valid_loss = float('nan')
        if math.isnan(valid_loss):
            valid_loss = 10000.
        
        trial.report(valid_loss, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return valid_loss

if __name__ == "__main__":

    train_base_df = pd.read_csv('data/train_fold_v6.csv')
    #filtered_df = pd.read_csv('data/train_fold_v6.csv')
    DATA_PATH = 'data/train_h5_256_40000_v5'
    BATCH_SIZE = 24
    WORKERS = 10
    cells_used = 8
    SEED = 1
    EPOCH = 5
    CUDA_DEVICE = 0
    fold = 0
    set_seed(SEED)
    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

    aug_fn = albu.Compose(
        [
            RandomAugMix(p=.5),
            #albu.OneOf([
            #    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.4, rotate_limit=40, border_mode = 1),
            #    albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, border_mode=1),
            #    albu.GridDistortion(num_steps=3, distort_limit=0.4, interpolation=1, border_mode=1),#num_steps=5, distort_limit=0.3
            #], p=.5),
            
            
            albu.HorizontalFlip(p=.5),
            albu.VerticalFlip(p=.5),
            albu.Cutout(
                num_holes=12,
                max_h_size=16,
                max_w_size=16,
                fill_value=0,
                p=0.5,
            ),
            #albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.4, rotate_limit=40, p=0.7),
            albu.ToFloat(max_value=255.,always_apply=True),
        ]
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(run, n_trials=20,)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))