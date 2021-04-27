
import torch
from utils import set_seed, score_metrics, hpa_dataset_v1, focal_loss, ImprovedPANNsLoss, score_pr
from model import  HpaModel_1#, HpaModel_1, HpaModel_2
import pandas as pd
import os
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import gc
import shutil
import argparse
import configparser
import random
# https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py
import albumentations as albu
from augmix import RandomAugMix
from torch.backends import cudnn
cudnn.benchmark = True



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
        #print(X[:,:,:4,:,:].min(), X.max())
        
        optimizer.zero_grad(set_to_none=True)#better mode
        with torch.cuda.amp.autocast():
            prediction = model(X)
            train_loss = criterion(prediction, Y) 
        
        scaler.scale(train_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()

        #train_loss.backward()
        #optimizer.step()
        #print(model.init_layer.weight)
        #train_loss_loop_list.append(train_loss.item())

    #train_total_loss = np.array(train_loss_loop_list)
    #train_total_loss = train_total_loss.sum() / len(train_total_loss)
    train_total_loss = 0.0
    print(f" \n train loss : {train_total_loss}")
    return train_total_loss


def validation(model,valid_dataloader,criterion):
    model.eval()
    #print('model.training = ',model.training)
    valid_loss_loop_list = []
    AUROC_loop_list = []
    F1_loop_list = []
    labels = []
    predictions = []
    with torch.no_grad():
        for data_t in tqdm(valid_dataloader):

            X, Y = data_t['image'],data_t['label']

            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, dtype=torch.float)
            X = X.permute(0,1,4,2,3)
            #print(X[:,:,:4,:,:].min(), X.max())
            
            with torch.cuda.amp.autocast():
                prediction = model(X)
                valid_loss = criterion(prediction, Y)
                
            valid_loss_loop_list.append(valid_loss.detach().cpu().item())

            labels.append(Y.detach().cpu().numpy())
            predictions.append(torch.sigmoid(prediction['final_output']).detach().cpu().numpy())
            #scores = score_metrics(torch.sigmoid(prediction['final_output']), Y)

            #AUROC_loop_list.append(scores['AUROC']  )
            #F1_loop_list.append(scores['F1_score']  )

    scores_val = score_pr(np.concatenate(predictions, axis = 0), np.concatenate(labels, axis = 0), n_classes = 19)
    valid_total_loss = np.array(valid_loss_loop_list)
    valid_total_loss = valid_total_loss.sum() / len(valid_total_loss)



    #valid_total_loss = 0.0
    print(f" \n valid loss : {valid_total_loss}")
    return valid_total_loss, scores_val

def val_oof(fold, metrics):
    
    if config['general']['model'] == 'HpaModel_1':
        print('using ',config['general']['model'])
        model = HpaModel_1(classes = int(config['general']['classes']), device = device, 
                            base_model_name = config['general']['pretrained_model'], 
                            features = int(config['general']['feature']), pretrained = True,)
        model = model.to(device)
    elif config['general']['model'] == 'HpaModel':
        print('using ',config['general']['model'])
        model = HpaModel(classes = int(config['general']['classes']), device = device, 
                            base_model_name = config['general']['pretrained_model'], 
                            features = int(config['general']['feature']), pretrained = True,)
        model = model.to(device)
    elif config['general']['model'] == 'HpaModel_2':
        print('using ',config['general']['model'])
        model = HpaModel_2(classes = int(config['general']['classes']), device = device, 
                            base_model_name = config['general']['pretrained_model'], 
                            features = int(config['general']['feature']), pretrained = True,)
    
    
    
    model.load_state_dict(torch.load(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/model_{metrics}_{fold}.pth",map_location = device))
    model.to(device)
    model.eval()                        

    valid_df = train_base_df[train_base_df['fold'] == fold]
    valid_dataset = hpa_dataset_v1(main_df = valid_df, path = DATA_PATH, 
                                    cells_used = cells_used, is_validation = True)
    valid_dataloader = data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )
    prediction_list = []
    with torch.no_grad():
        for data_t in tqdm(valid_dataloader):

            X, Y = data_t['image'],data_t['label']

            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, dtype=torch.float)
            X = X.permute(0,1,4,2,3)
            with torch.cuda.amp.autocast():
                prediction = torch.sigmoid(model(X)['final_output'])#model(X)
            prediction_list.append(prediction.detach().cpu().numpy()) 
    predictions = np.concatenate(prediction_list, axis=0)
    
    valid_df[[str(f'pred_{i}') for i in range(int(config['general']['classes']))]] = predictions
    return valid_df

def get_sample_counts(df, class_names):
    """
    Get total and class-wise positive sample count of a dataset
    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes
    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    
    total_count = df.shape[0]
    labels = df[class_names].values
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }
    def get_single_class_weight2(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return (denominator - pos_counts) / denominator

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight2(label_counts[i], total_counts))

    return class_weights

def run(fold):

    train_df = train_base_df[train_base_df['fold'] != fold]

    ## here we will upsample class 15 and 11 to see what it does to me 
    print('This is training shape ',train_df.shape)
    if config.getboolean('general','is_resample'):
        #print('15 class ',train_df[train_df['15'] == 1].shape)
        print('11 class ',train_df[train_df['11'] == 1].shape)
        #train_15 = train_df[train_df['15'] == 1].sample(n=500, replace=True, random_state=1)
        train_11 = train_df[train_df['11'] == 1].sample(n=100, replace=True, random_state=1)
        train_df = pd.concat([train_df,train_11])
        print('This is resampled training shape ',train_df.shape)
    else:
        print("NO RESAMPLING")

    valid_df = train_base_df[train_base_df['fold'] == fold]

    print(f'Train df {train_df.shape} valid df {valid_df.shape}')

    
    train_dataset = hpa_dataset_v1(main_df = train_df, path = DATA_PATH, augmentation = aug_fn, aug_per= 0.8, 
                                    cells_used = cells_used,label_smoothing = config.getboolean('general','label_smoothing'),
                                     l_alp = 0.3, size = int(config['general']['size']), cell_repetition = config.getboolean('general','cell_repetition'))
    valid_dataset = hpa_dataset_v1(main_df = valid_df, path = DATA_PATH, cells_used = cells_used, is_validation = True, 
                                size = int(config['general']['size']), cell_repetition = config.getboolean('general','cell_repetition'))

    if not os.path.exists(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}"):
        os.mkdir(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}")

    if  os.path.exists(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/log_dir"):
        shutil.rmtree(f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/log_dir")

    class_weights = None
    if config.getboolean('general','class_weights'):
        class_weights = ((train_df[[f'{i}' for i in range(0,19)]].sum().min())/train_df[[f'{i}' for i in range(0,19)]].sum()).values
        #print('this is class weights ',class_weights)

        total_count, class_positive_counts = get_sample_counts(train_df, [f'{i}' for i in range(0,19)])
        print('weight calcualtions')
        print(total_count, class_positive_counts)
        class_weights = get_class_weights(total_count, class_positive_counts, multiply = 1)
        print('class_weights')
        print(class_weights)

    if config['general']['loss'] == 'BCE':
        #criterion = nn.BCELoss().cuda()
        if class_weights != None:
            criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights, requires_grad = False)).cuda(device=device)
        else:
            criterion = nn.BCEWithLogitsLoss().cuda(device=device)
    elif config['general']['loss'] == 'MSE':
        criterion = torch.nn.MSELoss().cuda(device=device)
    elif config['general']['loss'] == 'focal':
        print('Using Focal loss')
        ## basic alp 0.25 gam 2 
        criterion = focal_loss(alpha=0.1, gamma=5).cuda(device=device)
    elif config['general']['loss'] == 'focal_improved':
        print('Using focal_improved loss')
        ## basic alp 0.25 gam 2 
        criterion = ImprovedPANNsLoss(device = device).cuda(device=device)

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
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
        pin_memory=True,
        
    )

    if config['general']['model'] == 'HpaModel_1':
        print('using ',config['general']['model'])
        model = HpaModel_1(classes = int(config['general']['classes']), device = device, 
                            base_model_name = config['general']['pretrained_model'], 
                            features = int(config['general']['feature']), pretrained = True,
                            spe_drop = float(config['general']['spe_drop']), 
                            att_drop = float(config['general']['att_drop']),
                            hidden_dropout_prob = float(config['general']['hidden_dropout_prob']))
        model = model.to(device)
    elif config['general']['model'] == 'HpaModel':
        print('using ',config['general']['model'])
        model = HpaModel(classes = int(config['general']['classes']), device = device, 
                            base_model_name = config['general']['pretrained_model'], 
                            features = int(config['general']['feature']), pretrained = True,)
        model = model.to(device)
    elif config['general']['model'] == 'HpaModel_2':
        print('using ',config['general']['model'])
        model = HpaModel_2(classes = int(config['general']['classes']), device = device, 
                            base_model_name = config['general']['pretrained_model'], 
                            features = int(config['general']['feature']), pretrained = True,)
        model = model.to(device)
    # Optimizer
    #optimizer = optim.AdamW(model.parameters(), lr= LR)
    #param_groups = model.trainable_parameters()
    #optimizer0 = optim.AdamW(param_groups[0], lr= 1e-5)
    #optimizer1 = optim.AdamW(param_groups[1], lr= LR)
    #https://github.com/pytorch/pytorch/tree/master/torch/optim/_multi_tensor
    optimizer = optim._multi_tensor.AdamW(model.parameters(), lr= LR)
    

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    
    best_val_loss = 10000.0
    improvement_tracker = 0
    for epoch in range (EPOCH):
        train_loss = train(model,train_dataloader,optimizer,criterion)
        val_loss,scores_val = validation(model,valid_dataloader,criterion)
        
        '''
        precision_ = scores_val['precision'] 
        recall_ = scores_val['recall']
        auc_pr_ = scores_val['auc']
        
        val_precision = dict()
        val_recall = dict()
        val_auc_pr = dict()

        for nk in range(int(config['general']['classes'])):

            val_precision[nk] = precision_[nk]
            val_recall[nk] = recall_[nk]
            val_auc_pr[nk] = auc_pr_[nk]
        '''
        scheduler.step()
        print('EPOCH ',epoch)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)

        #writer.add_scalar('AUROC/valid', val_scores['AUROC'], epoch)
        #writer.add_scalar('F1_score/valid', val_scores['F1_score'], epoch)
        #print(scores_val['precision'])
        #writer.add_scalars('precision', scores_val['precision'] , epoch)
        #writer.add_scalars('recall', scores_val['recall'], epoch)
        #writer.add_scalars('auc', scores_val['auc'], epoch)
        avg_pr = []
        for ig in range(19):
            avg_pr.append(scores_val['avg_precision'][str(ig)])
        avg_pr = np.array(avg_pr)
        print('mean ',avg_pr.mean())
        writer.add_scalars('avg_precision', scores_val['avg_precision'], epoch)
        writer.add_scalar('mean_AP', avg_pr.mean(), epoch)
        for param_group in optimizer.param_groups:
            #print('this is param_group ',param_group)
            writer.add_scalar('LR',param_group["lr"],epoch)
        '''
        if val_scores['AUROC'] > best_val_AUROC:
            print(f"saving as we have {val_scores['AUROC']} val_AUROC which is improvement over {best_val_AUROC}")
            best_val_AUROC = val_scores['AUROC']
            
            #torch.save(
            #            model.state_dict(),
            #            f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/model_AUC_{fold}.pth",)

        if val_scores['F1_score'] > best_val_F1_score:
            print(f"saving as we have {val_scores['F1_score']} val_F1_score which is improvement over {best_val_F1_score}")
            best_val_F1_score = val_scores['F1_score']
            
            #torch.save( model.state_dict(),
            #            f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/model_F1_{fold}.pth",)
        '''
        torch.save( model.state_dict(),
                        f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/model_epoch_{epoch}.pth",)
        if val_loss < best_val_loss:
            improvement_tracker = 0
            print(f"saving as we have {val_loss} val_loss which is improvement over {best_val_loss}")
            
            best_val_loss = val_loss
            
            torch.save( model.state_dict(),
                        f"weights/{WEIGHT_SAVE}/fold_{fold}_seed_{SEED}/best_loss_{fold}.pth",)
        else:
            improvement_tracker += 1
        print('improvement_tracker ',improvement_tracker)
        #early stoping
        if improvement_tracker > 5:# if we are not improving for more than 6 
            break

    ### now we do the master check once . it should be slow so we do it once
    print("### Training ended ###")
    del model
    gc.collect()

    writer.add_text('description',f'Here the loss {best_val_loss}',epoch)
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
    cells_used = int(config['general']['cells_used'])
    print('LR ',LR, type(LR))
    print('init_linear_comb', config.getboolean('general','init_linear_comb'), type(config.getboolean('general','init_linear_comb')))
    train_base_df = pd.read_csv(config['general']['data_csv'])
    #train_base_df = pd.read_csv('data/train_fold_v1.csv')

    


    if not os.path.exists(f"weights/{WEIGHT_SAVE}"):
        os.mkdir(f"weights/{WEIGHT_SAVE}")
    
    aug_fn = albu.Compose(
        [
            #'''
            RandomAugMix(p=.5),
            #albu.OneOf([
            #    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.4, rotate_limit=40, border_mode = 1),
            #    albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, border_mode=1),
            #    albu.GridDistortion(num_steps=3, distort_limit=0.4, interpolation=1, border_mode=1),#num_steps=5, distort_limit=0.3
            #], p=.5),
            
            #'''
            albu.HorizontalFlip(p=.5),
            albu.VerticalFlip(p=.5),
            albu.Cutout(
                num_holes=12,
                max_h_size=16,
                max_w_size=16,
                fill_value=0,
                p=0.5,
            ),
            albu.ToFloat(max_value=255.,always_apply=True),
        ]
    )
    

    #for amp
    scaler = torch.cuda.amp.GradScaler()

    for fold in range(FOLDS):
        print('FOLD ',fold)
        print('This is the first rand no ',random.randint(2,50))
        print('This is the sec rand no ',random.randint(2,50))
        print('This is the 3 rand no ',random.randint(2,50))
        run(fold)

    oof_list = []
    for fold in range(1,FOLDS):
        val_o_df = val_oof(fold, metrics = 'loss')
        
        oof_list.append(val_o_df)
    
    ## now we cancatenate the prediction datafram and save it in one 
    oof_df = pd.concat(oof_list)
    oof_df.to_csv(f"weights/{WEIGHT_SAVE}/oof_seed_{SEED}.csv", index = False)