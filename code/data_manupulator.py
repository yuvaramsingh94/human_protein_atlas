import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import random 
from utils import set_seed

SEED = 1
FOLDS = 3
set_seed(SEED)

train_fold_v2 = pd.read_csv('data/train_fold_v2.csv')
print('initial shape ',train_fold_v2.shape)
labels = [str(i) for i in range(19)]
train_fold_v2['is_single'] = train_fold_v2[[str(i) for i in range(0,19)]].apply(np.sum, axis=1)
v = (train_fold_v2['0'] ==1) & (train_fold_v2['16'] ==1) & (train_fold_v2['is_single'] ==2)
train_1_16_class = train_fold_v2[~v]
print(train_fold_v2[(train_fold_v2['0'] ==1) & (train_fold_v2['is_single'] ==1)].shape)
print(train_fold_v2[(train_fold_v2['16'] ==1) & (train_fold_v2['is_single'] ==1)].shape)

v_0 = train_fold_v2[(train_fold_v2['0'] ==1) & (train_fold_v2['is_single'] ==1)].sample(n=1000, random_state=1)
v_16 = train_fold_v2[(train_fold_v2['16'] ==1) & (train_fold_v2['is_single'] ==1)].sample(n=500, random_state=1)

train_1_16_class = train_1_16_class[~train_1_16_class['ID'].isin(v_0["ID"].unique())]
train_1_16_class = train_1_16_class[~train_1_16_class['ID'].isin(v_16["ID"].unique())]
print('final shape ',train_1_16_class.shape)


skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, 
          random_state=SEED)
#print(labels)
train_1_16_class = train_1_16_class.reset_index(drop = True)
train_1_16_class['fold'] = -1
for fold,(idxT,idxV) in enumerate( skf.split(train_1_16_class,train_1_16_class[labels].values)):
    train_1_16_class['fold'][idxV] = fold
train_1_16_class = train_1_16_class.reset_index(drop = True)



train_1_16_class.to_csv('data/train_fold_v3.csv',index = False)