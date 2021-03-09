from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import random 
import os
import numpy as np
import pandas as pd
from utils import set_seed


SEED = 1
FOLDS = 5
set_seed(SEED)
train_df = pd.read_csv('data/train.csv')
labels = [str(i) for i in range(19)]
for x in labels: train_df[x] = train_df['Label'].apply(lambda r: int(x in r.split('|')))
print(train_df.shape)

skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, 
          random_state=SEED)
print(labels)

train_df['fold'] = -1
for fold,(idxT,idxV) in enumerate( skf.split(train_df,train_df[labels].values)):
    train_df['fold'][idxV] = fold

print(train_df.head())

train_df.to_csv('data/train_fold_v1.csv')