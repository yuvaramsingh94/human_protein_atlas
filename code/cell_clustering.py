import pandas as pd
import numpy as np
from model import HpaModel
import torch
import torch.utils.data as data
import h5py
from tqdm import tqdm
import os
import albumentations as albu
import matplotlib.pyplot as plt
import sklearn#.cluster import DBSCAN
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

feature_multi = pd.read_csv('data/cell_feature_extracted_v1.csv')

pca_2d = PCA(n_components=2)
pca_2d_viz = pca_2d.fit_transform(feature_multi[[str(i) for i in range(1,2048+1)]].values)

filename_unique = feature_multi['ID'].unique()

for f in filename_unique:
    mini_df = feature_multi[feature_multi['ID'].isin([f])]
    label = len(mini_df.iloc[0]['Label'].split('|'))
    km_cluster = KMeans(n_clusters = label).fit_predict(mini_df[[str(i) for i in range(1,2048+1)]].values)
    
    feature_multi.loc[feature_multi['ID'].isin([f]),'cluster'] = km_cluster

feature_multi = feature_multi.drop([str(i) for i in range(1,2048+1)],axis=1)
feature_multi.to_csv('data/features_multi_label.csv',index=False)