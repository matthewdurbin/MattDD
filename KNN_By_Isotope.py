# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:38:28 2020

@author: matth

This script should load a master data set of Co, Cs, Ir data,
divide into isotope specific data sets,
and allow for training/testing on various combinations using
a stratified k fold split

Dataset is stored locally at the defined Path
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
import os
#%% Load data
path = "E:\\MattD\\NSS_DD"
os.chdir(path)
print("Loading data...")
sim_data = np.load('dataset_CoCsIr.npy')
ib_l, ib_u = 0, 10 # input features for isotope
db_l, db_u = 10, 14 # input features for detector totals
ab = -2 # angle 
rb = 3 # radius

#%% Create Isotope specific data
"""
The main dataset has 10,000 trials each of Co, Cs, Ir, each with
the same labels (angles). Thus 3 xa's are created, and only one ya.
"""
xa_co = sim_data[:10000,db_l:db_u]/np.sum(sim_data[:10000,db_l:db_u], axis=1)[:, None]
xa_cs = sim_data[10000:20000,db_l:db_u]/np.sum(sim_data[10000:20000,db_l:db_u], axis=1)[:, None]
xa_ir = sim_data[20000:,db_l:db_u]/np.sum(sim_data[20000:,db_l:db_u], axis=1)[:, None]
ya = np.round(sim_data[:10000,ab])
ra = sim_data[:10000,rb]

#%% KNN - Stratified K Fold
nfold_a = 5
knn_a = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
skf_a = StratifiedKFold(n_splits=nfold_a, shuffle=True)

# Train on Co, test on Co
print("Train/test: Co/Co")
for fno, (train_index, test_index) in enumerate(skf_a.split(xa_co, ya)):
    print("folding", fno+1, "/", nfold_a)
    xa_tr, ya_tr, ra_tr = xa_co[train_index], ya[train_index], ra[train_index]
    xa_ts, ya_ts, ra_ts = xa_co[test_index], ya[test_index], ra[test_index]
    knn_a.fit(xa_tr, ya_tr)
    pred_a = knn_a.predict(xa_ts)
    acc_a = accuracy_score(ya_ts, pred_a)
    print("Accuracy: ", acc_a)
    
# Train on Co, test on Cs
print("Train/test: Co/Cs")
for fno, (train_index, test_index) in enumerate(skf_a.split(xa_co, ya)):
    print("folding", fno+1, "/", nfold_a)
    xa_tr, ya_tr, ra_tr = xa_co[train_index], ya[train_index], ra[train_index]
    xa_ts, ya_ts, ra_ts = xa_cs[test_index], ya[test_index], ra[test_index]
    knn_a.fit(xa_tr, ya_tr)
    pred_a = knn_a.predict(xa_ts)
    acc_a = accuracy_score(ya_ts, pred_a)
    print("Accuracy: ", acc_a)
