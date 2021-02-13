# -*- coding: utf-8 -*-
"""
Run simulated expirements for NSS work:
    -Simple Isotope Predictor
    -Angular Predictor
    -Test Co60, Cs137, and Ir192 datasets on individual and combined
    datasets (Ex: test Co on Co, Cs, Ir, and combined)
    -Run Least Squared Reference Table for comparrison
    Test on untrained isotope: Na22
    
Author: Matthew Durbin
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import os
from scipy import stats
import knn_4det_functions as kf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load Data
print("Loading data...")
sim_data = np.load("dataset_CoCsIr.npy")  # Simulated data
na_data = np.load("dataset_Na.npy")  # Sodium-22 only Data
ref135_co, ref135_cs, ref135_ir = (
    np.load("Ref135_Co.npy"),
    np.load("Ref135_Cs.npy"),
    np.load("Ref135_Ir.npy"),
)  # reference table data

# define dumby variables
ebin_l, ebin_u = 0, 10  # Energy bins: isotope input feautres
total_l, total_u = 10, 14  # Total counts: angular predictor input features
isotope = -1  # isotope
angle = -2  # angle
radius = -3  # radius
refb_l, refb_u = 360, 720  # selects a 3 m reference table

# Assign and Normalize Simulated Data - Isotope
xi, yi, ri = (
    kf.norm(sim_data[:, ebin_l:ebin_u]),
    sim_data[:, isotope],
    sim_data[:, radius],
)
xi_na, yi_na, ri_na = (
    kf.norm(na_data[:, ebin_l:ebin_u]),
    na_data[:, isotope],
    na_data[:, radius],
)

# SKFold KNN - Isotope Predictor
print("Initiating Isotope Predictor")
nfold_i = 5
knn_i = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
skf_i = StratifiedKFold(n_splits=nfold_i, shuffle=True)
for fno, (train_index, test_index) in enumerate(skf_i.split(xi, yi)):
    print("folding", fno + 1, "/", nfold_i)
    xi_tr, yi_tr, ri_tr = xi[train_index], yi[train_index], ri[train_index]
    xi_ts, yi_ts, ri_ts = xi[test_index], yi[test_index], ri[test_index]
    plt.plot(train_index, ".")
    knn_i.fit(xi_tr, yi_tr)
    pred_i = knn_i.predict(xi_ts)
    acc_i = accuracy_score(yi_ts, pred_i)
    print("Accuracy: ", acc_i)

print("*********************************************")
print("*********************************************")
print("Isotope Predictor Accuracy: ", acc_i * 100, "%")

# Test Isotope Predictor on untrained Na data
knn_i_na = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_i_na.fit(xi, yi)
pred_i_na = knn_i.predict(xi_na)
na_guess = stats.mode(pred_i_na)
print("Mode Soidum Prediction: ", int(na_guess[0]))
print("*********************************************")
print("*********************************************")

# Assign and Normalize Simulated Data for Single Isotopes - Angle
xa_co = kf.norm(sim_data[:10000, total_l:total_u])
xa_cs = kf.norm(sim_data[10000:20000, total_l:total_u])
xa_ir = kf.norm(sim_data[20000:, total_l:total_u])
xa_na = kf.norm(na_data[:, total_l:total_u])
ya = np.round(sim_data[:10000, angle])
ra = sim_data[:10000, radius]

ref_co = kf.norm(ref135_co[refb_l:refb_u, :4])
ref_cs = kf.norm(ref135_cs[refb_l:refb_u, :4])
ref_ir = kf.norm(ref135_ir[refb_l:refb_u, :4])

# Create Combined Dataset
x_com_us = np.empty((10000, 4, 3))
x_com_us[:, :, 0] = xa_co
x_com_us[:, :, 1] = xa_cs
x_com_us[:, :, 2] = xa_ir

x_com = np.empty((xa_co.shape))
for i in range(len(x_com)):
    x_com[i, :] = x_com_us[i, :, np.random.randint(0, 2 + 1)]

r_com_us = np.empty((360, 4, 3))
r_com_us[:, :, 0] = ref_co
r_com_us[:, :, 1] = ref_cs
r_com_us[:, :, 2] = ref_ir

ref_com = np.empty((ref_co.shape))
for i in range(len(ref_com)):
    ref_com[i, :] = r_com_us[i, :, np.random.randint(0, 2 + 1)]


# test the knn
print("*********************************************")
print("Testing with Co")
print("*********************************************")
acc_coco, err_coco = kf.ttKNN(xa_co, xa_co, ref_co, ya, ra)
acc_cocs, err_cocs = kf.ttKNN(xa_co, xa_cs, ref_cs, ya, ra)
acc_coir, err_coir = kf.ttKNN(xa_co, xa_ir, ref_ir, ya, ra)
acc_cocom, err_cocom = kf.ttKNN(xa_co, x_com, ref_com, ya, ra)
print("*********************************************")
print("Testing with Cs")
print("*********************************************")
acc_csco, err_csco = kf.ttKNN(xa_cs, xa_co, ref_co, ya, ra)
acc_cscs, err_cscs = kf.ttKNN(xa_cs, xa_cs, ref_cs, ya, ra)
acc_csir, err_csir = kf.ttKNN(xa_cs, xa_ir, ref_ir, ya, ra)
acc_cscom, err_cscom = kf.ttKNN(xa_cs, x_com, ref_com, ya, ra)
print("*********************************************")
print("Testing with Ir")
print("*********************************************")
acc_irco, err_irco = kf.ttKNN(xa_ir, xa_co, ref_co, ya, ra)
acc_ircs, err_ircs = kf.ttKNN(xa_ir, xa_cs, ref_cs, ya, ra)
acc_irir, err_irir = kf.ttKNN(xa_ir, xa_ir, ref_ir, ya, ra)
acc_ircom, err_ircom = kf.ttKNN(xa_ir, x_com, ref_com, ya, ra)
print("*********************************************")
print("Testing with Na")
print("*********************************************")
acc_naco, err_naco = kf.ttKNN(xa_na, xa_co, ref_co, ya, ra)
acc_nacs, err_nacs = kf.ttKNN(xa_na, xa_cs, ref_cs, ya, ra)
acc_nair, err_nair = kf.ttKNN(xa_na, xa_ir, ref_ir, ya, ra)
acc_nacom, err_nacom = kf.ttKNN(xa_na, x_com, ref_com, ya, ra)


# Combine Results
print("Combining and Saving Results...")
all_acc = np.zeros((4, 5, 2))
all_acc[0, :, 0] = (
    np.mean(acc_coco[:, 0]),
    np.mean(acc_cocs[:, 0]),
    np.mean(acc_coir[:, 0]),
    np.mean(acc_cocom[:, 0]),
    np.mean(acc_coco[:, 0]),
)
all_acc[0, :, 1] = (
    np.std(acc_coco[:, 0]),
    np.std(acc_cocs[:, 0]),
    np.std(acc_coir[:, 0]),
    np.std(acc_cocom[:, 0]),
    np.std(acc_coco[:, 0]),
)
all_acc[1, :, 0] = (
    np.mean(acc_csco[:, 0]),
    np.mean(acc_cscs[:, 0]),
    np.mean(acc_csir[:, 0]),
    np.mean(acc_cscom[:, 0]),
    np.mean(acc_cscs[:, 0]),
)
all_acc[1, :, 1] = (
    np.std(acc_csco[:, 0]),
    np.std(acc_cscs[:, 0]),
    np.std(acc_csir[:, 0]),
    np.std(acc_cscom[:, 0]),
    np.std(acc_cscs[:, 0]),
)
all_acc[2, :, 0] = (
    np.mean(acc_irco[:, 0]),
    np.mean(acc_ircs[:, 0]),
    np.mean(acc_irir[:, 0]),
    np.mean(acc_ircom[:, 0]),
    np.mean(acc_irir[:, 0]),
)
all_acc[2, :, 1] = (
    np.std(acc_irco[:, 0]),
    np.std(acc_ircs[:, 0]),
    np.std(acc_irir[:, 0]),
    np.std(acc_ircom[:, 0]),
    np.std(acc_irir[:, 0]),
)

all_acc[3, :, 0] = (
    np.mean(all_acc[:-1, 0, 0]),
    np.mean(all_acc[:-1, 1, 0]),
    np.mean(all_acc[:-1, 2, 0]),
    np.mean(all_acc[:-1, 3, 0]),
    np.mean(all_acc[:-1, 4, 0]),
)
all_acc[3, :, 1] = (
    np.mean(all_acc[:-1, 0, 1]),
    np.mean(all_acc[:-1, 1, 1]),
    np.mean(all_acc[:-1, 2, 1]),
    np.mean(all_acc[:-1, 3, 1]),
    np.mean(all_acc[:-1, 4, 1]),
)

all_err = np.zeros((4, 5, 2))
all_err[0, :, 0] = (
    np.mean(err_coco[0, :, 0]),
    np.mean(err_cocs[0, :, 0]),
    np.mean(err_coir[0, :, 0]),
    np.mean(err_cocom[0, :, 0]),
    np.mean(err_coco[0, :, 0]),
)
all_err[0, :, 1] = (
    np.std(err_coco[0, :, 0]),
    np.std(err_cocs[0, :, 0]),
    np.std(err_coir[0, :, 0]),
    np.std(err_cocom[0, :, 0]),
    np.std(err_coco[0, :, 0]),
)
all_err[1, :, 0] = (
    np.mean(err_csco[0, :, 0]),
    np.mean(err_cscs[0, :, 0]),
    np.mean(err_csir[0, :, 0]),
    np.mean(err_cscom[0, :, 0]),
    np.mean(err_cscs[0, :, 0]),
)
all_err[1, :, 1] = (
    np.std(err_csco[0, :, 0]),
    np.std(err_cscs[0, :, 0]),
    np.std(err_csir[0, :, 0]),
    np.std(err_cscom[0, :, 0]),
    np.std(err_cscs[0, :, 0]),
)
all_err[2, :, 0] = (
    np.mean(err_irco[0, :, 0]),
    np.mean(err_ircs[0, :, 0]),
    np.mean(err_irir[0, :, 0]),
    np.mean(err_ircom[0, :, 0]),
    np.mean(err_irir[0, :, 0]),
)
all_err[2, :, 1] = (
    np.std(err_irco[0, :, 0]),
    np.std(err_ircs[0, :, 0]),
    np.std(err_irir[0, :, 0]),
    np.std(err_ircom[0, :, 0]),
    np.std(err_irir[0, :, 0]),
)


all_err[3, :, 0] = (
    np.mean(all_err[:-1, 0, 0]),
    np.mean(all_err[:-1, 1, 0]),
    np.mean(all_err[:-1, 2, 0]),
    np.mean(all_err[:-1, 3, 0]),
    np.mean(all_err[:-1, 4, 0]),
)
all_err[3, :, 1] = (
    np.mean(all_err[:-1, 0, 1]),
    np.mean(all_acc[:-1, 1, 1]),
    np.mean(all_err[:-1, 2, 1]),
    np.mean(all_err[:-1, 3, 1]),
    np.mean(all_err[:-1, 4, 1]),
)

all_accr = np.zeros((4, 5, 2))
all_accr[0, :, 0] = (
    np.mean(acc_coco[:, 1]),
    np.mean(acc_cocs[:, 1]),
    np.mean(acc_coir[:, 1]),
    np.mean(acc_cocom[:, 1]),
    np.mean(acc_coco[:, 1]),
)
all_accr[0, :, 1] = (
    np.std(acc_coco[:, 1]),
    np.std(acc_cocs[:, 1]),
    np.std(acc_coir[:, 1]),
    np.std(acc_cocom[:, 1]),
    np.std(acc_coco[:, 1]),
)
all_accr[1, :, 0] = (
    np.mean(acc_csco[:, 1]),
    np.mean(acc_cscs[:, 1]),
    np.mean(acc_csir[:, 1]),
    np.mean(acc_cscom[:, 1]),
    np.mean(acc_cscs[:, 1]),
)
all_accr[1, :, 1] = (
    np.std(acc_csco[:, 1]),
    np.std(acc_cscs[:, 1]),
    np.std(acc_csir[:, 1]),
    np.std(acc_cscom[:, 1]),
    np.std(acc_cscs[:, 1]),
)
all_accr[2, :, 0] = (
    np.mean(acc_irco[:, 1]),
    np.mean(acc_ircs[:, 1]),
    np.mean(acc_irir[:, 1]),
    np.mean(acc_ircom[:, 1]),
    np.mean(acc_irir[:, 1]),
)
all_accr[2, :, 1] = (
    np.std(acc_irco[:, 1]),
    np.std(acc_ircs[:, 1]),
    np.std(acc_irir[:, 1]),
    np.std(acc_ircom[:, 1]),
    np.std(acc_irir[:, 1]),
)

all_accr[3, :, 0] = (
    np.mean(all_accr[:-1, 0, 0]),
    np.mean(all_accr[:-1, 1, 0]),
    np.mean(all_accr[:-1, 2, 0]),
    np.mean(all_accr[:-1, 3, 0]),
    np.mean(all_accr[:-1, 4, 0]),
)
all_accr[3, :, 1] = (
    np.mean(all_accr[:-1, 0, 1]),
    np.mean(all_accr[:-1, 1, 1]),
    np.mean(all_accr[:-1, 2, 1]),
    np.mean(all_accr[:-1, 3, 1]),
    np.mean(all_accr[:-1, 4, 1]),
)


all_errr = np.zeros((4, 5, 2))
all_errr[0, :, 0] = (
    np.mean(err_coco[0, :, 1]),
    np.mean(err_cocs[0, :, 1]),
    np.mean(err_coir[0, :, 1]),
    np.mean(err_cocom[0, :, 1]),
    np.mean(err_coco[0, :, 1]),
)
all_errr[0, :, 1] = (
    np.std(err_coco[0, :, 1]),
    np.std(err_cocs[0, :, 1]),
    np.std(err_coir[0, :, 1]),
    np.std(err_cocom[0, :, 1]),
    np.std(err_coco[0, :, 1]),
)
all_errr[1, :, 0] = (
    np.mean(err_csco[0, :, 1]),
    np.mean(err_cscs[0, :, 1]),
    np.mean(err_csir[0, :, 1]),
    np.mean(err_cscom[0, :, 1]),
    np.mean(err_cscs[0, :, 1]),
)
all_errr[1, :, 1] = (
    np.std(err_csco[0, :, 1]),
    np.std(err_cscs[0, :, 1]),
    np.std(err_csir[0, :, 1]),
    np.std(err_cscom[0, :, 1]),
    np.std(err_cscs[0, :, 1]),
)
all_errr[2, :, 0] = (
    np.mean(err_irco[0, :, 1]),
    np.mean(err_ircs[0, :, 1]),
    np.mean(err_irir[0, :, 1]),
    np.mean(err_ircom[0, :, 1]),
    np.mean(err_irir[0, :, 1]),
)
all_errr[2, :, 1] = (
    np.std(err_irco[0, :, 1]),
    np.std(err_ircs[0, :, 1]),
    np.std(err_irir[0, :, 1]),
    np.std(err_ircom[0, :, 1]),
    np.std(err_irir[0, :, 1]),
)


all_errr[3, :, 0] = (
    np.mean(all_errr[:-1, 0, 0]),
    np.mean(all_errr[:-1, 1, 0]),
    np.mean(all_errr[:-1, 2, 0]),
    np.mean(all_err[:-1, 3, 0]),
    np.mean(all_errr[:-1, 4, 0]),
)
all_errr[3, :, 1] = (
    np.mean(all_errr[:-1, 0, 1]),
    np.mean(all_acc[:-1, 1, 1]),
    np.mean(all_errr[:-1, 2, 1]),
    np.mean(all_errr[:-1, 3, 1]),
    np.mean(all_errr[:-1, 4, 1]),
)

na_acc = np.zeros((2, 5, 2))
na_acc[0, :, 0] = (
    np.mean(acc_naco[:, 0]),
    np.mean(acc_nacs[:, 0]),
    np.mean(acc_nair[:, 0]),
    np.mean(acc_nacom[:, 0]),
    np.mean(acc_naco[:, 0]),
)
na_acc[0, :, 1] = (
    np.std(acc_naco[:, 0]),
    np.std(acc_nacs[:, 0]),
    np.std(acc_nair[:, 0]),
    np.std(acc_nacom[:, 0]),
    np.std(acc_naco[:, 0]),
)
na_acc[1, :, 0] = (
    np.mean(acc_naco[:, 1]),
    np.mean(acc_nacs[:, 1]),
    np.mean(acc_nair[:, 1]),
    np.mean(acc_nacom[:, 1]),
    np.mean(acc_naco[:, 1]),
)
na_acc[1, :, 1] = (
    np.std(acc_naco[:, 1]),
    np.std(acc_nacs[:, 1]),
    np.std(acc_nair[:, 1]),
    np.std(acc_nacom[:, 1]),
    np.std(acc_naco[:, 1]),
)

na_err = np.zeros((2, 5, 2))
na_err[0, :, 0] = (
    np.mean(err_naco[:, 0]),
    np.mean(err_nacs[:, 0]),
    np.mean(err_nair[:, 0]),
    np.mean(err_nacom[:, 0]),
    np.mean(err_naco[:, 0]),
)
na_err[0, :, 1] = (
    np.std(err_naco[:, 0]),
    np.std(err_nacs[:, 0]),
    np.std(err_nair[:, 0]),
    np.std(err_nacom[:, 0]),
    np.std(err_naco[:, 0]),
)
na_err[1, :, 0] = (
    np.mean(err_naco[:, 1]),
    np.mean(err_nacs[:, 1]),
    np.mean(err_nair[:, 1]),
    np.mean(err_nacom[:, 1]),
    np.mean(err_naco[:, 1]),
)
na_err[1, :, 1] = (
    np.std(err_naco[:, 1]),
    np.std(err_nacs[:, 1]),
    np.std(err_nair[:, 1]),
    np.std(err_nacom[:, 1]),
    np.std(err_naco[:, 1]),
)


# Save Data
np.save("all_acc_knn", all_acc)
np.save("all_acc_ref", all_accr)
np.save("all_err_knn", all_err)
np.save("all_err_ref", all_errr)
np.save("Na_acc", na_acc)
np.save("Na_err", na_err)


# Visualize
print("Generating Figures...")
clrs = [(178, 97, 219), (2, 68, 142), (0, 152, 218), (177, 203, 71)]
clrs_r = [(177, 203, 71), (0, 152, 218), (2, 68, 142), (178, 97, 219)]
my_cm = kf.make_cmap(clrs, bit=True)
my_cm_r = kf.make_cmap(clrs_r, bit=True)

kf.plot_matrix(all_acc, my_cm, "All_ACC_matrix.pdf")
kf.plot_matrix(all_accr, my_cm, "All_ACC_reftable_matrix.pdf")
kf.plot_matrix(all_err, my_cm_r, "All_ERR_matrix.pdf")
kf.plot_matrix(all_errr, my_cm_r, "All_ERR_reftable_matrix.pdf")
kf.plot_matrix_na(na_acc, my_cm, "Na_ACC_matrix.pdf")
kf.plot_matrix_na(na_err, my_cm_r, "Na_ACC_matrix.pdf")

print("*********************************************")
print("Fin")
print("*********************************************")
