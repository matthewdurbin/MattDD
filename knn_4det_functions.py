# -*- coding: utf-8 -*-
"""
Various support functions for NSS work
"""
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def norm(inp):
    """Takes input and normalizes it to unity"""
    out = np.zeros((np.shape(inp)))
    sums = np.sum(inp, axis=1)[:, None]
    for i in range(len(inp)):
        if sums[i] > 0:
            out[i] = inp[i] / sums[i]
    return out


def least_squares(test, ref):
    """
    Lest Squares Reference Table Alogrithm
    test - data to perform algoirthm to
    ref - reference table 
    """
    res = np.zeros((len(test), len(ref)))
    ls_guess = np.zeros(len(test))
    for i in range(len(test)):
        for j in range(len(ref)):
            res[i, j] = (
                (test[i, 0] - ref[j, 0]) ** 2
                + (test[i, 1] - ref[j, 1]) ** 2
                + (test[i, 2] - ref[j, 2]) ** 2
                + (test[i, 3] - ref[j, 3]) ** 2
            )
        ls_guess[i] = np.argmin(res[i])
    return (ls_guess, res)


def error_analysis(predictions, truth):
    """ Calculates the errors, accounting for circular statistis
    predictions
    truth
    """
    score = accuracy_score(truth, predictions)
    error = 180 - abs(abs(truth - predictions) - 180)
    avg_error = np.mean(error)
    std_error = np.std(error)
    return (score, error, avg_error, std_error)


def distance_analysis(bins, r, error):
    """ Calculates error wrt radius
    bins - radial buckets to perform analysis 
    r - radius of trial
    error - error of trial
    """
    nbins = len(bins) - 1
    bin_counter = np.zeros(nbins)
    zero_counter = np.zeros(nbins)
    total_error = np.zeros(nbins)
    bin_error = np.zeros(nbins)
    bin_error_std = np.zeros(nbins)
    binwise = 0
    for i in range(nbins):
        for j in range(len(r)):
            if bins[i] <= r[j] < bins[i + 1]:
                bin_counter[i] += 1
                total_error[i] += error[j]
                binwise = np.append(binwise, error[j])
                if error[j] == 0:
                    zero_counter[i] += 1
        binwise = np.delete(binwise, 0)
        bin_error[i] = binwise.mean()
        binwise = 0
    rad_score = zero_counter / bin_counter
    return (rad_score, bin_error)


def analysis(y, y_p):
    """ Get score, with circular statists """
    score = accuracy_score(y, y_p)
    error = np.mean(180 - abs(abs(y - y_p) - 180))
    return (score, error)


def analysis_r(y, y_p, r, bins):
    """ Analyze results wrt radius
    y - trial labels
    y - trial predictions
    r - radius
    bins - radial buckets to perform analysis
    """
    nbins = len(bins) - 1
    error = 180 - abs(abs(y - y_p) - 180)
    bin_counter = np.zeros(nbins)
    zero_counter = np.zeros(nbins)
    total_error = np.zeros(nbins)
    bin_error = np.zeros(nbins)
    bin_error_std = np.zeros(nbins)
    binwise = 0
    for i in range(nbins):
        for j in range(len(r)):
            if bins[i] <= r[j] < bins[i + 1]:
                bin_counter[i] += 1
                total_error[i] += error[j]
                binwise = np.append(binwise, error[j])
                if error[j] == 0:
                    zero_counter[i] += 1
        binwise = np.delete(binwise, 0)
        bin_error[i] = binwise.mean()
        bin_error_std[i] = binwise.std()
        binwise = 0
    rad_score = zero_counter / bin_counter
    return (rad_score, bin_error, bin_error_std)


def ttKNN(test, train, ref, y, r):
    """
    KNN - specify datsets to train/test with
    **** labels must be identicle for both datasets! ***
    test - test data
    train - train dadta
    ref - reference table
    y - labels 
    r - radius
    """
    nfold_a = 5
    knn_a = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    skf_a = StratifiedKFold(n_splits=nfold_a, shuffle=True)
    acc_a = np.zeros((nfold_a, 2))
    err_a = np.zeros((2, nfold_a, 2))
    for fno, (train_index, test_index) in enumerate(skf_a.split(train, y)):
        print("folding", fno + 1, "/", nfold_a)
        xa_tr, ya_tr, ra_tr = train[train_index], y[train_index], r[train_index]
        xa_ts, ya_ts, ra_ts = test[test_index], y[test_index], r[test_index]
        knn_a.fit(xa_tr, ya_tr)
        pred_a = knn_a.predict(xa_ts)
        acc, b, err, s_err = error_analysis(ya_ts, pred_a)
        acc_a[fno, 0] = acc
        err_a[:, fno, 0] = [err, s_err]
        rt_pred, b = least_squares(xa_ts, ref)
        rt_acc, bb, rt_err, rt_s_err = error_analysis(ya_ts, rt_pred)
        acc_a[fno, 1] = rt_acc
        err_a[:, fno, 1] = [rt_err, rt_s_err]
        print("KNN/RT Accuracy: ", acc, "/", rt_acc)
        print("KN/RT Avg Ang Error: ", err, "+/-", s_err, "/", rt_err, "+/-", rt_s_err)
    return (acc_a, err_a)


def make_cmap(colors, position=None, bit=False):
    """
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    """
    import matplotlib as mpl
    import numpy as np

    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (
                bit_rgb[colors[i][0]],
                bit_rgb[colors[i][1]],
                bit_rgb[colors[i][2]],
            )
    cdict = {"red": [], "green": [], "blue": []}
    for pos, color in zip(position, colors):
        cdict["red"].append((pos, color[0], color[0]))
        cdict["green"].append((pos, color[1], color[1]))
        cdict["blue"].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap("my_colormap", cdict, 256)
    return cmap


def plot_matrix(data, color_map, filename):
    """ 
    NSS Figure/Result Matrix: 
    Columns: Training dataset/approach
    Rows: Testing dataset
    data - results to plot
    color_map - color map
    file name- string to name file (put in quotes)
        
    """
    fs = 15
    train = ["$^{60}Co$", "$^{137}Cs$", "$^{192}Ir$", "Combined", "Multi-Step"]
    test = ["$^{60}Co$", "$^{137}Cs$", "$^{192}Ir$", "Overall"]
    fig, ax = plt.subplots()
    ax.matshow(data[:, :, 0], cmap=color_map)
    ax.set_xticklabels([""] + train, fontsize=fs)
    ax.set_yticklabels([""] + test, fontsize=fs)
    cax = ax.matshow(data[:, :, 0], cmap=color_map, interpolation="nearest")
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(data[:, :, 0]):
        ax.text(j, i, "{:1.3f}".format(z), ha="center", va="bottom", fontsize=fs)

    for (i, j), z in np.ndenumerate(data[:, :, 1]):
        ax.text(j, i, "+/-" + "{:1.3f}".format(z), ha="center", va="top", fontsize=fs)

    fig.set_size_inches(10, 7)
    plt.savefig(filename)


def plot_matrix_na(data, color_map, filename):
    """ 
    NSS Figure/Result Matrix: Na only data
    Columns: Training dataset/approach
    Rows: Testing dataset
    data - results to plot
    color_map - color map
    file name- string to name file (put in quotes)
        
    """
    fs = 15
    train = ["$^{60}Co$", "$^{137}Cs$", "$^{192}Ir$", "Combined", "Multi-Step"]
    test = ["KNN", "LSRT"]
    fig, ax = plt.subplots()
    ax.matshow(data[:, :, 0], cmap=color_map)
    ax.set_xticklabels([""] + train, fontsize=fs)
    ax.set_yticklabels([""] + test, fontsize=fs)
    cax = ax.matshow(data[:, :, 0], cmap=color_map, interpolation="nearest")
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(data[:, :, 0]):
        ax.text(j, i, "{:1.3f}".format(z), ha="center", va="bottom", fontsize=fs)

    for (i, j), z in np.ndenumerate(data[:, :, 1]):
        ax.text(j, i, "+/-" + "{:1.3f}".format(z), ha="center", va="top", fontsize=fs)

    fig.set_size_inches(10, 7)
    plt.savefig(filename)
