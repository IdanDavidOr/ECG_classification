import os

import numpy as np
import pandas as pd

from scipy.signal import resample

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

ORD_DATA_DIR = r'C:\Users\Idanl\Desktop\AccuLine noisy ECG classification assignment\DATA\Ordered'
PROC_DATA_DIR = r'C:\Users\Idanl\Desktop\AccuLine noisy ECG classification assignment\DATA\Processed'


X_train_11 = np.load(ORD_DATA_DIR + '\X_train_2011.npy')
X_train_17 = np.load(ORD_DATA_DIR + '\X_train_2017.npy')
X_test = np.load(ORD_DATA_DIR + '\X_val_2017.npy')

y_train_11 = np.load(ORD_DATA_DIR + '\y_train_2011.npy')
y_train_17 = np.load(ORD_DATA_DIR + '\y_train_2017.npy')
y_test = np.load(ORD_DATA_DIR + '\y_val_2017.npy')

# Downsample signals to 1000 samples, i.e. 100 Hz

X_train_11 = resample(X_train_11, 1000, axis=1)  # undersampling from 300 Hz
X_train_17 = resample(X_train_17, 1000, axis=1)  # undersampling from 500 Hz
X_test = resample(X_test, 1000, axis=1)      # undersampling from 500 Hz

X_train = np.append(X_train_11, X_train_17, axis=0)
y_train = np.append(y_train_11, y_train_17)

del X_train_11, X_train_17, y_train_11, y_train_17

if 'X_train.npy' not in os.listdir(PROC_DATA_DIR):
    np.save(PROC_DATA_DIR + '\\X_train.npy', X_train)
    np.save(PROC_DATA_DIR + '\\y_train.npy', y_train)
    np.save(PROC_DATA_DIR + '\\X_test.npy', X_test)
    np.save(PROC_DATA_DIR + '\\y_test.npy', y_test)
