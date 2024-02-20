import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import scipy as sp

# raw data directory path - it has a specific structure basen on my convenience
RAW_DATA_DIR = r'C:\Users\Idanl\Desktop\AccuLine noisy ECG classification assignment\DATA\Raw'
ORD_DATA_DIR = r'C:\Users\Idanl\Desktop\AccuLine noisy ECG classification assignment\DATA\Ordered'

c17_dir = os.path.join(RAW_DATA_DIR, 'CINC 2017')

### Load files from CINC 2017 directory - only annotated data was acquired

## Initiate the X and y data aggregators

X_train_17 = np.empty((0, 3000))
y_train_17 = np.empty((0, 0))

X_val_17 = np.empty((0, 3000))
y_val_17 = np.empty((0, 0))

## Load Training - takes a very long time
training_dir = os.path.join(c17_dir, 'training2017')

df = pd.read_csv(training_dir + '\\REFERENCE.csv', header=None, index_col=0).T

# Load references
train_labels = (df != '~').astype(int)

# Load data
data_files = [f[:-4] for f in os.listdir(training_dir) if f.endswith('mat')]

for file_name in tqdm(data_files):
    # load subject data an label as arrays
    subject_data = sp.io.loadmat(training_dir + '\\' + file_name + '.mat')['val']

    # We are dealing with 10 sec in 300 Hz - 3000 samples segments
    # We only take full 10 sec signals with no overlap - This should be reviewed in real life scenario
    subject_data = subject_data[:, :(subject_data.shape[1] // 3000) * 3000]
    subject_data = subject_data.reshape(-1, 3000)

    n_samples = subject_data.shape[0]
    subject_label = train_labels[file_name].values.repeat(n_samples)

    # data_dict[file_name] = (subject_data, subject_data.shape[1], subject_label)

    # concat to existing data
    X_train_17 = np.append(X_train_17, subject_data, axis=0)
    y_train_17 = np.append(y_train_17, subject_label)

if 'X_train_2017.npy' not in os.listdir(ORD_DATA_DIR) and 'y_train_2017.npy' not in os.listdir(ORD_DATA_DIR):
    np.save(os.path.join(ORD_DATA_DIR, 'X_train_2017.npy'), X_train_17)
    np.save(os.path.join(ORD_DATA_DIR, 'y_train_2017.npy'), y_train_17)

#########################################################################################
## Load Validation

validation_dir = os.path.join(c17_dir, 'validation')

df = pd.read_csv(validation_dir + '\\REFERENCE.csv', header=None, index_col=0).T

val_labels = (df != '~').astype(int)

# Load data
data_files = [f[:-4] for f in os.listdir(validation_dir) if f.endswith('mat')]

for file_name in data_files:
    # load subject data an label as arrays
    subject_data = sp.io.loadmat(validation_dir + '\\' + file_name + '.mat')['val']

    subject_data = subject_data[:, :(subject_data.shape[1] // 3000) * 3000]
    subject_data = subject_data.reshape(-1, 3000)

    n_samples = subject_data.shape[0]
    subject_label = train_labels[file_name].values.repeat(n_samples)

    # concat to existing data
    X_val_17 = np.append(X_val_17, subject_data, axis=0)
    y_val_17 = np.append(y_val_17, subject_label)

if 'X_val_2017.npy' not in os.listdir(ORD_DATA_DIR) and 'y_val_2017.npy' not in os.listdir(ORD_DATA_DIR):
    np.save(os.path.join(ORD_DATA_DIR, 'X_val_2017.npy'), X_val_17)
    np.save(os.path.join(ORD_DATA_DIR, 'y_val_2017.npy'), y_val_17)

#########################################################################################
