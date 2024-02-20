import os
from tqdm import tqdm

import numpy as np
import pandas as pd

# raw data directory path - it has a specific structure basen on my convenience
RAW_DATA_DIR = r'C:\Users\Idanl\Desktop\AccuLine noisy ECG classification assignment\DATA\Raw'
ORD_DATA_DIR = r'C:\Users\Idanl\Desktop\AccuLine noisy ECG classification assignment\DATA\Ordered'

c11_dir = os.path.join(RAW_DATA_DIR, 'CINC 2011')

### Load files from CINC 2011 directory - only annotated data was acquired

## Initiate the X and y data aggregators

X_11 = np.empty((0, 5000))
y_11 = np.empty((0, 0))

# create a labels dict
# getting the 1 labels
with open(os.path.join(c11_dir, 'RECORDS-acceptable')) as f:
    acc_cont = f.read().split('\n')
# initialize the label dict with acceptable recordings
label_dict = {k: 1 for k in acc_cont}

# getting the 0 labels
with open(os.path.join(c11_dir, 'RECORDS-unacceptable')) as f:
    unc_cont = f.read().split('\n')
# adding them to the dictionary
for k in unc_cont:
    label_dict[k] = 0

# get data file names (clean from format)
data_files = [f[:-4] for f in os.listdir(c11_dir) if f.endswith('txt')]

# for all the files:
for data_file in tqdm(data_files):
    # some recordings miss a label so we discard them
    if data_file not in label_dict:
        print(f'no label for recording {data_file}')
        continue

    # set labels
    subject_label = np.ones(12) * label_dict[data_file]

    # Load data - it's a csv saved with txt ending
    data = pd.read_csv(os.path.join(c11_dir, data_file) + '.txt', index_col=0, header=None)
    subject_data = data.T.to_numpy()

    # concat to existing data
    X_11 = np.append(X_11, subject_data, axis=0)
    y_11 = np.append(y_11, subject_label)

if 'X_train_2011.npy' not in os.listdir(ORD_DATA_DIR) and 'X_train_2011.npy' not in os.listdir(ORD_DATA_DIR):
    np.save(os.path.join(ORD_DATA_DIR, 'X_train_2011.npy'), X_11)
    np.save(os.path.join(ORD_DATA_DIR, 'y_train_2011.npy'), y_11)
