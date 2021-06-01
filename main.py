# standard libraries
import h5py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras import optimizers, callbacks
import time

# modules part of 21cmVAE
import preprocess as pp

HOME_DIR = '21cmVAE'  # replace this with wherever you want to run the code
DATA_DIR = 'data'  # subdir of HOME_DIR where data is stored
# this is the working directory, everything gets saved here (except data):
WORK_DIR = '21cm_emulator_GAN_%d'%(round(time.time()))

os.chdir(HOME_DIR)
if not DATA_DIR in os.listdir():
    os.mkdir(DATA_DIR)
os.chdir(DATA_DIR)

# load the datasets from the h5 file
with h5py.File('dataset.h5', 'r') as hf:
    # signals
    signal_train = hf['signal_train'][:]
    signal_val = hf['signal_val'][:]
    signal_test = hf['signal_test'][:]
    # parameters
    par_train = hf['par_train'][:]
    par_val = hf['par_val'][:]
    par_test = hf['par_test'][:]

# get the amplitudes of the signals
train_amplitudes = np.max(np.abs(signal_train), axis=-1)  # training set
val_amplitudes = np.max(np.abs(signal_val), axis=-1)  # validation set
test_amplitudes = np.max(np.abs(signal_test), axis=-1)  # test set


# preprocessed signals
signal_train_preproc = pp.preproc(signal_train, signal_train)  # training set
signal_val_preproc = pp.preproc(signal_val, signal_train)  # validation set
signal_test_preproc = pp.preproc(signal_test, signal_train)  # test set

