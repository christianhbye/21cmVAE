import h5py
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

import preprocess as pp
from build_models import build_direct_emulator
from training_tools import train_emulator


MAX_TRIALS = 500 # number of models to build
EPOCHS = 350 # max epochs for training (we use early stopping as well)

with h5py.File('dataset_21cmVAE.h5', 'r') as hf:
    signal_train = hf['signal_train'][:]
    signal_val = hf['signal_val'][:]
    signal_test = hf['signal_test'][:]
    par_train = hf['par_train'][:]
    par_val = hf['par_val'][:]
    par_test = hf['par_test'][:]

par_labels = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp']

signal_train_preproc = pp.preproc(signal_train, signal_train)
signal_val_preproc = pp.preproc(signal_val, signal_train)
signal_test_preproc = pp.preproc(signal_test, signal_train)

# (fixed) hyperparameters that define the search range
min_hidden_layers = 1 # min. no. of hidden layers
max_step_h_layers = 4 # max. no. of hidden layers
h_layer_step = 1
min_hidden_dim = 32 # min. dimensionality of hidden layer
max_step_hidden_dim = 6 # max. dimensionality of hidden layer
hidden_dim_step = 64 # step size

# fixed hyperparameters
activation_func = 'relu'
em_lr = 0.01

# Reduce LR callback (https://keras.io/api/callbacks/reduce_lr_on_plateau/)
em_lr_factor = 0.95
em_lr_patience = 5
em_min_lr = 1e-4
lr_min_delta = 5e-9

# for early stopping (https://keras.io/api/callbacks/early_stopping/)
early_stop_patience = 15
es_min_delta = 1e-10


def generate_hp(min_val, step_size, max_step):
    """
    Function that generates one hyperparameter given the search range
    :param min_val: float, min value of the hp
    :param step_size: float or int, resolution of grid
    :param max_step: int, max multiple of step_size that can be added to min
    :return: float, the hyperparameter
    """
    step = np.random.randint(0, max_step+1)
    return min_val + step * step_size


def generate_layer_hps(no_hidden_layers=(min_hidden_layers, h_layer_step, max_step_h_layers),
                       hidden_dims=(min_hidden_dim, hidden_dim_step, max_step_hidden_dim)):
    """
    Get the hyperparameters that control the architecture, i.e. the number of layers and their dimensions. Input must be tuples of the form
    (min, step size, max number of steps).
    :param no_hidden_layers: Number of emulator layers
    :param hidden_dims: Dimensionality of the layers
    :return: tuple of lists, giving the dimensionality of each layer for the encoder, decoder, and emulator
    """
    #generate hyperparams for layers
    number_of_layers = generate_hp(*no_hidden_layers)
    # initialize empty array
    hidden_dims = np.empty(number_of_layers)
    for i in range(number_of_layers):
        dim = generate_hp(*hidden_dims) # dimensionality of each layer
        hidden_dims[i] = dim
    return hidden_dims

# Input variables
X_train = pp.par_transform(par_train, par_train)
X_val = pp.par_transform(par_val, par_train)

# Output variables
y_train = pp.preproc(signal_train, signal_train)
y_val = pp.preproc(signal_val, signal_train)


# save search range:
with open('search_range.txt', 'w') as f:
    f.write('\nmin_hidden_layers = {}'.format(min_hidden_layers))
    f.write('\nmax_step_h_layers = {}'.format(max_step_h_layers))
    f.write('\nh_layer_step = {}'.format(h_layer_step))
    f.write('\nmin_hidden_dim = {}'.format(min_hidden_dim))
    f.write('\nmax_step_hidden_dim = {}'.format(max_step_hidden_dim))
    f.write('\nhidden_dim_step = {}'.format(hidden_dim_step))
    f.write('\nactivation_func = {}'.format(activation_func))
    f.write('\nLearning rate = {}'.format(em_lr))
    f.write('\nlr_factor = {}'.format(em_lr_factor))
    f.write('\nlr_patience = {}'.format(em_lr_patience))
    f.write('\nLearning rate min delta = {}'.format(lr_min_delta))
    f.write('\nmin_lr = {}'.format(em_min_lr))
    f.write('\nearly_stop_patience = {}'.format(early_stop_patience+))
    f.write('\nEarly stop min delta = {}'.format(es_min_delta))


def save_results(trial, layer_hps, emulator, losses):
    """
    Function that saves the results of the trial to the file "results_trial_<trial_number>.txt"
    :param trial: int, the trial number
    :param layer_hps: list of lists of dimensions of layers in the emulator
    :param emulator: keras model instance of emulator
    :param losses: list of floats giving the training and validation loss of the vae and emulator for the epoch
    that minimizes the emulator validation loss
    :return: None
    """
    fname = 'results_trial_' + str(trial)
    with open(fname+'.txt', 'w') as f:
        f.write('\n\nEmulator Losses:')
        f.write('\nTrain = {:.4f}'.format(losses[0]))
        f.write('\nValidation = {:.4f}'.format(losses[1]))
        f.write('\n')
        emulator.summary(print_fn=lambda x: f.write(x + '\n'))
    # save hps
    np.save(fname+'_layer_hps.npy', layer_hps)

def delete_results(trial):
    """
    Function that delete results of trial. This gets called when the trial is no longer among the five best.
    :param trial: int, the trial number
    :return: None
    """
    fname = 'results_trial_' + str(int(trial))
    os.remove(fname+'.txt')
    os.remove(fname+'_layer_hps.npy')

def run_tuner(max_trials=MAX_TRIALS):
    """
    The main function. Calls the other functions to do the hyperparameter search and saves the intersting parameters
    and results.
    :param max_trials: int, max number of trials used in the run.
    :return: None
    """
    five_best_val = np.empty((5,2))  # array of the best validation losses
    for i in range(max_trials):  # for each trial, we train the emulator and VAE
        layer_hps = generate_layer_hps()  # generate architecture
        # build the NN
        emulator = build_direct_emulator(layer_hps, signal_train, par_train)
        # train and get the losses
        losses = train_emulator(emulator, signal_train, signal_val, par_train, par_val, EPOCHS, em_lr, em_lr_factor, em_lr_patience,
                                em_lr_min_delta, em_min_lr, es_min_delta, es_patience)
        # we care about the epoch that gives the minimum validation loss for the emulator ...
        em_loss_val = losses[-1]
        min_indx = np.argmin(em_loss_val)  # ... this is that epoch
        min_losses = [loss[min_indx] for loss in losses]  # make a list of the values of the losses for that epoch
        em_loss_val_min = min_losses[-1]
        if i < 5:  # if this is among the first five trials, we save the results anyway
            five_best_val[i, 0] = i
            five_best_val[i, 1] = em_loss_val_min
            save_results(i, layer_hps, emulator, min_losses)
        elif em_loss_val_min < np.max(five_best_val[:, 1]):  # for subsequent trials, we save the results if they're better than previous
            idx = np.argmax(five_best_val[:, 1])
            worst_trial = five_best_val[idx, 0]
            delete_results(worst_trial)
            five_best_val[idx, 0] = i
            five_best_val[idx, 1] = em_loss_val_min
            save_results(i, layer_hps, emulator, min_losses)
        # hopefully limit memory usage
        del emulator
        K.clear_session()
    return five_best_val

five_best_trials = run_tuner()

# the very best trial:
best_trial_idx = np.argmin(five_best_trials[:, 1])
best_trial = int(five_best_trials[best_trial_idx, 0])
# load the hyperparameters of that trial
best_layer_hps = []
best_layer_hps_list = np.load('results_trial_'+str(best_trial)+'_layer_hps.npy', allow_pickle=True)
for i in range(len(best_layer_hps_list)):
    best_layer_hps.append(best_layer_hps_list[i])

# now, retrain the emulator and VAE with the best hyperparameters
tuned_em = build_direct_emulator(best_layer_hps, signal_train, par_train)
em_loss, em_loss_val = train_emulator(tuned_emulator, signal_train, signal_val, par_train, par_val, EPOCHS, em_lr, em_lr_factor,
                                      em_lr_patience, em_lr_min_delta, em_min_lr, es_min_delta, es_patience)

# Can plot the  training curves like this
# import matplotlib.pyplot as plt
# if not 'plots' in os.listdir():
#   os.makedirs('plots')
#
# x_vals = np.arange(len(em_loss)) + 1
# plt.plot(x_vals, em_loss, label='Train')
# plt.plot(x_vals, em_loss_val, label='Validation')
# plt.axvline(np.argmin(em_loss_val)+1, ls='--', c='black')
# plt.ylabel('Loss')
# plt.title('Emulator')
# plt.xlabel('Epoch')
# plt.legend()
# plt.savefig('plots/tuned_em_training_curve.png')
# plt.show()

