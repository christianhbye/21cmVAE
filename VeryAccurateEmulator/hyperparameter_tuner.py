import h5py
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

import VeryAccurateEmulator.preprocess as pp
from VeryAccurateEmulator.build_models import build_direct_emulator
from VeryAccurateEmulator.training_tools import train_emulator


class HyperParameterTuner:
    def __init__(self, max_trials=500, epochs=350, min_hidden_layers=1, max_step_h_layers=4, h_layer_step=1,
                 min_hidden_dim = 32, max_step_hidden_dim = 6, hidden_dim_step=64):
        self.max_trials = max_trials  # number of models to build
        self.epochs = epochs  # max epochs for training (we use early stopping as well)

        script_path = os.path.realpath(__file__)[:-len('VeryAccurateEmulator.py')]
        with h5py.File(script_path + 'dataset_21cmVAE.h5', 'r') as hf:
            self.signal_train = hf['signal_train'][:]
            self.signal_val = hf['signal_val'][:]
            self.signal_test = hf['signal_test'][:]
            self.par_train = hf['par_train'][:]
            self.par_val = hf['par_val'][:]
            self.par_test = hf['par_test'][:]
        self.par_labels = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp']

        # (fixed) hyperparameters that define the search range
        self.min_hidden_layers = min_hidden_layers  # min. no. of hidden layers
        self.max_step_h_layers = max_step_h_layers  # max. no. of hidden layers
        self.h_layer_step = h_layer_step
        self.min_hidden_dim = min_hidden_dim  # min. dimensionality of hidden layer
        self.max_step_hidden_dim = max_step_hidden_dim  # max. dimensionality of hidden layer
        self.hidden_dim_step = hidden_dim_step  # step size

        assert type(self.min_hidden_layers) == int
        assert type(self.max_step_h_layers) == int
        assert type(self.h_layer_step) == int
        assert type(self.min_hidden_dim) == int
        assert type(self.max_step_hidden_dim) == int
        assert type(self.hidden_dim_step) == int

        # fixed hyperparameters
        self.activation_func = 'relu'
        self.em_lr = 0.01
        assert type(self.em_lr) == float or int

        # Reduce LR callback (https://keras.io/api/callbacks/reduce_lr_on_plateau/)
        self.em_lr_factor = 0.95
        self.em_lr_patience = 5
        self.em_min_lr = 1e-4
        self.em_lr_min_delta = 5e-9
        assert type(self.em_lr_factor) == float or int
        assert type(self.em_lr_patience) == int
        assert type(self.em_min_lr) == float or int
        assert type(self.em_lr_min_delta) == flaot or int
        assert self.em_min_lr <= self.em_lr, "Min LR must be <= initial LR"

signal_train_preproc = pp.preproc(signal_train, signal_train)
signal_val_preproc = pp.preproc(signal_val, signal_train)
signal_test_preproc = pp.preproc(signal_test, signal_train)





# for early stopping (https://keras.io/api/callbacks/early_stopping/)
es_patience = 15
es_min_delta = 1e-10
assert type(es_patience) == int
assert type(es_min_delta) == float or int


def generate_hp(min_val: int, step_size: int, max_step: int) -> int:
    """
    Function that generates one hyperparameter given the search range
    :param min_val: int, min value of the hp
    :param step_size: int, resolution of grid
    :param max_step: int, max multiple of step_size that can be added to min
    :return: int, the hyperparameter
    """
    step = np.random.randint(0, max_step+1)
    return min_val + step * step_size


def generate_layer_hps(no_hidden_layers=(min_hidden_layers, h_layer_step, max_step_h_layers),
                       hidden_dims=(min_hidden_dim, hidden_dim_step, max_step_hidden_dim)):
    """
    Get the hyperparameters that control the architecture, i.e. the number of layers and their dimensions.
    Input must be tuples of the form (min, step size, max number of steps).
    :param no_hidden_layers: Number of emulator layers
    :param hidden_dims: Dimensionality of the layers
    :return: tuple of lists, giving the dimensionality of each layer for the encoder, decoder, and emulator
    """
    assert len(no_hidden_layers) == 3
    assert len(hidden_dims) == 3
    assert all(isinstance(h, int) for h in no_hidden_layers)
    assert all(isinstance(h, int) for h in hidden_dims)
    assert min_hidden_dim > 0
    #generate hyperparams for layers
    number_of_layers = generate_hp(*no_hidden_layers)
    # initialize empty array
    hidden_dim_arr = np.empty(number_of_layers)
    for i in range(number_of_layers):
        dim = generate_hp(*hidden_dims)  # dimensionality of each layer
        hidden_dim_arr[i] = dim
    return hidden_dim_arr

# Input variables
X_train = pp.par_transform(par_train, par_train)
X_val = pp.par_transform(par_val, par_train)

# Output variables
y_train = pp.preproc(signal_train, signal_train)
y_val = pp.preproc(signal_val, signal_train)

TIME = time.time() # to be saved in files to make them identifiable between different runs
# save search range:
with open(sr_fname, 'a') as f:
    f.write('\n ---------- \n HYPERPARAMETER RUN, time = {:.0f}'.format(TIME))
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
    f.write('\nLearning rate min delta = {}'.format(em_lr_min_delta))
    f.write('\nmin_lr = {}'.format(em_min_lr))
    f.write('\nearly_stop_patience = {}'.format(es_patience))
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
    with open(fname+'.txt', 'a') as f:
        f.write('\n ---------- \n HYPERPARAMETER RUN, time = {:.0f}'.format(TIME))
        f.write('\n\nEmulator Losses:')
        f.write('\nTrain = {:.4f}'.format(losses[0]))
        f.write('\nValidation = {:.4f}'.format(losses[1]))
        f.write('\n')
        emulator.summary(print_fn=lambda x: f.write(x + '\n'))
    # save hps
    np.save(fname+'_{:.0f}_layer_hps.npy'.format(TIME), layer_hps)


def delete_results(trial):
    """
    Function that delete results of trial. This gets called when the trial is no longer among the five best.
    :param trial: int, the trial number
    :return: None
    """
    fname = 'results_trial_' + str(int(trial))
    os.remove(fname+'.txt')
    os.remove(fname+'_{:.0f}_layer_hps.npy'.format(TIME))


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
        losses = train_emulator(emulator, signal_train, signal_val, par_train, par_val, EPOCHS, em_lr, em_lr_factor,
                                em_lr_patience, em_lr_min_delta, em_min_lr, es_min_delta, es_patience)
        # we care about the epoch that gives the minimum validation loss for the emulator ...
        validation_loss = losses[-1]
        min_indx = np.argmin(validation_loss)  # ... this is that epoch
        min_losses = [loss[min_indx] for loss in losses]  # make a list of the values of the losses for that epoch
        em_loss_val_min = min_losses[-1]
        if i < 5:  # if this is among the first five trials, we save the results anyway
            five_best_val[i, 0] = i
            five_best_val[i, 1] = em_loss_val_min
            save_results(i, layer_hps, emulator, min_losses)
        # for subsequent trials, we save the results if they're better than previous
        elif em_loss_val_min < np.max(five_best_val[:, 1]):
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
best_layer_hps = np.load('results_trial_'+str(best_trial)+'_{:.0f}_layer_hps.npy'.format(TIME), allow_pickle=True)

# now, retrain the emulator and VAE with the best hyperparameters
tuned_em = build_direct_emulator(best_layer_hps, signal_train, par_train)
em_loss, em_loss_val = train_emulator(tuned_em, signal_train, signal_val, par_train, par_val, EPOCHS, em_lr,
                                      em_lr_factor, em_lr_patience, em_lr_min_delta, em_min_lr, es_min_delta,
                                      es_patience)

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

