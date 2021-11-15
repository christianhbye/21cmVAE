import h5py
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import VeryAccurateEmulator.preprocess as pp
from VeryAccurateEmulator.build_models import build_direct_emulator
from VeryAccurateEmulator.training_tools import train_emulator


def generate_hp(min_val: int, step_size: int, max_step: int) -> int:
    """
    Function that generates one hyperparameter given the search range
    :param min_val: int, min value of the hp
    :param step_size: int, resolution of grid
    :param max_step: int, max multiple of step_size that can be added to min
    :return: int, the hyperparameter
    """
    step = np.random.randint(0, max_step + 1)
    return min_val + step * step_size


def generate_layer_hps(no_hidden_layers, hidden_dims):
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

def save_results(trial, layer_hps, emulator, losses, time):
    """
    Function that saves the results of the trial to the file "results_trial_<trial_number>.txt"
    :param trial: int, the trial number
    :param layer_hps: list of lists of dimensions of layers in the emulator
    :param emulator: keras model instance of emulator
    :param losses: list of floats giving the training and validation loss of the vae and emulator for the epoch
    that minimizes the emulator validation loss
    :param time: float, unix time identifying the hp run
    :return: None
    """
    fname = 'results_trial_' + str(trial)
    with open(fname+'.txt', 'a') as f:
        f.write('\n ---------- \n HYPERPARAMETER RUN, time = {:.0f}'.format(time))
        f.write('\n\nEmulator Losses:')
        f.write('\nTrain = {:.4f}'.format(losses[0]))
        f.write('\nValidation = {:.4f}'.format(losses[1]))
        f.write('\n')
        emulator.summary(print_fn=lambda x: f.write(x + '\n'))
    # save hps
    np.save(fname+'_{:.0f}_layer_hps.npy'.format(TIME), layer_hps)


def delete_results(trial, time):
    """
    Function that delete results of trial. This gets called when the trial is no longer among the five best.
    :param trial: int, the trial number
    :param time: float, unix time identifying the hp run
    :return: None
    """
    fname = 'results_trial_' + str(int(trial))
    os.remove(fname+'.txt')
    os.remove(fname+'_{:.0f}_layer_hps.npy'.format(time))

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

        self.signal_train_preproc = pp.preproc(self.signal_train, self.signal_train)
        self.signal_val_preproc = pp.preproc(self.signal_val, self.signal_train)
        self.signal_test_preproc = pp.preproc(self.signal_test, self.signal_train)

        # Input variables
        self.X_train = pp.par_transform(self.par_train, self.par_train)
        self.X_val = pp.par_transform(self.par_val, self.par_train)
        # Output variables
        self.y_train = pp.preproc(self.signal_train, self.signal_train)
        self.y_val = pp.preproc(self.signal_val, self.signal_train)

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

        # for early stopping (https://keras.io/api/callbacks/early_stopping/)
        self.es_patience = 15
        self.es_min_delta = 1e-10
        assert type(self.es_patience) == int
        assert type(self.es_min_delta) == float or int

        self.time = time.time() # to be saved in files to make them identifiable between different runs

    # save search range
    def save_sr(self):
        with open('search_range.txt', 'a') as f:
            f.write('\n ---------- \n HYPERPARAMETER RUN, time = {:.0f}'.format(self.time))
            f.write('\nmin_hidden_layers = {}'.format(self.min_hidden_layers))
            f.write('\nmax_step_h_layers = {}'.format(self.max_step_h_layers))
            f.write('\nh_layer_step = {}'.format(self.h_layer_step))
            f.write('\nmin_hidden_dim = {}'.format(self.min_hidden_dim))
            f.write('\nmax_step_hidden_dim = {}'.format(self.max_step_hidden_dim))
            f.write('\nhidden_dim_step = {}'.format(self.hidden_dim_step))
            f.write('\nactivation_func = {}'.format(self.activation_func))
            f.write('\nLearning rate = {}'.format(self.em_lr))
            f.write('\nlr_factor = {}'.format(self.em_lr_factor))
            f.write('\nlr_patience = {}'.format(self.em_lr_patience))
            f.write('\nLearning rate min delta = {}'.format(self.em_lr_min_delta))
            f.write('\nmin_lr = {}'.format(self.em_min_lr))
            f.write('\nearly_stop_patience = {}'.format(self.es_patience))
            f.write('\nEarly stop min delta = {}'.format(self.es_min_delta))


    def run_tuner(self):
        """
        The main function. Calls the other functions to do the hyperparameter search and saves the intersting parameters
        and results.
        :return: None
        """
        five_best_val = np.empty((5,2))  # array of the best validation losses
        for i in range(self.max_trials):  # for each trial, we train the emulator and VAE
            no_hidden_layers = (self.min_hidden_layers, self.h_layer_step, self.max_step_h_layers)
            hidden_dims = (self.min_hidden_dim, self.hidden_dim_step, self.max_step_hidden_dim)
            layer_hps = generate_layer_hps(no_hidden_layers, hidden_dims)  # generate architecture
            # build the NN
            emulator = build_direct_emulator(layer_hps, self.signal_train, self.par_train)
            # train and get the losses
            losses = train_emulator(emulator, self.signal_train, self.signal_val, self.par_train, self.par_val,
                                    self.epochs, self.em_lr, self.em_lr_factor, self.em_lr_patience,
                                    self.em_lr_min_delta, self.em_min_lr, self.es_min_delta, self.es_patience)
            # we care about the epoch that gives the minimum validation loss for the emulator ...
            validation_loss = losses[-1]
            min_indx = np.argmin(validation_loss)  # ... this is that epoch
            min_losses = [loss[min_indx] for loss in losses]  # make a list of the values of the losses for that epoch
            em_loss_val_min = min_losses[-1]
            if i < 5:  # if this is among the first five trials, we save the results anyway
                five_best_val[i, 0] = i
                five_best_val[i, 1] = em_loss_val_min
                save_results(i, layer_hps, emulator, min_losses, self.time)
            # for subsequent trials, we save the results if they're better than previous
            elif em_loss_val_min < np.max(five_best_val[:, 1]):
                idx = np.argmax(five_best_val[:, 1])
                worst_trial = five_best_val[idx, 0]
                delete_results(worst_trial, self.time)
                five_best_val[idx, 0] = i
                five_best_val[idx, 1] = em_loss_val_min
                save_results(i, layer_hps, emulator, min_losses, self.time)
            # hopefully limit memory usage
            del emulator
            K.clear_session()
        return five_best_val

    def get_results(self):
        five_best_trials = self.run_tuner()

        # the very best trial:
        best_trial_idx = np.argmin(five_best_trials[:, 1])
        best_trial = int(five_best_trials[best_trial_idx, 0])
        # load the hyperparameters of that trial
        best_layer_hps = np.load('results_trial_'+str(best_trial)+'_{:.0f}_layer_hps.npy'.format(self.time),
                                 allow_pickle=True)

        # now, retrain the emulator and VAE with the best hyperparameters
        tuned_em = build_direct_emulator(best_layer_hps, self.signal_train, self.par_train)
        em_loss, em_loss_val = train_emulator(tuned_em, self.signal_train, self.signal_val, self.par_train,
                                              self.par_val, self.epochs, self.em_lr, self.em_lr_factor,
                                              self.em_lr_patience, self.em_lr_min_delta, self.em_min_lr,
                                              self.es_min_delta, self.es_patience)

        return em_loss, em_loss_val
