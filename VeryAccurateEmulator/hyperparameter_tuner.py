import h5py
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import tqdm

import VeryAccurateEmulator.preprocess as pp
from VeryAccurateEmulator.build_models import (
        build_direct_emulator,
        build_ae_emulator
    )
from VeryAccurateEmulator.training_tools import (
        train_emulator,
        train_ae_based_emulator
        )
from VeryAccurateEmulator import __path__

def generate_hp(min_val, step_size, max_step):
    """
    Function that generates one hyperparameter given the search range
    :param min_val: int, min value of the hp
    :param step_size: int, resolution of grid
    :param max_step: int, max multiple of step_size that can be added to min
    :return: int, the hyperparameter
    """
    step = np.random.randint(0, max_step + 1)
    hp = min_val + step * step_size
    return hp


def generate_layer_hps(no_hidden_layers, hidden_dims):
    """
    Get the hyperparameters that control the architecture, i.e. the number of
    layers and their dimensions.
    Input must be tuples of the form (min, step size, max number of steps).
    :param no_hidden_layers: Number of emulator layers
    :param hidden_dims: Dimensionality of the layers
    :return: tuple of lists, giving the dimensionality of each layer for the
    encoder, decoder, and emulator
    """
    assert len(no_hidden_layers) == 3
    assert len(hidden_dims) == 3
    min_hidden_dim = hidden_dims[0]
    # generate hyperparams for layers
    number_of_layers = generate_hp(*no_hidden_layers)
    # initialize empty array
    hidden_dim_list = []
    for i in range(number_of_layers):
        dim = generate_hp(*hidden_dims)  # dimensionality of each layer
        hidden_dim_list.append(dim)
    return hidden_dim_list


def save_results(trial, layer_hps, emulator, losses, time):
    """
    Function that saves the results of the trial to the file
    "results_trial_<trial_number>.txt"
    :param trial: int, the trial number
    :param layer_hps: list of lists of dimensions of layers in the emulator
    :param emulator: keras model instance of emulator
    :param losses: list of floats giving the training and validation loss of
    the vae and emulator for the epoch
    that minimizes the emulator validation loss
    :param time: float, unix time identifying the hp run
    :return: None
    """
    fname = 'results_trial_' + str(trial) + '_{:.0f}'.format(time)
    with open(fname+'.txt', 'a') as f:
        f.write('\n ---------- \n HYPERPARAMETER RUN, time = {:.0f}'
                .format(time))
        f.write('\n\nEmulator Losses:')
        f.write('\nTrain = {:.4f}'.format(losses[0]))
        f.write('\nValidation = {:.4f}'.format(losses[1]))
        f.write('\n')
        emulator.summary(print_fn=lambda x: f.write(x + '\n'))
    # save hps
    np.save(fname+'_layer_hps.npy', layer_hps)


def delete_results(trial, time):
    """
    Function that delete results of trial. This gets called when the trial is
    no longer among the five best.
    :param trial: int, the trial number
    :param time: float, unix time identifying the hp run
    :return: None
    """
    fname = 'results_trial_' + str(int(trial)) + '_{:.0f}'.format(time)
    os.remove(fname+'.txt')
    os.remove(fname+'_layer_hps.npy')


def get_best_epoch_losses(losses):
    """
    :param losses: tuple of the form (training_loss, validation_loss) for the
    emulator where each element is a list of losses (floats) for each epoch of
    the training
    :return: tuple of floats, the training and validation loss at the epoch
    where the validation loss is minimum
    """
    assert len(losses) == 2
    assert len(losses[0]) == len(losses[1])
    # we care about the epoch that gives the minimum emulator validation loss
    validation_loss = losses[-1]
    min_indx = np.argmin(validation_loss)  # ... this is that epoch
    assert validation_loss[min_indx] == np.min(validation_loss)
    # make a list of the values of the losses for that epoch
    min_losses = [loss[min_indx] for loss in losses] 
    assert len(min_losses) == 2
    assert min_losses[-1] == np.min(validation_loss)
    return min_losses


class HyperParameterTuner:
    def __init__(
            self,
            X_train,
            X_val,
            y_train,
            y_val,
            em_type="direct",  # or "ae" for ae-based
            max_trials=500,
            epochs=350,
            min_hidden_layers=1,
            h_layer_step=1,
            max_step_h_layers=4,
            min_hidden_dim=32,
            hidden_dim_step=64,
            max_step_hidden_dim=6
            ):

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.em_type = em_type
        self.max_trials = max_trials  # number of models to build
        self.epochs = epochs  # max epochs for training

        script_path = __path__[0]

        with h5py.File(script_path + '/dataset_21cmVAE.h5', 'r') as hf:
            self.signal_train = hf['signal_train'][:]
            self.signal_val = hf['signal_val'][:]
            self.signal_test = hf['signal_test'][:]
            self.par_train = hf['par_train'][:]
            self.par_val = hf['par_val'][:]
            self.par_test = hf['par_test'][:]
        self.par_labels = [
                'fstar',
                'Vc',
                'fx',
                'tau',
                'alpha',
                'nu_min',
                'Rmfp'
                ]

        self.signal_train_preproc = pp.preproc(
                self.signal_train,
                self.signal_train
                )
        self.signal_val_preproc = pp.preproc(
                self.signal_val,
                self.signal_train
                )
        self.signal_test_preproc = pp.preproc(
                self.signal_test,
                self.signal_train
                )

        # (fixed) hyperparameters that define the search range
        # max dim = min dim + max step * step size
        self.min_hidden_layers = min_hidden_layers
        self.max_step_h_layers = max_step_h_layers
        self.h_layer_step = h_layer_step
        self.min_hidden_dim = min_hidden_dim
        self.max_step_hidden_dim = max_step_hidden_dim
        self.hidden_dim_step = hidden_dim_step


        # fixed hyperparameters
        self.activation_func = 'relu'
        self.em_lr = 0.01

        # Reduce LR callback
        # (https://keras.io/api/callbacks/reduce_lr_on_plateau/)
        self.em_lr_factor = 0.95
        self.em_lr_patience = 5
        self.em_min_lr = 1e-4
        self.em_lr_min_delta = 5e-9
        assert self.em_min_lr <= self.em_lr, "Min LR must be <= initial LR"

        # for early stopping (https://keras.io/api/callbacks/early_stopping/)
        self.es_patience = 15
        self.es_min_delta = 1e-10

        # to be saved in files to make them identifiable between different runs
        self.time = time.time()  

    # save search range
    def save_sr(self):
        with open('search_range.txt', 'a') as f:
            f.write('\n ---------- \n HYPERPARAMETER RUN, time = {:.0f}'
                    .format(self.time))
            f.write('\nmin_hidden_layers = {}'.format(self.min_hidden_layers))
            f.write('\nmax_step_h_layers = {}'.format(self.max_step_h_layers))
            f.write('\nh_layer_step = {}'.format(self.h_layer_step))
            f.write('\nmin_hidden_dim = {}'.format(self.min_hidden_dim))
            f.write('\nmax_step_hidden_dim = {}'
                    .format(self.max_step_hidden_dim))
            f.write('\nhidden_dim_step = {}'.format(self.hidden_dim_step))
            f.write('\nactivation_func = {}'.format(self.activation_func))
            f.write('\nLearning rate = {}'.format(self.em_lr))
            f.write('\nlr_factor = {}'.format(self.em_lr_factor))
            f.write('\nlr_patience = {}'.format(self.em_lr_patience))
            f.write('\nLearning rate min delta = {}'
                    .format(self.em_lr_min_delta))
            f.write('\nmin_lr = {}'.format(self.em_min_lr))
            f.write('\nearly_stop_patience = {}'.format(self.es_patience))
            f.write('\nEarly stop min delta = {}'.format(self.es_min_delta))

    def run_tuner(self, progress_bar=True):
        """
        The main function. Calls the other functions to do the hyperparameter
        search and saves the intersting parameters and results.
        :return: None
        """
        five_best_val = np.empty((5,2))  # array of the best validation losses
        # for each trial, we train the emulator and VAE
        no_hidden_layers = (
            self.min_hidden_layers,
            self.h_layer_step,
            self.max_step_h_layers
        )
        hidden_dims = (
            self.min_hidden_dim,
            self.hidden_dim_step,
            self.max_step_hidden_dim
        )
        iterable = range(self.max_trials)
        if progress_bar == True:
            iterable = tqdm.trange(self.max_trial)
        elif progress_bar == "notebook":
            iterable = tqdm.notebook.trange(self.max_trial)
        else:
            iterable = range(self.max_trial)
        for i in iterable: 
            # generate architecture
            layer_hps = generate_layer_hps(no_hidden_layers, hidden_dims)
            # build the NN
            if self.em_type == "direct":
                emulator = build_direct_emulator(
                        layer_hps,
                        self.signal_train,
                        self.par_train,
                        activation_func=self.activation_func
                    )
                losses = train_emulator(
                        emulator,
                        self.signal_train,
                        self.signal_val,
                        self.par_train,
                        self.par_val,
                        self.epochs,
                        self.em_lr,
                        self.em_lr_factor,
                        self.em_lr_patience,
                        self.em_lr_min_delta,
                        self.em_min_lr,
                        self.es_min_delta,
                        self.es_patience,
                        verbose=0
                    )
            elif self.em_type == "ae":
                emulator = build_ae_emulator(
                        [self.latent_dim, layer_hps],
                        self.par_train,
                        activation_func=self.activation_func
                    )
                losses = train_ae_based_emulator(
                    emulator,
                    self.X_train,
                    self.X_val,
                    self.y_train,
                    self.y_val,
                    self.epochs,
                    self.em_lr,
                    self.em_lr_factor,
                    self.em_lr_patience,
                    self.em_lr_min_delta,
                    self.em_min_lr,
                    self.es_min_delta,
                    self.es_patience,
                    verbose=0
                )
            else:
                raise ValueError("Emulator type must be direct or ae.")
            # train and get the losses
            min_losses = get_best_epoch_losses(losses)
            em_loss_val_min = min_losses[-1]
            # save results for first five trials
            if i < 5:
                five_best_val[i, 0] = i
                five_best_val[i, 1] = em_loss_val_min
                save_results(i, layer_hps, emulator, min_losses, self.time)
            # for subsequent trials, save the results if better than previous
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

    def get_results(self, progress_bar=True):
        five_best_trials = self.run_tuner(progress_bar=progress_bar)

        # the very best trial:
        best_trial_idx = np.argmin(five_best_trials[:, 1])
        best_trial = int(five_best_trials[best_trial_idx, 0])
        assert five_best_trials[best_trial_idx, 1] \
                == np.min(five_best_trials[:, 1])
        # load the hyperparameters of that trial
        best_layer_hps = np.load(
                'results_trial_'+str(best_trial)+'_{:.0f}_layer_hps.npy'
                .format(self.time), 
                allow_pickle=True
                )
        # now, retrain the emulator with the best hyperparameters
        if self.em_type == "direct":
            tuned_em = build_direct_emulator(
                    best_layer_hps,
                    self.signal_train,
                    self.par_train,
                    activation_func=self.activation_func
                    )
            em_loss, em_loss_val = train_emulator(
                    tuned_em, self.signal_train,
                    self.signal_val,
                    self.par_train,
                    self.par_val,
                    self.epochs,
                    self.em_lr,
                    self.em_lr_factor,
                    self.em_lr_patience,
                    self.em_lr_min_delta,
                    self.em_min_lr,
                    self.es_min_delta,
                    self.es_patience,
                    verbose=1
                    )

        else:
            tuned_em = build_ae_emulator(
                    [self.latent_dim, best_layer_hps],
                    self.par_train,
                    activation_func=self.activation_func
                    )
            em_loss, em_loss_val = train_ae_based_emulator(
                    tuned_em,
                    self.X_train,
                    self.X_val,
                    self.y_train,
                    self.y_val,
                    self.epochs,
                    self.em_lr,
                    self.em_lr_factor,
                    self.em_lr_patience,
                    self.em_lr_min_delta,
                    self.em_min_lr,
                    self.es_min_delta,
                    self.es_patience,
                    verbose=1
                    )
        return tuned_em, five_best_trials, em_loss, em_loss_val

    def run_all(self, progress_bar=True):
        self.save_sr()
        tuned_emulator, five_best_trials, training_loss, validation_loss \
                = self.get_results(progress_bar=progress_bar)
        tuned_emulator.save('best_emulator_{:.0f}.h5'.format(self.time))
        np.savez(
                'tuner_results.npz',
                five_best_trials=five_best_trials,
                time=self.time,
                signal_train=self.signal_train,
                best_training_loss=training_loss,
                best_validation_loss=validation_loss
                )

class Direct_HP_Tuner(HyperParameterTuner):

    def __init__(
            self,
            max_trials=500,
            epochs=350,
            min_hidden_layers=1,
            h_layer_step=1,
            max_step_h_layers=4,
            min_hidden_dim=32,
            hidden_dim_step=64,
            max_step_hidden_dim=6
        ):
        with h5py.File(__path__[0] + '/dataset_21cmVAE.h5', 'r') as hf:
            signal_train = hf['signal_train'][:]
            signal_val = hf['signal_val'][:]
            par_train = hf['par_train'][:]
            par_val = hf['par_val'][:]

        # Input variables
        X_train = pp.par_transform(par_train, par_train)
        X_val = pp.par_transform(par_val, par_train)
        # Output variables
        y_train = pp.preproc(signal_train, signal_train)
        y_val = pp.preproc(signal_val, signal_train)
        super().__init__(
            X_train,
            X_val,
            y_train,
            y_val,
            em_type="direct",
            max_trials=max_trials,
            epochs=epochs,
            min_hidden_layers=min_hidden_layers,
            h_layer_step=h_layer_step,
            max_step_h_layers=max_step_h_layers,
            min_hidden_dim=min_hidden_dim,
            hidden_dim_step=hidden_dim_step,
            max_step_hidden_dim=max_step_hidden_dim
        )


class AE_HP_Tuner(HyperParameterTuner):

    def __init__(
            self,
            encoder_path,
            max_trials=500,
            epochs=350,
            min_hidden_layers=1,
            h_layer_step=1,
            max_step_h_layers=4,
            min_hidden_dim=8,
            hidden_dim_step=16,
            max_step_hidden_dim=5
        ):
        with h5py.File(__path__[0] + '/dataset_21cmVAE.h5', 'r') as hf:
            signal_train = hf['signal_train'][:]
            signal_val = hf['signal_val'][:]
            par_train = hf['par_train'][:]
            par_val = hf['par_val'][:]

        encoder = load_model(encoder_path)
        # Input variables
        X_train = pp.par_transform(par_train, par_train)
        X_val = pp.par_transform(par_val, par_train)
        # Output variables
        y_train = encoder.predict(pp.preproc(signal_train, signal_train))
        y_val = encoder.predict(pp.preproc(signal_val, signal_train))
        self.latent_dim = y_train.shape[-1]
        super().__init__(
            X_train,
            X_val,
            y_train,
            y_val,
            em_type="ae",
            max_trials=max_trials,
            epochs=epochs,
            min_hidden_layers=min_hidden_layers,
            h_layer_step=h_layer_step,
            max_step_h_layers=max_step_h_layers,
            min_hidden_dim=min_hidden_dim,
            hidden_dim_step=hidden_dim_step,
            max_step_hidden_dim=max_step_hidden_dim
        )
