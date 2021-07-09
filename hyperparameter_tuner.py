import h5py
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras import optimizers

import preprocess as pp
from build_models import build_models
from training_tools import em_loss_fcn, train_models


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

train_amplitudes = np.max(np.abs(signal_train), axis=-1)
val_amplitudes = np.max(np.abs(signal_val), axis=-1)
test_amplitudes = np.max(np.abs(signal_test), axis=-1)

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
min_encoding_dim = 15 # min. dimensionality of latent space
max_step_encoding_dim = 8 # max. dimensionality of latent space
encoding_dim_step = 1 # step size
min_beta = 0 # min. value of scaling factor in loss function, 0 == vanilla autoencoder
max_step_beta = 10 # max. value of scaling factor in loss function
beta_step = 0.2 # step size
min_gamma = 1.5
max_step_gamma = 11
gamma_step = 0.5


# fixed hyperparameters
activation_func = 'relu'
vae_lr = 0.01
em_lr = 0.01

# Reduce LR
vae_lr_factor = 0.7
vae_lr_patience = 5
vae_min_lr = 1e-6
em_lr_factor = 0.7
em_lr_patience = 5
em_min_lr = 1e-6
lr_max_factor = 0.95

# for early stopping
early_stop_patience = 15
es_max_factor = 0.99
#vae_early_stop_delta = 1e-4

'''
Random hp grid search. Input params are tuples of the form (min, step_size, max_step) for the given hp.
The possible outcomes are of the form min + k*step_size, 0 <= k <= max_step.
latent_dim: dimensionality of latent space
hidden_dims: dimnesionality of all hidden layers
no_encoder_layers = number of encoder layers (VAE)
no_decoding_layers = number of decoder layers (shared between emulator and VAE)
beta, gamma: for vae loss function
no_em_layers = number of hidden layers between input params and latent space (emulator) 
'''


def generate_hp(min, step_size, max_step):
  """
  Function that generates one hyperparameter given the search range
  :param min: float, min value of the hp
  :param step_size: float or int, resolution of grid
  :param max_step: int, max multiple of step_size that can be added to min
  :return: float, the hyperparameter
  """
  step = np.random.randint(0, max_step)
  return min + step * step_size


def generate_hps(latent_dim=(min_encoding_dim, encoding_dim_step, max_step_encoding_dim),
                 beta=(min_beta, beta_step, max_step_beta),
                 gamma=(min_gamma, gamma_step, max_step_gamma)):
  """
  Generate all the hyperparameters, by repeatedly calling the generate_hp()-function. All input args are tuples
  of the form (min, step size, max step).
  :param latent_dim: latent dimension of VAE
  :param beta: beta in VAE loss
  :param gamma: gamma in VAE loss
  :return: dict, the values of the latent dimension, beta, and gamma.
  """
  # generate hyperparams for latent dimension, beta and gamma
  args = [latent_dim, beta, gamma]
  arglabels = ['latent_dim', 'beta', 'gamma']
  hps = {}
  for i, arg in enumerate(args):
    hp = generate_hp(*arg)
    argname = arglabels[i]
    hps[argname] = hp
  return hps


def generate_layer_hps(no_encoder_layers=(min_hidden_layers, h_layer_step, max_step_h_layers),
                       no_decoding_layers=(min_hidden_layers, h_layer_step, max_step_h_layers),
                       no_em_layers=(min_hidden_layers, h_layer_step, max_step_h_layers),
                       hidden_dims=(min_hidden_dim, hidden_dim_step, max_step_hidden_dim)):
  """
  Get the hyperparameters that control the architecture, i.e. the number of layers and their dimensions. Again,
  input must be tuples on the form (min, step size, max number of steps).
  :param no_encoder_layers: Number of encoder layers
  :param no_decoding_layers: Number of decoder layers
  :param no_em_layers: Number of emulator layers
  :param hidden_dims: Dimensionality of the layers
  :return: tuple of lists, giving the dimensionality of each layer for the encoder, decoder, and emulator
  """
  #generate hyperparams for layers
  layer_args = [no_encoder_layers, no_decoding_layers, no_em_layers]
  # initialize empty lists
  enc_dims = []
  dec_dims = [] 
  em_dims = []
  # go through each model and find the layers and dimensionality for each
  for i, arg in enumerate(layer_args):
    no_layer = generate_hp(*arg)  # first, find the number of layers in the model
    dims = []
    for j in range(no_layer):
      dim = generate_hp(*hidden_dims) # dimensionality of each layer
      dims.append(dim)
    if i == 0:
      enc_dims = dims
    elif i == 1:
      dec_dims = dims
    else:
      em_dims = dims
  return enc_dims, dec_dims, em_dims


# Input variables
X_train = pp.par_transform(par_train, par_train)
X_val = pp.par_transform(par_val, par_train)

# Output variables
y_train = pp.preproc(signal_train, signal_train)
y_val = pp.preproc(signal_val, signal_train)

# create the training and validation minibatches
dataset = create_batch(X_train, y_train, train_amplitudes)
val_dataset = create_batch(X_val, y_val, val_amplitudes)

# save search range:
with open('search_range.txt', 'w') as f:
  f.write('\nmin_hidden_layers = {}'.format(min_hidden_layers))
  f.write('\nmax_step_h_layers = {}'.format(max_step_h_layers))
  f.write('\nh_layer_step = {}'.format(h_layer_step))
  f.write('\nmin_hidden_dim = {}'.format(min_hidden_dim))
  f.write('\nmax_step_hidden_dim = {}'.format(max_step_hidden_dim))
  f.write('\nhidden_dim_step = {}'.format(hidden_dim_step))
  f.write('\nmin_encoding_dim = {}'.format(min_encoding_dim))
  f.write('\nmax_step_encoding_dim = {}'.format(max_step_encoding_dim))
  f.write('\nencoding_dim_step = {}'.format(encoding_dim_step))
  f.write('\nmin_beta = {}'.format(min_beta))
  f.write('\nmax_step_beta = {}'.format(max_step_beta))
  f.write('\nbeta_step = {}'.format(beta_step))
  f.write('\nmin_gamma = {}'.format(min_gamma))
  f.write('\nmax_step_gamma = {}'.format(max_step_gamma))
  f.write('\ngamma_step = {}'.format(gamma_step))
  f.write('\nactivation_func = {}'.format(activation_func))
  f.write('\nvae_lr = {}'.format(vae_lr))
  f.write('\nem_lr = {}'.format(em_lr))
  f.write('\nvae_lr_factor = {}'.format(vae_lr_factor))
  f.write('\nvae_lr_patience = {}'.format(vae_lr_patience))
  f.write('\nvae_min_lr = {}'.format(vae_min_lr))
  f.write('\nem_lr_factor = {}'.format(em_lr_factor))
  f.write('\nem_lr_patience = {}'.format(em_lr_patience))
  f.write('\nLearning rate max factor = {}'.format(lr_max_factor))
  f.write('\nem_min_lr = {}'.format(em_min_lr))
  f.write('\nearly_stop_patience = {}'.format(early_stop_patience))
  f.write('\nEarly stop max factor = {}'.format(es_max_factor))


def save_results(trial, hps, layer_hps, vae, emulator, losses):
  """
  Function that saves the results of the trial to the file "results_trial_<trial_number>.txt"
  :param trial: int, the trial number
  :param hps: dict, the hyperparameters for latent dimension, beta, and gamma
  :param layer_hps: list of lists of dimensions of layers in encoder, decoder, and emulator
  :param vae: keras model instance of VAE
  :param emulator: keras model instance of emulator
  :param losses: list of floats giving the training and validation loss of the vae and emulator for the epoch
  that minimizes the emulator validation loss
  :return: None
  """
  fname = 'results_trial_' + str(trial)
  with open(fname+'.txt', 'w') as f:
    f.write('Hyperparameters:\n')
    f.write(str(hps))
    f.write('\n-----------\n')
    f.write('\n-----------\n')
    f.write('VAE Losses:\n')
    f.write('Train = {:.4f}'.format(losses[0]))
    f.write('\nValidation = {:.4f}'.format(losses[1]))
    f.write('\n\nEmulator Losses:')
    f.write('\nTrain = {:.4f}'.format(losses[2]))
    f.write('\nValidation = {:.4f}'.format(losses[3]))
    f.write('\n')
    vae.summary(print_fn=lambda x: f.write(x + '\n'))
    emulator.summary(print_fn=lambda x: f.write(x + '\n'))
  # save hps
  np.save(fname+'_hps.npy', hps)
  np.save(fname+'_layer_hps.npy', layer_hps)

def delete_results(trial):
  """
  Function that delete results of trial. This gets called when the trial is no longer among the five best.
  :param trial: int, the trial number
  :return: None
  """
  fname = 'results_trial_' + str(int(trial))
  os.remove(fname+'.txt')
  os.remove(fname+'_hps.npy')
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
    hps = generate_hps()  # first generate the hyperparameters
    layer_hps = generate_layer_hps()  # then the architecture
    # define optimizers
    vae_opt = optimizers.Adam(learning_rate=vae_lr)
    em_opt = optimizers.Adam(learning_rate=em_lr)
    # build the models
    vae, emulator = build_models(hps, layer_hps, signal_train, par_train)
    # compile the models
    vae.compile(vae_opt)
    emulator.compile(em_opt, loss=em_loss_fcn)
    # get the losses
    losses = train_models(vae, emulator, em_lr, vae_lr, signal_train, dataset, val_dataset, EPOCHS, vae_lr_factor,
                          em_lr_factor, vae_min_lr, em_min_lr, vae_lr_patience, em_lr_patience, lr_max_factor,
                          es_patience, es_max_factor)
    # we care about the epoch that gives the minimum validation loss for the emulator ...
    em_loss_val = losses[-1]
    min_indx = np.argmin(em_loss_val)  # ... this is that epoch
    min_losses = [loss[min_indx] for loss in losses]  # make a list of the values of the losses for that epoch
    em_loss_val_min = min_losses[-1]
    if i < 5:  # if this is among the first five trials, we save the results anyway
      five_best_val[i, 0] = i
      five_best_val[i, 1] = em_loss_val_min
      save_results(i, hps, layer_hps, vae, emulator, min_losses)
    elif em_loss_val_min < np.max(five_best_val[:, 1]):  # for subsequent trials, we save the results if they're better
      # than previous ones
      idx = np.argmax(five_best_val[:, 1])
      worst_trial = five_best_val[idx, 0]
      delete_results(worst_trial)
      five_best_val[idx, 0] = i
      five_best_val[idx, 1] = em_loss_val_min
      save_results(i, hps, layer_hps, vae, emulator, min_losses)
    # hopefully limit memory usage
    del vae
    del emulator
    K.clear_session()
  return five_best_val

five_best_trials = run_tuner()

# the very best trial:
best_trial_idx = np.argmin(five_best_trials[:, 1])
best_trial = int(five_best_trials[best_trial_idx, 0])
# load the hyperparameters of that trial
best_hps = np.load('results_trial_'+str(best_trial)+'_hps.npy', allow_pickle=True).item()
best_layer_hps = []
best_layer_hps_list = np.load('results_trial_'+str(best_trial)+'_layer_hps.npy', allow_pickle=True)
for i in range(len(best_layer_hps_list)):
  best_layer_hps.append(best_layer_hps_list[i])

# now, retrain the emulator and VAE with the best hyperparameters
tuned_vae, tuned_em = build_models(best_hps, best_layer_hps, signal_train)
vae_opt = optimizers.Adam(learning_rate=vae_lr)
em_opt = optimizers.Adam(learning_rate=em_lr)
tuned_vae.compile(vae_opt)
tuned_em.compile(em_opt, loss=em_loss_fcn)

vae_loss, vae_loss_val, em_loss, em_loss_val = train_models(tuned_vae, tuned_emulator, em_lr, vae_lr, signal_train,
                                                            dataset, val_dataset, 250, vae_lr_factor, em_lr_factor,
                                                            vae_min_lr, em_min_lr, vae_lr_patience, em_lr_patience,
                                                            lr_max_factor, es_patience, es_max_factor)

# Can plot the  training curves like this
# import matplotlib.pyplot as plt
# if not 'plots' in os.listdir():
#   os.makedirs('plots')
#
# x_vals = np.arange(len(vae_loss)) + 1
# plt.plot(x_vals, vae_loss, label='Train')
# plt.plot(x_vals, vae_loss_val, label='Validation')
# plt.ylabel('Loss')
# plt.title('VAE')
# plt.xlabel('Epoch')
# plt.legend()
# plt.savefig('plots/tuned_vae_training_curve.png')
# plt.show()
#
# plt.plot(x_vals, em_loss, label='Train')
# plt.plot(x_vals, em_loss_val, label='Validation')
# plt.axvline(np.argmin(em_loss_val)+1, ls='--', c='black')
# plt.ylabel('Loss')
# plt.title('Emulator')
# plt.xlabel('Epoch')
# plt.legend()
# plt.savefig('plots/tuned_em_training_curve.png')
# plt.show()

