import h5py
import numpy as np
import tensorflow as tf

from build_models import build_models
import preprocess as pp
from training_tools import train_models

class 21cmVAE:
    def __init__(self, vae=None, emulator=None):
        """
        :param vae: Keras model object, sets the default VAE if you already have a trained one
        :param emulator: Keras model object, sets the default emulator if you have one
        These models can be loaded from file with tf.keras.models.load_model(). If None, then the trained models
        giving the best performance in Bye et al. 2021 will be default (the h5 files are in the 'models' directory).
        Please note: the models *must* be Keras model objects (or None) for things to work properly if they're used.
        The default models will be updated to the most recently trained models with the train() method.
        """
        # initialize training set, validation set, and test set variables
        with h5py.File('dataset.h5', 'r') as hf:
            self.signal_train = hf['signal_train'][:]
            self.signal_val = hf['signal_val'][:]
            self.signal_test = hf['signal_test'][:]
            self.par_train = hf['par_train'][:]
            self.par_val = hf['par_val'][:]
            self.par_test = hf['par_test'][:]


        # initialize standard hyperparameters (used for the pretrained model)
        self.latent_dim = 22
        self.encoder_dims = [96, 224, 288, 32]
        self.decoder_dims = [288]
        self.em_dims = [160, 224]
        self.beta = 0.4
        self.gamma = 1.5

        # initialize parameters for training
        self.epochs = 350 # max number of epochs (can be less due to early stopping)
        self.vae_lr = 0.01 # initial learning rate for VAE
        self.em_lr = 0.01 # initial learning rate for emulator
        self.activation_func = 'relu' # activation function in all hidden layers
        self.lr_patience = # patience for learning rate scheduler
        self.es_patience =  # patience for early stopping

        # the default models are the pretrained ones
        if vae:
            self.vae = vae
        else:
            self.vae = tf.keras.models.load_model('models/vae.h5')
        if emulator:
            self.emulator = emulator
        else:
            self.emulator = tf.keras.models.load_model('models/emulator.h5')

    def set_hyperparameters(self, latent_dim=self.latent_dim, encoder_dims=self.encoder_dims,
                            decoder_dims=self.decoder_dims, em_dims=self.em_dims, beta=self.beta,
                            gamma=self.gamma):
        """
        Set the hyperparameters of the model.
        :param latent_dim: int, dimensionality of latent space
        :param encoder_dims: list of ints, dimensions of each encoder layer, e.g. [96, 224, 288, 32]
        :param decoder_dims: list of ints, dimensions of each decoder layer, e.g [288]
        :param em_dims: list of ints, dimensions of each emulator layer (not in the decoder), e.g [160, 224]
        :param beta: float, parameter in VAE loss function (see equation 3 in Bye et al. 2021)
        :param gamma: float, parameter in VAE loss function (see equation 3 in Bye et al. 2021)
        :return: None
        """
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.em_dims = em_dims
        self.beta = beta
        self.gamma = gamma
        return None

    def train(self, signal_train=self.signal_train, par_train=self.par_train, signal_val=self.signal_val, par_val=self.par_val,
              vae_lr=self.vae_lr, em_lr=self.em_lr, activation_func=self.activation_func, epochs=self.epochs):

        # update the properties
        self.signal_train = signal_train
        self.par_train = par_train
        self.signal_val = signal_val
        self.par_val = par_val
        self.vae_lr = vae_lr
        self.em_lr = em_lr
        self.activation_func = activation_func
        self.epochs = epochs

        # hyperparameters
        hps = {}
        hps['latent_dim'] = self.latent_dim
        hps['beta'] = self.beta
        hps['gamma'] = self.gamma
        layer_hps = [self.encoder_dims, self.decoder_dims, self.em_dims]

        # build vae and emulator
        vae, emulator = build_models(hps, layer_hps, vae_lr, em_lr, activation_func)

        # update the default models
        self.vae = vae
        self.emulator = emulator

        # Input variables
        X_train = pp.par_transform(par_train, par_train)
        X_val = pp.par_transform(par_val, par_train)
        train_amplitudes = np.max(np.abs(signal_train), axis=-1)
        val_amplitudes = np.max(np.abs(signal_val), axis=-1)

        # Output variables
        y_train = pp.preproc(signal_train, signal_train)
        y_val = pp.preproc(signal_val, signal_train)

        # create the training and validation minibatches
        dataset = create_batch(X_train, y_train, train_amplitudes)
        val_dataset = create_batch(X_val, y_val, val_amplitudes)

        losses = train_models(vae, emulator, dataset, val_dataset)

    def predict(self, params, model_name='21cmVAE', model_dir='models'):
        """
        Predict global signals from input parameters.
        :param params: Array of shape (N, 7) where N = number of signals to predict and the columns are the values
        of the parameters. The parameters must be in the order [fstar, Vc, fx, tau, alpha, nu_min, Rmfp]
        :param model: str, name of h5 file in the 'models' directory
        with the keras model that can be loaded with tf.models.load_emulator(). Default is the pretrained 21cmVAE.
        :param model_dir: str, path to directory where the model is saved. Default is 'models'.
        :return: Array with shape (N, 451) where each row is a global signal
        """
        assert type(model_name) == str, 'model name must be string'
        assert type(model_dir) == str, 'model directory must be string'
        model_path = model_dir+'/' + model_name
        emulator = tf.keras.models.load_model(model_path)
        transformed_params = pp.par_transform(params, self.par_train)  # transform the input parameters
        preprocessed_signal = emulator.predict(transformed_params)  # predict signal with emulator
        predicted_signal = pp.unpreproc(preprocessed_signal, self.signal_train)  # unpreprocess the signal
        return predicted_signal




def evaulate():
    """

    :return:
    """


def