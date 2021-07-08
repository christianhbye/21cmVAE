import h5py
import numpy as np
import tensorflow as tf

from build_models import build_models
import preprocess as pp
from training_tools import train_models, create_batch, em_loss_fcn


class VeryAccurateEmulator():
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

        self.par_labels = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp']

        # initialize standard hyperparameters (used for the pretrained model)
        self.latent_dim = 22
        self.encoder_dims = [96, 224, 288, 32]
        self.decoder_dims = [288]
        self.em_dims = [160, 224]
        self.beta = 0.4
        self.gamma = 1.5

        # initialize parameters for training
        self.epochs = 350  # max number of epochs (can be less due to early stopping)
        self.vae_lr = 0.01  # initial learning rate for VAE
        self.em_lr = 0.01  # initial learning rate for emulator
        self.activation_func = 'relu'  # activation function in all hidden layers

        # Parameters that control the learning rate schedule during training:
        self.vae_lr_factor = 0.7  # factor LR is multiplied with when reduced
        self.vae_lr_patience = 5  # number of epochs to wait before reducing LR
        self.vae_min_lr = 1e-6  # minimum allowed LR
        self.em_lr_factor = 0.7
        self.em_lr_patience = 5
        self.em_min_lr = 1e-6
        # if the loss doesn't decrease to less than this factor, LR is reduced:
        self.lr_max_factor = 0.95
        # for early stopping
        self.es_patience = 15  # number of epochs to wait before stopping training
        # if the loss doesn't decrease to less than this factor, training is stopped:
        self.es_max_factor = 0.99

        # the default models are the pretrained ones
        if vae:
            self.vae = vae
        else:
            self.vae = tf.keras.models.load_model('models/vae.h5')
        if emulator:
            self.emulator = emulator
        else:
            self.emulator = tf.keras.models.load_model('models/emulator.h5', custom_objects={'em_loss_fcn': em_loss_fcn})

        # initialize lists with losses, these get updated when models are trained
        self.vae_train_losses = []  # training set losses for VAE
        self.vae_val_losses = []  # validation set losses for VAE
        self.em_train_losses = []  # training set losses for emulator
        self.em_val_losses = []  # validation set losses for emulator

        # sampling redshifts: it is possible to train with signals that are not sampled with the same resolution
        # or across the same redshift/frequency range. IN that case, these properties should be updated.
        self.z_sampling = np.arange(5, 50+0.1, 0.1)  # redshifts

        def z_to_nu(redshift):
            rest_frequency = 1420405751.7667  # rest frequency in Hz
            freqs = rest_frequency / (1 + redshift)
            freqs /= 1e6  # convert to MHz
            return freqs

        self.nu_sampling = z_to_nu(self.z_sampling)  # frequencies

    def set_hyperparameters(self, **kwargs):
        """
        Set the hyperparameters of the model.
        Possible **kwargs are:
        :param latent_dim: int, dimensionality of latent space
        :param encoder_dims: list of ints, dimensions of each encoder layer, e.g. [96, 224, 288, 32]
        :param decoder_dims: list of ints, dimensions of each decoder layer, e.g [288]
        :param em_dims: list of ints, dimensions of each emulator layer (not in the decoder), e.g [160, 224]
        :param beta: float, parameter in VAE loss function (see equation 3 in Bye et al. 2021)
        :param gamma: float, parameter in VAE loss function (see equation 3 in Bye et al. 2021)
        :return: None
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['latent_dim', 'encoder_dims', 'decoder_dims', 'em_dims', 'beta', 'gamma']):
                raise KeyError("Unexpected keyword argument in set_hyperparameters()")

        self.latent_dim = kwargs.pop('latent_dim', self.latent_dim)
        self.encoder_dims = kwargs.pop('encoder_dims', self.encoder_dims)
        self.decoder_dims = kwargs.pop('decoder_dims', self.decoder_dims)
        self.em_dims = kwargs.pop('em_dims', self.em_dims)
        self.beta = kwargs.pop('beta', self.beta)
        self.gamma = kwargs.pop('gamma', self.gamma)

        return None

    def get_hyperparameters(self):
        print('Hyperparameters are set to:')
        print('Latent dimension:', self.latent_dim)
        print('Encoder dimensions:', self.encoder_dims)
        print('Decoder dimensions:', self.decoder_dims)
        print('Emulator dimensions:', self.em_dims)
        print('Beta:', self.beta)
        print('Gamma:', self.gamma)

    def train(self, **kwargs):
        """
        Builds and trains a VAE and emulator simultaneously. Possible kwargs are
        signal_train:
        ...
        To be completed
        :return: None
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['signal_train', 'par_train', 'signal_val', 'par_val', 'vae_lr', 'em_lr', 'activation_func',
                     'epochs', 'vae_lr_factor', 'vae_lr_patience', 'vae_min_lr', 'em_lr_factor', 'em_lr_patience',
                     'em_lr_factor', 'em_min_lr', 'lr_max_factor', 'es_patience', 'es_max_factor']):
                raise KeyError("Unexpected keyword argument in train()")

        # update the properties
        self.signal_train = kwargs.pop('signal_train', self.signal_train)
        self.par_train = kwargs.pop('par_train', self.par_train)
        self.signal_val = kwargs.pop('signal_val', self.signal_val)
        self.par_val = kwargs.pop('par_val', self.par_val)
        self.vae_lr = kwargs.pop('vae_lr', self.vae_lr)
        self.em_lr = kwargs.pop('em_lr', self.em_lr)
        self.activation_func = kwargs.pop('activation_func', self.activation_func)
        self.epochs = kwargs.pop('epochs', self.epochs)
        self.vae_lr_factor = kwargs.pop('vae_lr_factor', self.vae_lr_factor)
        self.vae_lr_patience = kwargs.pop('vae_lr_patience', self.vae_lr_patience)
        self.vae_min_lr = kwargs.pop('vae_min_lr', self.vae_min_lr)
        self.em_lr_factor = kwargs.pop('em_lr_factor', self.em_lr_factor)
        self.em_lr_patience = kwargs.pop('em_lr_patience', self.em_lr_patience)
        self.em_min_lr = kwargs.pop('em_min_lr', self.em_min_lr)
        self.lr_max_factor = kwargs.pop('lr_max_factor', self.lr_max_factor)
        self.es_patience = kwargs.pop('es_patience', self.es_patience)
        self.es_max_factor = kwargs.pop('es_max_factor', self.es_max_factor)

        # hyperparameters
        hps = dict(latent_dim=self.latent_dim, beta=self.beta, gamma=self.gamma)
        layer_hps = [self.encoder_dims, self.decoder_dims, self.em_dims]

        # build vae and emulator
        vae, emulator = build_models(hps, layer_hps, self.signal_train, self.par_train, self.activation_func)

        # update the default models
        self.vae = vae
        self.emulator = emulator

        # Input variables
        X_train = pp.par_transform(self.par_train, self.par_train)
        X_val = pp.par_transform(self.par_val, self.par_train)
        train_amplitudes = np.max(np.abs(self.signal_train), axis=-1)
        val_amplitudes = np.max(np.abs(self.signal_val), axis=-1)

        # Output variables
        y_train = pp.preproc(self.signal_train, self.signal_train)
        y_val = pp.preproc(self.signal_val, self.signal_train)

        # create the training and validation minibatches
        dataset = create_batch(X_train, y_train, train_amplitudes)
        val_dataset = create_batch(X_val, y_val, val_amplitudes)

        losses = train_models(vae, emulator, self.em_lr, self.vae_lr,
                              self.signal_train, dataset, val_dataset, self.epochs, self.vae_lr_factor,
                              self.em_lr_factor, self.vae_min_lr, self.em_min_lr, self.vae_lr_patience,
                              self.em_lr_patience, self.lr_max_factor, self.es_patience, self.es_max_factor)

        self.vae_train_losses = losses[0]
        self.vae_val_losses = losses[1]
        self.em_train_losses = losses[2]
        self.em_val_losses = losses[3]

    def predict(self, params):
        """
        Predict global signals from input parameters. The training parameters and training signals matters
        for inverting the preprocessing of signals so these parameters must correspond to the training set the emulator
        was trained on. These parameters are set correctly if a model is trained with the train() function or
        the default model is used. Otherwise, the properties must be updated manually with 21cmVAE.par_train = ...
        and 21cmVAE.signal_train = ...
        :param params: Array of shape (N, 7) where N = number of signals to predict and the columns are the values
        of the parameters. The parameters must be in the order [fstar, Vc, fx, tau, alpha, nu_min, Rmfp]
        :return: Array with shape (N, 451) where each row is a global signal
        """
        model = self.emulator
        training_params = self.par_train
        training_signals = self.signal_train
        transformed_params = pp.par_transform(params, training_params)  # transform the input parameters
        preprocessed_signal = model.predict(transformed_params)  # predict signal with emulator
        predicted_signal = pp.unpreproc(preprocessed_signal, training_signals)  # unpreprocess the signal
        if predicted_signal.shape[0] == 1:
            return predicted_signal[0, :]
        else:
            return predicted_signal

    def compute_rms_error(self, **kwargs):
        """
        Computes the rms error as given in the paper, either a relative error or an absolute error in mK. If absolute
        error, then different frequency bands can be chosen.
        Possible kwargs
        :param test_params: array, with shape (N, 7) of parameters to test on where N is the number of different
        parameters to try at once (for  a vectorised call)
        :param test_signals: array with shape (N, 451) [451 is flexible, depends on what signals the model is trained on]
        of global signals corresponding to the test parameters
        :param relative: boolean, whether the error computed should be relative or absolute
        :param flow: float or None, lower bound for range of frequencies over which the rms error is computed. If None,
        there's no lower bound.
        :param fhigh: float or None, upper bound for range of frequencies over which the rms error is computed. If None,
        there's no upper bound.
        :return: array of shape (N, ), each row is the error for that signal
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['test_params', 'test_signals', 'relative', 'flow', 'fhigh']):
                raise KeyError("Unexpected keyword argument in compute_rms_error()")

        test_params = kwargs.pop('test_params', self.par_test)
        test_signals = kwargs.pop('test_signals', self.signal_test)
        relative = kwargs.pop('relative', True)
        flow = kwargs.pop('flow', None)
        fhigh = kwargs.pop('fhigh', None)

        predicted_signal_input = self.predict(test_params)
        true_signal_input = test_signals
        assert predicted_signal_input.shape == true_signal_input.shape
        if len(predicted_signal_input.shape) == 1:
            predicted_signal = np.expand_dims(predicted_signal_input, axis=0)
            true_signal = np.expand_dims(true_signal_input, axis=0)
        else:
            predicted_signal = predicted_signal_input.copy()
            true_signal = true_signal_input.copy()
        if not relative:
            nu_arr = self.nu_sampling
            assert nu_arr.shape[0] == predicted_signal.shape[1], "double check 21cmVAE.nu_sampling, it " \
                                                                 "does not seem to match the shape of the" \
                                                                 "predicted signal"
            if flow and fhigh:
                f = np.argwhere((nu_arr >= flow) & (nu_arr <= fhigh))[:, 0]
                predicted_signal = predicted_signal[:, f]
                true_signal = true_signal[:, f]
            elif flow:
                f1 = np.argwhere(nu_arr >= flow)
                predicted_signal = predicted_signal[:, f1]
                true_signal = true_signal[:, f1]
            elif fhigh:
                f2 = np.argwhere(nu_arr <= fhigh)
                predicted_signal = predicted_signal[:, f2]
                true_signal = true_signal[:, f2]
        num = np.sqrt(np.mean((predicted_signal - true_signal) ** 2, axis=1))
        if relative:  # give error as fraction of amplitude
            den = np.max(np.abs(true_signal), axis=1)
        else:  # give error in mK
            den = 1
        return num / den

