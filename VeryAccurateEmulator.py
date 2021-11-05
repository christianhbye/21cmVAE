import h5py
import numpy as np
import tensorflow as tf
import build_models as bm
import preprocess as pp
from training_tools import train_ae_emulator, train_direct_emulator, em_loss_fcn


class VeryAccurateEmulator:
    def __init__(self, direct_emulator=None):
        """
        :param direct_emulator: Keras model object, sets the default direct emulator if you have one
        :return None
        """
        # initialize training set, validation set, and test set variables
        with h5py.File('dataset_21cmVAE.h5', 'r') as hf:
            self.signal_train = hf['signal_train'][:]
            self.signal_val = hf['signal_val'][:]
            self.signal_test = hf['signal_test'][:]
            self.par_train = hf['par_train'][:]
            self.par_val = hf['par_val'][:]
            self.par_test = hf['par_test'][:]

        self.par_labels = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp']

        # initialize standard hyperparameters (used for the pretrained model)
        self.direct_em_dims = [288, 352, 288, 224]
        self.activation_func = 'relu'  # activation function in all hidden layers

        # initialize parameters for training
        self.epochs = 350  # max number of epochs (can be less due to early stopping)
        self.em_lr = 0.01  # initial learning rate for emulator

        # Parameters that control the learning rate schedule during training:
        self.em_lr_factor = 0.7
        self.em_lr_patience = 5
        self.em_min_lr = 1e-6
        # if the loss doesn't decrease to less than this factor, LR is reduced:
        self.lr_max_factor = 0.95
        # for early stopping
        self.es_patience = 15  # number of epochs to wait before stopping training
        # if the loss doesn't decrease to less than this factor, training is stopped:
        self.es_max_factor = 0.99

        if direct_emulator:
            self.direct_emulator = direct_emulator
        else:
            self.direct_emulator = tf.keras.models.load_model('models/emulator.h5',
                                                              custom_objects={'em_loss_fcn': em_loss_fcn})

        # initialize lists with losses, these get updated when models are trained
        self.direct_em_train_losses = []  # training set losses for direct emulator
        self.direct_em_val_losses = []  # validation set losses for direct emulator

        # sampling redshifts: it is possible to train with signals that are not sampled with the same resolution
        # or across the same redshift/frequency range. IN that case, these properties should be updated.
        self.z_sampling = np.arange(5, 50 + 0.1, 0.1)  # redshifts

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
        direct_em_dims: list of ints, dimensions of each direct emulator layer, e.g [288, 352, 288, 224]
        activation_function: str, name of a keras recognized activation function or a tf.keras.activations instance
        (see https://keras.io/api/layers/activations/). Used in all hidden layers.
        :return: None
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['direct_em_dims', 'activation_func']):
                raise KeyError("Unexpected keyword argument in set_hyperparameters()")

        self.direct_em_dims = kwargs.pop('direct_em_dims', self.direct_em_dims)
        self.activation_func = kwargs.pop('activation_function', self.activation_func)

        return None

    def get_hyperparameters(self):
        """
        Method that prints the current hyperparameters. Takes no input arguments.
        :return: None
        """
        print('Hyperparameters are set to:')
        print('Direct emulator dimensions:', self.direct_em_dims)
        print('Activation function:', self.activation_func)

    def train(self, **kwargs):
        """
        Builds and trains the direct emulator without VAE. Possible kwargs are
        signal_train: numpy array of training signals
        par_train: numpy array of training set parameters
        signal_val: numpy array of validation signals
        par_val: numpy array of validation set parameters
        em_lr: float, initial learning rate of emulator
        epochs: int, number of epochs to train for
        em_lr_factor: float, factor * old LR (learning rate) is the new LR for the emulator (used for LR schedule)
        em_lr_patience: float, max number of epochs loss has not decreased for the emulator before reducing LR
        (used for LR schedule)
        em_min_lr: float, minimum allowed learning rate for emulator
        lr_max_factor: float, max_factor * current loss is the max acceptable loss, a larger loss means that the counter
        is added to, when it reaches the 'patience', the LR is reduced (used for LR schedule)
        es_patience: float, max number of epochs loss has not decreased before early stopping
        es_max_factor: float, max_factor * current loss is the max acceptable loss, a larger loss for either the VAE or
        the emulator means that the counter is added to, when it reaches the 'patience', early stopping is applied

        :return: None
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['signal_train', 'par_train', 'signal_val', 'par_val', 'em_lr', 'epochs',
                     'em_lr_factor', 'em_lr_patience', 'em_min_lr', 'lr_max_factor', 'es_patience', 'es_max_factor']):
                raise KeyError("Unexpected keyword argument in train()")

        # update the properties
        self.signal_train = kwargs.pop('signal_train', self.signal_train)
        self.par_train = kwargs.pop('par_train', self.par_train)
        self.signal_val = kwargs.pop('signal_val', self.signal_val)
        self.par_val = kwargs.pop('par_val', self.par_val)
        self.em_lr = kwargs.pop('em_lr', self.em_lr)
        self.epochs = kwargs.pop('epochs', self.epochs)
        self.em_lr_factor = kwargs.pop('em_lr_factor', self.em_lr_factor)
        self.em_lr_patience = kwargs.pop('em_lr_patience', self.em_lr_patience)
        self.em_min_lr = kwargs.pop('em_min_lr', self.em_min_lr)
        self.lr_max_factor = kwargs.pop('lr_max_factor', self.lr_max_factor)
        self.es_patience = kwargs.pop('es_patience', self.es_patience)
        self.es_max_factor = kwargs.pop('es_max_factor', self.es_max_factor)

        # build direct emulator
        direct_emulator = bm.build_direct_emulator(self.direct_em_dims, self.signal_train, self.par_train,
                                                self.activation_func)

        # update the default models
        self.direct_emulator = direct_emulator

        losses = train_direct_emulator(direct_emulator, self.em_lr, self.signal_train, self.signal_val, self.par_train,
                                       self.par_val, self.epochs, self.em_lr_factor, self.em_min_lr,
                                       self.em_lr_patience, self.lr_max_factor, self.es_patience, self.es_max_factor)

        self.direct_em_train_losses = losses[0]
        self.direct_em_val_losses = losses[1]

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
        model = self.direct_emulator
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

        if flow is not None or fhigh is not None:
            if relative:
                print("One or two frequency bounds are specified, but 'relative' is set to True so the relative"
                      " error will be computed and the frequency bounds ignored. Did you mean to set relative=False?")

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


class AutoEncoderEmulator:
    def __init__(self, **kwargs):
        """
        :param autoencoder: Keras model object, sets the default VAE if you already have a trained one
        :param encoder:
        :param decoder:  make sure it matches the autoencoder
        :param emulator: Keras model object, sets the default emulator if you have one
        These models can be loaded from file with tf.keras.models.load_model(). If None, then the trained models
        giving the best performance in Bye et al. 2021 will be default (the h5 files are in the 'models' directory).
        Please note: the models *must* be Keras model objects (or None) for things to work properly if they're used.
        The default models will be updated to the most recently trained models with the train() method.
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['autoencoder', 'encoder', 'decoder', 'emulator']):
                raise KeyError("Unexpected keyword argument in class AutoEncoderEmulator")

        # initialize training set, validation set, and test set variables
        with h5py.File('dataset_21cmVAE.h5', 'r') as hf:
            self.signal_train = hf['signal_train'][:]
            self.signal_val = hf['signal_val'][:]
            self.signal_test = hf['signal_test'][:]
            self.par_train = hf['par_train'][:]
            self.par_val = hf['par_val'][:]
            self.par_test = hf['par_test'][:]

        self.par_labels = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp']

        # initialize standard hyperparameters (used for the pretrained model)
        self.latent_dim = 9
        self.encoder_dims = [352]
        self.decoder_dims = [32, 352]
        self.em_dims = [352, 352, 352, 224]
        self.activation_func = 'relu'  # activation function in all hidden layers

        # initialize parameters for training
        self.epochs = 350  # max number of epochs (can be less due to early stopping)
        self.ae_lr = 0.001  # initial learning rate for VAE
        self.em_lr = 0.01  # initial learning rate for emulator

        # Parameters that control the learning rate schedule during training:
        self.ae_lr_factor = 0.75  # factor LR is multiplied with when reduced
        self.ae_lr_patience = 5  # number of epochs to wait before reducing LR
        self.ae_min_lr = 1e-6  # minimum allowed LR
        self.em_lr_factor = 0.85
        self.em_lr_patience = 5
        self.em_min_lr = 1e-6
        # if the loss doesn't decrease by more than this, LR is reduced:
        self.ae_min_delta = 1e-5
        self.em_min_delta = 1e-4
        # for early stopping
        self.es_patience = 15  # number of epochs to wait before stopping training
        # if the loss doesn't decrease by more than this, training is stopped:
        self.ae_earlystop_delta = 5e-6
        self.em_earlystop_delta = 1e-4

        # the default models are the pretrained ones
        model_path = 'models/autoencoder_based_emulator/'
        self.autoencoder = kwargs.pop('autoencoder', tf.keras.models.load_model(model_path+'autoencoder.h5'))
        self.encoder = kwargs.pop('encoder', tf.keras.models.load_model(model_path+'encoder.h5'))
        self.decoder = kwargs.pop('decoder', tf.keras.models.load_model(model_path+'decoder.h5'))
        self.emulator = kwargs.pop('emulator', tf.keras.models.load_model(model_path+'ae_emulator.h5',
                                                                          custom_objects={'em_loss_fcn': em_loss_fcn}))

        # initialize lists with losses, these get updated when models are trained
        self.ae_train_losses = []  # training set losses for VAE
        self.ae_val_losses = []  # validation set losses for VAE
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
        latent_dim: int, dimensionality of latent space
        encoder_dims: list of ints, dimensions of each encoder layer, e.g. [96, 224, 288, 32]
        decoder_dims: list of ints, dimensions of each decoder layer, e.g [288]
        em_dims: list of ints, dimensions of each emulator layer (not in the decoder), e.g [160, 224]
        activation_function: str, name of a keras recognized activation function or a tf.keras.activations instance
        (see https://keras.io/api/layers/activations/). Used in all hidden layers.
        :return: None
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['latent_dim', 'encoder_dims', 'decoder_dims', 'em_dims', 'activation_func']):
                raise KeyError("Unexpected keyword argument in set_hyperparameters()")

        self.latent_dim = kwargs.pop('latent_dim', self.latent_dim)
        self.encoder_dims = kwargs.pop('encoder_dims', self.encoder_dims)
        self.decoder_dims = kwargs.pop('decoder_dims', self.decoder_dims)
        self.em_dims = kwargs.pop('em_dims', self.em_dims)
        self.activation_func = kwargs.pop('activation_function', self.activation_func)

        return None

    def get_hyperparameters(self):
        """
        Method that prints the current hyperparameters. Takes no input arguments.
        :return: None
        """
        print('Hyperparameters are set to:')
        print('Latent dimension:', self.latent_dim)
        print('Encoder dimensions:', self.encoder_dims)
        print('Decoder dimensions:', self.decoder_dims)
        print('Emulator dimensions:', self.em_dims)
        print('Activation function:', self.activation_func)

    def train(self, **kwargs):
        """
        Builds and trains a VAE and emulator simultaneously. Possible kwargs are
        signal_train: numpy array of training signals
        par_train: numpy array of training set parameters
        signal_val: numpy array of validation signals
        par_val: numpy array of validation set parameters
        vae_lr: float, initial learning rate of VAE
        em_lr: float, initial learning rate of emulator
        epochs: int, number of epochs to train for
        vae_lr_factor: float, factor * old LR (learning rate) is the new LR for the VAE (used for LR schedule)
        vae_lr_patience: float, max number of epochs loss has not decreased for the VAE before reducing LR (used for
        LR schedule)
        vae_min_lr: float, minimum allowed learning rate for VAE
        em_lr_factor: float, factor * old LR (learning rate) is the new LR for the emulator (used for LR schedule)
        em_lr_patience: float, max number of epochs loss has not decreased for the emulator before reducing LR
        (used for LR schedule)
        em_min_lr: float, minimum allowed learning rate for emulator
        lr_max_factor: float, max_factor * current loss is the max acceptable loss, a larger loss means that the counter
        is added to, when it reaches the 'patience', the LR is reduced (used for LR schedule)
        es_patience: float, max number of epochs loss has not decreased before early stopping
        es_max_factor: float, max_factor * current loss is the max acceptable loss, a larger loss for either the VAE or
        the emulator means that the counter is added to, when it reaches the 'patience', early stopping is applied

        :return: None
        """
        for key, values in kwargs.items():
            if key not in set(
                    ['signal_train', 'par_train', 'signal_val', 'par_val', 'vae_lr', 'em_lr', 'epochs',
                     'vae_lr_factor', 'vae_lr_patience', 'vae_min_lr', 'em_lr_factor', 'em_lr_patience',
                     'em_min_lr', 'lr_max_factor', 'es_patience', 'es_max_factor']):
                raise KeyError("Unexpected keyword argument in train()")

        # update the properties
        self.signal_train = kwargs.pop('signal_train', self.signal_train)
        self.par_train = kwargs.pop('par_train', self.par_train)
        self.signal_val = kwargs.pop('signal_val', self.signal_val)
        self.par_val = kwargs.pop('par_val', self.par_val)
        self.ae_lr = kwargs.pop('ae_lr', self.ae_lr)
        self.em_lr = kwargs.pop('em_lr', self.em_lr)
        self.epochs = kwargs.pop('epochs', self.epochs)
        self.ae_min_delta = kwargs.pop('ae_min_delta', self.ae_min_delta)
        self.ae_earlystop_delta = kwargs.pop('ae_earlystop_delta', self.ae_earlystop_delta)
        self.ae_lr_factor = kwargs.pop('ae_lr_factor', self.ae_lr_factor)
        self.ae_lr_patience = kwargs.pop('ae_lr_patience', self.ae_lr_patience)
        self.ae_min_lr = kwargs.pop('ae_min_lr', self.ae_min_lr)
        self.em_lr_factor = kwargs.pop('em_lr_factor', self.em_lr_factor)
        self.em_lr_patience = kwargs.pop('em_lr_patience', self.em_lr_patience)
        self.em_min_lr = kwargs.pop('em_min_lr', self.em_min_lr)
        self.em_min_delta = kwargs.pop('em_min_delta', self.em_min_delta)
        self.em_earlystop_delta = kwargs.pop('em_earlystop_delta', self.em_earlystop_delta)
        self.es_patience = kwargs.pop('es_patience', self.es_patience)

        # hyperparameters
        layer_hps = [self.encoder_dims, self.latent_dim, self.decoder_dims, self.em_dims]

        # build autoencoder and emulator
        autoencoder, encoder, decoder = bm.build_autoencoder(layer_hps, self.activation_func)
        emulator = bm.build_ae_emulator(layer_hps, self.activation_func)

        # update the default models
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.emulator = emulator

        losses = train_ae_emulator(self.autoencoder, self.encoder, self.emulator, self.signal_train, self.signal_val,
                                   self.par_train, self.par_val, self.epochs, self.ae_lr_factor, self.ae_lr_patience,
                                   self.ae_lr_min_delta, self.ae_min_lr, self.ae_earlystop_delta, self.es_patience,
                                   self.em_lr_factor, self.em_lr_patience, self.em_lr_min_delta, self.em_min_lr,
                                   self.em_earlystop_delta, self.es_patience)

        self.ae_train_losses = losses[0]
        self.ae_val_losses = losses[1]
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
        decoder = self.decoder
        training_params = self.par_train
        training_signals = self.signal_train
        transformed_params = pp.par_transform(params, training_params)  # transform the input parameters
        preprocessed_signal = model.predict(transformed_params)  # predict signal with emulator
        decoded = decoder.predict(preprocessed_signal)  # decode predicted signal
        predicted_signal = pp.unpreproc(decoded, training_signals)  # unpreprocess the signal
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

        if flow is not None or fhigh is not None:
            if relative:
                print("One or two frequency bounds are specified, but 'relative' is set to True so the relative"
                      " error will be computed and the frequency bounds ignored. Did you mean to set relative=False?")

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


