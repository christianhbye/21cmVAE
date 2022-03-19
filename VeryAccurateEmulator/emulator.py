import h5py
import tensorflow as tf
from tqdm.keras import TqdmCallback
import numpy as np

from VeryAccurateEmulator import __path__
import VeryAccurateEmulator.preprocess as pp

PATH = __path__[0] + "/"


def gen_model(in_dim, hidden_dims, out_dim, activation_func, name=None):
    layers = []
    if in_dim is not None:
        input_layer = tf.keras.Input(shape=(in_dim,))
        layers.append(input_layer)
    if len(hidden_dims):
        for dim in hidden_dims:
            layer = tf.keras.layers.Dense(dim, activation=activation_func)
            layers.append(layer)
    output_layer = tf.keras.layers.Dense(out_dim)
    layers.append(output_layer)
    model = tf.keras.Sequential(layers, name=name)
    return model


def relative_mse_loss(signal_train):
    """
    The square of the FoM in the paper, in units of standard deviation
    since the signals are preproccesed. We need a wrapper function to
    pass signal_train as an input param.
    :param signal_train: numpy array of training signals
    :param y_true: array, the true signal concatenated with the amplitude
    :param y_pred: array, the predicted signal (by the emulator)
    :return: callable, the loss function
    """

    def loss_function(y_true, y_pred):
        # unpreproc signal to get the ampltiude
        signal = y_true + tf.convert_to_tensor(np.mean(signal_train, axis=0))
        # get amplitude in units of standard deviation of signals
        reduced_amp = tf.math.reduce_max(tf.abs(signal), axis=1, keepdims=True)
        # loss is mse / square of amplitude
        loss = tf.keras.metrics.mean_squared_error(y_pred, y_true)
        loss /= tf.keras.backend.square(reduced_amp)
        return loss

    return loss_function


NU_0 = 1420405751.7667  # Hz, rest frequency of 21-cm line


def redshift2freq(z):
    nu = NU_0 / (1 + z)
    nu /= 1e6  # convert to MHz
    return nu


def freq2redshift(nu):
    """
    Frequency in MHz.
    """
    nu *= 1e6  # to Hz
    z = NU_0 / nu - 1
    return z


def error(
    true_signal, pred_signal, nu_arr, relative=True, flow=None, fhigh=None
):
    if len(pred_signal.shape) == 1:
        pred_signal = np.expand_dims(pred_signal, axis=0)
        true_signal = np.expand_dims(true_signal, axis=0)

    if flow and fhigh:
        f = np.argwhere((nu_arr >= flow) & (nu_arr <= fhigh))[:, 0]
    elif flow:
        f = np.argwhere(nu_arr >= flow)
    elif fhigh:
        f = np.argwhere(nu_arr <= fhigh)

    if flow or fhigh:
        pred_signal = pred_signal[:, f]
        true_signal = true_signal[:, f]

    err = np.sqrt(np.mean((pred_signal - true_signal) ** 2, axis=1))
    if relative:  # give error as fraction of amplitude in the desired band
        err /= np.max(np.abs(true_signal), axis=1)
        err *= 100  # %
    return err


# default parameters
hidden_dims = [288, 352, 288, 224]
redshifts = np.linspace(5, 50, 451)
with h5py.File(PATH + "dataset_21cmVAE.h5") as hf:
    par_train = hf["par_train"][:]
    par_val = hf["par_val"][:]
    par_test = hf["par_test"][:]
    signal_train = hf["signal_train"][:]
    signal_val = hf["signal_val"][:]
    signal_test = hf["signal_test"][:]



class DirectEmulator:
    def __init__(
        self,
        par_train=par_train,
        par_val=par_val,
        par_test=par_test,
        signal_train=signal_train,
        signal_val=signal_val,
        signal_test=signal_test,
        hidden_dims=hidden_dims,
        activation_func="relu",
        redshifts=redshifts,
        frequencies=None,
    ):

        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [
            "fstar",
            "Vc",
            "fx",
            "tau",
            "alpha",
            "nu_min",
            "Rmfp",
        ]

        self.emulator = gen_model(
            self.par_train.shape[-1],
            hidden_dims,
            self.signal_train.shape[-1],
            activation_func,
            name="emulator",
        )

        if frequencies is None:
            if redshifts is not None:
                frequencies = redshift2freq(redshifts)
        elif redshifts is None:
            redshifts = freq2redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=PATH + "models/emulator.h5"):
        custom_obj = {"loss_function": relative_mse_loss(self.signal_train)}
        self.emulator = tf.keras.models.load_model(
            model_path, custom_objects=custom_obj
        )

    def train(self, epochs, callbacks=[], verbose="tqdm"):

        X_train = pp.par_transform(self.par_train, self.par_train)
        X_val = pp.par_transform(self.par_val, self.par_train)
        y_train = pp.preproc(self.signal_train, self.signal_train)
        y_val = pp.preproc(self.signal_val, self.signal_train)

        if verbose == "tqdm":
            callbacks.append(TqdmCallback())
            verbose = 0
        hist = self.emulator.fit(
            x=X_train,
            y=y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=(X_val, y_val),
            validation_batch_size=256,
            callbacks=callbacks,
            verbose=verbose,
        )
        loss = hist.history["loss"]
        val_loss = hist.history["val_loss"]
        return loss, val_loss

    def predict(self, params):
        transformed_params = pp.par_transform(params, self.par_train)
        proc_pred = self.emulator.predict(transformed_params)
        pred = pp.unpreproc(proc_pred, self.signal_train)
        if pred.shape[0] == 1:
            return pred[0, :]
        else:
            return pred

    def test_error(self, relative=True, flow=None, fhigh=None):
        err = error(
            self.signal_test,
            self.predict(self.par_test),
            self.frequencies,
            relative=relative,
            flow=flow,
            fhigh=fhigh,
        )
        return err


class AutoEncoder(tf.keras.models.Model):
    def __init__(
        self,
        signal_train=signal_train,
        enc_hidden_dims=[],
        dec_hidden_dims=[],
        latent_dim=9,
        activation_func="relu",
    ):
        super().__init__()
        self.encoder = gen_model(
            signal_train.shape[-1],
            enc_hidden_dims,
            latent_dim,
            activation_func,
            name="encoder",
        )

        self.decoder = gen_model(
            None,
            dec_hidden_dims,
            signal_train.shape[-1],
            activation_func,
            name="decoder",
        )

    def call(self, x):
        return self.decoder(self.encoder(x))


# default parameters
latent_dim = 9
enc_hidden_dims = [352]
dec_hidden_dims = [32, 352]
em_hidden_dims = [352, 352, 352, 224]


class AutoEncoderEmulator:
    def __init__(
        self,
        par_train=par_train,
        par_val=par_val,
        par_test=par_test,
        signal_train=signal_train,
        signal_val=signal_val,
        signal_test=signal_test,
        latent_dim=latent_dim,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
        em_hidden_dims=em_hidden_dims,
        activation_func="relu",
        redshifts=redshifts,
        frequencies=None,
    ):

        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [
            "fstar",
            "Vc",
            "fx",
            "tau",
            "alpha",
            "nu_min",
            "Rmfp",
        ]

        if frequencies is None:
            if redshifts is not None:
                frequencies = redshift2freq(redshifts)
        elif redshifts is None:
            redshifts = freq2redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

        autoencoder = AutoEncoder(
            self.signal_train,
            enc_hidden_dims,
            dec_hidden_dims,
            latent_dim,
            activation_func,
        )

        # build autoencoder by calling it on a batch of data
        _ = autoencoder(pp.preproc(self.signal_test, self.signal_train))
        self.autoencoder = autoencoder

        self.emulator = gen_model(
            self.par_train.shape[-1],
            em_hidden_dims,
            latent_dim,
            activation_func,
            name="ae_emualtor",
        )

    AE_PATH = PATH + "models/autoencoder_based_emulator/"

    def load_model(
        self,
        emulator_path=AE_PATH + "ae_emulator.h5",
        encoder_path=AE_PATH + "encoder.h5",
        decoder_path=AE_PATH + "decoder.h5",
    ):
        self.emulator = tf.keras.models.load_model(emulator_path)
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
        autoencoder = AutoEncoder(signal_train=self.signal_train)
        autoencoder.encoder = encoder
        autoencoder.decoder = decoder
        # build autoencoder by calling it on a batch of data
        _ = autoencoder(pp.preproc(self.signal_test, self.signal_train))
        self.autoencoder = autoencoder

    def train(self, epochs, ae_callbacks=[], em_callbacks=[], verbose="tqdm"):

        y_train = pp.preproc(self.signal_train, self.signal_train)
        y_val = pp.preproc(self.signal_val, self.signal_train)

        if verbose == "tqdm":
            ae_callbacks.append(TqdmCallback())
            em_callbacks.append(TqdmCallback())
            verbose = 0
        hist = self.autoencoder.fit(
            x=y_train,
            y=y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=(y_val, y_val),
            callbacks=ae_callbacks,
            verbose=verbose,
        )
        ae_loss = hist.history["loss"]
        ae_val_loss = hist.history["val_loss"]

        X_train = pp.par_transform(self.par_train, self.par_train)
        X_val = pp.par_transform(self.par_val, self.par_train)
        y_train = self.autoencoder.encoder.predict(y_train)
        y_val = self.autoencoder.encoder.predict(y_val)

        hist = self.emulator.fit(
            x=X_train,
            y=y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=em_callbacks,
            verbose=verbose,
        )
        loss = hist.history["loss"]
        val_loss = hist.history["val_loss"]

        return ae_loss, ae_val_loss, loss, val_loss

    def predict(self, params):
        transformed_params = pp.par_transform(params, self.par_train)
        em_pred = self.emulator.predict(transformed_params)
        decoded = self.autoencoder.decoder.predict(em_pred)
        pred = pp.unpreproc(decoded, self.signal_train)
        if pred.shape[0] == 1:
            return pred[0, :]
        else:
            return pred

    def test_error(
        self, use_autoencoder=False, relative=True, flow=None, fhigh=None
    ):
        if use_autoencoder:
            pred = self.autoencoder(self.par_test)
        else:
            pred = self.predict(self.par_test)
        err = error(
            self.signal_test,
            pred,
            self.frequencies,
            relative=relative,
            flow=flow,
            fhigh=fhigh,
        )
        return err
