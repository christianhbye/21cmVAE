import h5py
import tensorflow as tf
from tqdm.keras import TqdmCallback
import numpy as np

from VeryAccurateEmulator import __path__
import VeryAccurateEmulator.preprocess as pp

PATH = __path__[0] + "/"


def _gen_model(in_dim, hidden_dims, out_dim, activation_func, name=None):
    """
    Generate a new keras model.

    Parameters
    ----------
    in_dim : int or None
        The dimension of the input layer of the model. Should be None if the
        model is succeeding another model (e.g. a decoder in an autoencoder).
    hidden_dims : list of ints
        The dimension of the hidden layers of the model.
    out_dim : int
        The dimension of the output layer of the model.
    activation_func: str or instance of tf.keras.activations
        Activation function between hidden layers. Must be recognizable by
        keras.
    name : str or None
       Name of the model. Default : None.

    Returns
    -------
    model : tf.keras.Model
        The generated keras model.

    """
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
    The square of the FoM in the paper, in units of standard deviation as the
    signals are preproccesed.

    Parameters
    ----------
    signal_train : np.ndarray
        Training signals.
    
    Returns
    -------
    loss_function : callable
        The loss function.
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
    """
    Convert redshift to frequency.

    Parameters
    ----------
    z : float or np.ndarray
        The redshift or array of redshifts to convert.

    Returns
    -------
    nu : float or np.ndarray
        The corresponding frequency or array of frequencies in MHz.

    """
    nu = NU_0 / (1 + z)
    nu /= 1e6  # convert to MHz
    return nu


def freq2redshift(nu):
    """
    Convert frequency to redshfit.

    Parameters
    ----------
    nu : float or np.ndarray
        The frequency or array of frequencies in MHz to convert.

    Returns
    -------
    z : float or np.ndarray
        The corresponding redshift or array of redshifts.

    """
    nu *= 1e6  # to Hz
    z = NU_0 / nu - 1
    return z


def error(
    true_signal, pred_signal, relative=True, nu_arr=None, flow=None, fhigh=None
):
    """
    Compute the error (Eq. 1 in the paper) given the true and predicted
    signal(s).

    Parameters
    ----------
    true_signal : np.ndarray
        The true signal(s). An array of temperature for different redshifts
        or frequencies. For multiple signals must each row correspond to a
        signal.
    pred_signal : np.ndarray
        The predicted signal(s). Must have the same shape as true_signal.
    relative : bool
        Whether to compute the error in % relative to the signal amplitude
        (True) or in mK (False). Default : True.
    nu_arr : np.ndarray or None
        The frequency array corresponding to the signals. Needed for computing
        the error in different frequency bands. Default : None.
    flow : float or None
        The lower bound of the frequency band to compute the error in. Cannot
        be set without nu_arr. Default : None.
    fhigh : float or None
        The upper bound of the frequency bnd to compute the error in. Cannot
        be set without nu_arr. Default : None.

    Returns
    -------
    err : float or np.ndarray
        The computed errors. An array if multiple signals were input.

    Raises
    ------
    ValueError :
        If nu_arr is None and flow or fhigh are not None.
    
    """
    if (flow or fhigh) and nu_arr is None:
        raise ValueError(
            "No frequency array is given, cannot compute error in specified"
            "frequency band."
        )
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

        self.emulator = _gen_model(
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
            relative=relative,
            nu_arr=self.frequencies,
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
        self.encoder = _gen_model(
            signal_train.shape[-1],
            enc_hidden_dims,
            latent_dim,
            activation_func,
            name="encoder",
        )

        self.decoder = _gen_model(
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

        self.emulator = _gen_model(
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
            relative=relative,
            nu_arr=self.frequencies,
            flow=flow,
            fhigh=fhigh,
        )
        return err
