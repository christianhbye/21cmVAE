import tensorflow as tf
from tqdm.keras import TqdmCallback
import numpy as np

import .preprocess as pp

def gen_model(in_dim, hidden_dims, out_dim, activation_func, name=None)
    input_layer = tf.keras.Input(shape=(in_dim,))
    layers = [input_layer]
    for dim in hidden_dims:
        layer = tf.keras.layers.Dense(dim, activation=activation_func)
        layers.append(layer)
    output_layer = tf.keras.layers.Dense(out_dim)
    layers.append(output_layer)
    model = tf.keras.Sequential(layers, name=name)
    return model

NU_0 = 1420405751.7667  # Hz

def redshift2freq(z):
    nu = NU_0 / (1 + z)
    nu /= 1e6  # convert to MHz
    return nu

def freq2redshift(nu):
    """
    Frequency in MHz.
    """
    nu *= 1e6  # to Hz
    z = NU_0/nu - 1
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
    
    pred_signal = predicted_signal[:, f]
    true_signal = true_signal[:, f]
    
    err = np.sqrt(np.mean((pred_signal - true_signal) ** 2, axis=1))
    if relative:  # give error as fraction of amplitude in the desired band
        err /= np.max(np.abs(true_signal), axis=1)
        err *= 100  # %
    return err

class DirectEmulator(tf.keras.models.Model):

    def __init__(
        self,
        par_train,
        par_val,
        par_test,
        signal_train,
        signal_val,
        signal_test,
        hidden_dims,
        activation_func="relu",
        redshifts=np.linspace(5, 50, num=451),
        frequencies=None,
        name="emulator"
    ):
        super().__init__()
        
        self.model = gen_model(
            par_train.shape[-1],
            hiddden_dims,
            signal_train.shape[-1],
            activation_func,
            name=name
        )

        self.X_train = pp.par_transform(par_train, par_train)
        self.X_val = pp.par_transform(par_val, par_train)
        self.y_train = pp.preproc(signal_train, signal_train)
        self.y_val = pp.preproc(signal_val, signal_train)

        if frequencies is None:
            if redshifts is not None:
                frequencies = redshift2freq(redshifts)
        elif redshifts is None:
            redshifts = freq2redshifts(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def train(
        self, epochs, callbacks=[], verbose="tqdm"
    ):
        if verbose == "tqdm":
            callbakcs.append(TqdmCallback())
            verbose = 0
        hist = self.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            validation_batch_size=256,
            callbacks=callbacks,
            verbose=verbose
        )
        loss = hist.history["loss"]
        val_loss = hist.history["val_loss"]
        return loss, val_loss

    def call(self, params):
        transformed_params = pp.par_transform(params, self.par_train)
        proc_pred = self.model.predict(transformed_params)
        pred = pp.unpreproc(proc_pred, self.signal_train)
        if pred.shape[0] == 1:
            return pred[0, :]
        else:
            return pred

    def test_error(self, relative=True, flow=None, fhigh=None):
        err = error(
            self.signal_test,
            self.call(self.par_test),
            self.frequencies,
            relative=relative,
            flow=flow,
            fhigh=fhigh
        )
        return err
        

