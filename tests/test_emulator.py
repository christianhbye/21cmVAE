import h5py
import numpy as np
import tensorflow as tf
from VeryAccurateEmulator import emulator, __path__
import VeryAccurateEmulator.preprocess as pp

FILE = __path__[0] + "/dataset_21cmVAE.h5"
with h5py.File(FILE, "r") as hf:
    signal_train = hf["signal_train"][:]


def test_gen_model():
    in_dim = 7
    hidden_dims = [32, 64, 256]
    out_dim = 451
    model = emulator._gen_model(in_dim, hidden_dims, out_dim, "relu")
    all_dims = hidden_dims + [out_dim]
    assert len(model.layers) == len(all_dims)
    for i, layer in enumerate(model.layers):
        shape = layer.output_shape[-1]
        assert shape == all_dims[i]


def test_relative_mse_loss():
    loss_fcn = emulator.relative_mse_loss(signal_train)
    y_true = pp.preproc(signal_train[42], signal_train)
    amplitude = np.max(np.abs(signal_train[42]))
    amplitude_proc = pp.preproc(amplitude, signal_train)
    y_pred = pp.preproc(signal_train[123], signal_train)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rel_mse = np.mean(mse.numpy()) / amplitude_proc**2
    assert np.allclose(
        rel_mse,
        loss_fcn(
            np.expand_dims(y_true, axis=0), np.expand_dims(y_pred, axis=0)
        ).numpy()
    )


def test_z_nu():
    z = 30
    nu = emulator.redshift2freq(z)
    assert np.isclose(z, emulator.freq2redshift(nu))


def test_error():
    z = np.linspace(5, 50, 451)
    nu = emulator.redshift2freq(z)
    assert np.allclose(
        emulator.error(signal_train, signal_train), np.zeros(len(signal_train))
    )
