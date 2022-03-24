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
    y_true = tf.convert_to_tensor(pp.preproc(signal_train[:10], signal_train))
    y_pred = tf.convert_to_tensor(pp.preproc(signal_train[-10:], signal_train))
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    amplitude = tf.convert_to_tensor(
        np.max(np.abs(signal_train[:10] / np.std(signal_train)), axis=1)
    )
    rel_mse = mse / tf.keras.backend.square(amplitude)
    assert np.allclose(rel_mse.numpy(), loss_fcn(y_true, y_pred).numpy())


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


# direct emulator class
direm = emulator.DirectEmulator()
direm.load_model()


def test_predict():
    # some random parameters:
    pars = direm.par_test[0]
    pred = direm.predict(pars)
    true = direm.signal_test[0]
    assert pred.shape == true.shape
    # the emulator has a max error of 1.84 %
    assert np.sqrt(np.mean((pred - true) ** 2)) / np.max(np.abs(true)) < 0.02

    # vectorized call
    pars = direm.par_test[:10]
    pred_signals = direm.predict(pars)
    assert pred_signals[0].shape == pred.shape
    assert np.allclose(pred_signals[0], pred, atol=5e-5)
    assert pred_signals.shape == (10, true.shape[0])


def test_test_error():
    err = direm.test_error()
    assert err.shape == (direm.signal_test.shape[0],)
    # compare to table 1 in Bye et al. (2021)
    assert np.allclose(err.mean(), 0.34, atol=1e-2)
    assert np.allclose(np.median(err), 0.29, atol=1e-2)
    err_mk = direm.test_error(relative=False)
    assert np.allclose(err_mk.mean(), 0.54, atol=1e-2)
    assert np.allclose(np.median(err_mk), 0.50, atol=1e-2)


# autoencoder-based emulator class
ae_em = emulator.AutoEncoderEmulator()
ae_em.load_model()


def test_predict_ae():
    # some random parameters:
    pars = ae_em.par_test[0]
    pred = ae_em.predict(pars)
    true = ae_em.signal_test[0]
    assert pred.shape == true.shape
    # error should be less than 5 % in all cases
    assert np.sqrt(np.mean((pred - true) ** 2)) / np.max(np.abs(true)) < 0.05

    # vectorized call
    pars = ae_em.par_test[:10]
    pred_signals = ae_em.predict(pars)
    assert pred_signals[0].shape == pred.shape
    assert np.allclose(pred_signals[0], pred, atol=5e-5)
    assert pred_signals.shape == (10, true.shape[0])


def test_test_error():
    err = ae_em.test_error()
    assert err.shape == (direm.signal_test.shape[0],)
    # compare to appendix A in Bye et al. (2021)
    assert np.allclose(err.mean(), 0.39, atol=1e-2)
    assert np.allclose(np.median(err), 0.35, atol=1e-2)
    err_ae = ae_em.test_error(use_autoencoder=True)
    assert np.allclose(err_ae.mean(), 0.33, atol=1e-2)
    assert np.allclose(np.median(err_ae), 0.29, atol=1e-2)
