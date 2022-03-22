import h5py
import numpy as np
from VeryAccurateEmulator import __path__
import VeryAccurateEmulator.preprocess as pp

FILE = __path__[0] + "/dataset_21cmVAE.h5"
with h5py.File(FILE, "r") as hf:
    signal_train = hf["signal_train"][:]
    par_train = hf["par_train"][:]


def test_proc():
    proc_signal = pp.preproc(signal_train, signal_train)
    mean = np.mean(proc_signal, axis=0)
    assert np.allclose(mean, np.zeros_like(mean), atol=1e-3)

    unproc = pp.unpreproc(proc_signal, signal_train)
    assert np.allclose(unproc, signal_train, atol=5e-5)


def test_par_transform():
    transformed = pp.par_transform(par_train, par_train)
    max_par = transformed.max(axis=0)
    min_par = transformed.min(axis=0)
    assert np.allclose(max_par, np.ones_like(max_par))
    assert np.allclose(min_par, -1 * np.ones_like(min_par))
