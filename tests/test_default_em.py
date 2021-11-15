import h5py
import numpy as np
import os
import VeryAccurateEmulator.VeryAccurateEmulator as VAE

HERE = os.path.realpath(__file__)[:-len('test_data.py')]
ROOT_DIR = HERE[:-len('tests/')]
DATA_PATH = ROOT_DIR + '../VeryAccurateEmulator/dataset_21cmVAE.h5'

default_em = VAE.VeryAccurateEmulator()


def test_data():
    assert os.path.exists(DATA_PATH),\
        "The data file is not at {}, download it from http://doi.org/10.5281/zenodo.5084114".format(DATA_PATH)
    with h5py.File(DATA_PATH, 'r') as hf:
        signal_train = hf['signal_train'][:]
        signal_val = hf['signal_val'][:]
        signal_test = hf['signal_test'][:]
        par_train = hf['par_train'][:]
        par_val = hf['par_val'][:]
        par_test = hf['par_test'][:]

    assert signal_train.shape == (24562, 451), "Training set has wrong shape, datafile is not loaded properly"
    assert signal_val.shape == (2730, 451), "Validation set has wrong shape, datafile is not loaded properly"
    assert signal_test.shape == (1704, 451), "Test set has wrong shape, datafile is not loaded properly"
    assert par_train.shape == (24562, 7), "Training set has wrong shape, datafile is not loaded properly"
    assert par_val.shape == (2730, 7), "Validation set has wrong shape, datafile is not loaded properly"
    assert par_test.shape == (1704, 7), "Test set has wrong shape, datafile is not loaded properly"

    # check that the default emulator uses the right data
    default_signal = np.concatenate((default_em.signal_train, default_em.signal_val, default_em.signal_test))
    default_par = np.concatenate((default_em.par_train, default_em.par_val, default_em.par_test))
    loaded_signal = np.concatenate((signal_train, signal_val, signal_test))
    loaded_par = np.concatenate((default_em.par_train, default_em.par_val, default_em.par_test))

    assert np.allclose(default_signal, loaded_signal), "The 21cmVAE default global signals don't match the datafile"
    assert np.allclose(default_par, loaded_par), "The 21cmVAE default params don't match the datafile"


def test_hps():
    assert type(default_em.hidden_dims) == list, "Emulator hidden dimensions must be list"
    assert all(isinstance(h, (int, np.integer)) for h in default_em.hidden_dims), "Hidden dimensions must be int"
