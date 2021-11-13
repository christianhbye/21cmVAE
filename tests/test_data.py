import os

HERE = os.path.realpath(__file__)[:-len('test_data.py')]
DATA_PATH = HERE + '../VeryAccurateEmulator/dataset_21cmVAE.h5'


def test_data_present():
    assert os.path.exists(DATA_PATH),\
        "The data file is not in 21cmVAE/VeryAccurateEmulator, download it from http://doi.org/10.5281/zenodo.5084114"
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
