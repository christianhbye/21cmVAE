import h5py
import pytest
from VeryAccurateEmulator import build_models as bm
from VeryAccurateEmulator import __path__

DATA_FILE = __path__[0] + "/dataset_21cmVAE.h5"

with h5py.File(DATA_FILE, "r") as hf:
    signal_train = hf["signal_train"][:]
    par_train = hf["par_train"][:]

def test_build_direct_emulator():
    layer_hps = [32, 128, 64]
    activation = "relu"
    em = bm.build_direct_emulator(
            layer_hps, signal_train, par_train, activation_func=activation
        )
    assert len(em.layers) == len(layer_hps) + 2  # + input/output layers
    all_dims = [par_train.shape[-1]] + layer_hps + [signal_train.shape[-1]]
    for i, layer in enumerate(em.layers):
        if i == 0:
            shape = layer.output_shape[0][-1]
        else:
            shape = layer.output_shape[-1]
        assert shape == all_dims[i], f"{i=}"
