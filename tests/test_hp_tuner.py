import numpy as np
from VeryAccurateEmulator import hyperparameter_tuner as hpt


def test_gen_hp():
    """
    Test that function hpt.generate_hp() returns hps in desired range and of desired type
    :return: None
    """
    min_vals, step_size, max_step = np.random.randint(1000, size=(1000, 3))
    hps = hpt.generate_hp(min_vals, step_size, max_step)
    assert all(isinstance(h, int) for h in hps)
    max_vals = min_vals + max_step * step_size
    assert np.less_equal(hps, max_vals)
    assert np.less_equal(min_vals, hps)


def test_gen_layers():
    """
    Test that function hpt.generate_layers_hps() returns the expected result
    :return: None
    """
    x = np.random.randint(1, 1000, size=(3, 2))
    no_hidden_layers, hidden_dims = x[0], x[1]
    hdims_arr = hpt.generate_layers_hps(no_hidden_layers, hidden_dims)
    assert len(hdims_arr.shape) == 1
    max_number_of_layers = no_hidden_layers[0] + no_hidden_layers[1] * no_hidden_layers[2]
    assert hdims_arr.shape[0] <= max_number_of_layers
    max_dim = hidden_dims[0] + hidden_dims[1] * hidden_dims[2]
    assert 0 < hdims_arr.all() <= max_dim


def test_gen_0layers():
    """
    Test that we get the expected output if we try to generate 0 layers
    """
    hidden_dims = np.random.randint(1, 1000, size=3)  # can be arbitrary in this test
    hdims_arr = hpt.generate_layers_hps((0, 0, 0), *hidden_dims)
    assert hdims_arr.shape == (0,)

def test_run_tuner():

