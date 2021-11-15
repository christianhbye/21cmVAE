import numpy as np
import os
from tensorflow.keras.Models import load_model
from VeryAccurateEmulator import hyperparameter_tuner as hpt
from VeryAccurateEmulator.training_tools import em_loss_fcn


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
    no_hidden_layers, hidden_dims = x.T
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
    nfiles = len(os.listdir())  # ensures all files created here gets deleted
    max_trials = 10
    epochs = 30
    x = np.random.randint(1, 1000, size=(3, 2))
    no_hidden_layers, hidden_dims = x.T
    tuner = hpt.HyperParameterTuner(max_trials, epochs, *no_hidden_layers, *hidden_dims)
    tuner.save_sr()
    assert os.path.exists('search_range.txt'), "Search range should have been saved to file"
    tuner.run_all()
    results = np.load('tuner_results.npz')
    time = results['time']
    fivebest = results['five_best_trials']
    assert fivebest.shape == (5, 2)
    best_indices = fivebest[:, 0]
    for index in best_indices:
        fname = 'results_trial_' + str(index) + '_{:.0f}'.format(time)
        assert fname+'.txt' in os.listdir()
        assert fname+'_layer_hps.npy' in os.listdir()
    tr_loss = results['training_loss']
    val_loss = results['validation_loss']
    assert len(tr_loss) == len(val_loss)
    best_index = best_indices[np.argmin(fivebest[:, 1])]
    best_layer_hps = np.load('results_trial_' + str(best_index) + '_{:.0f}_layer_hps.npy'.format(time))
    signal_train = results['signal_train']
    tuned_em = load_model('best_emulator_{:.0f}.h5'.format(time),
                          custom_objects={'loss_function': em_loss_fcn(signal_train)})
    assert len(tuned_em.layers) == len(best_layer_hps)
    os.remove('search_range.txt')
    for index in best_indices:
        fname = 'results_trial_' + str(index) + '_{:.0f}'.format(time)
        os.remove(fname+'.txt')
        os.remove(fname+'_layer_hps.npy')
    os.remove('tuner_results.npz')
    os.remove('best_emulator_{:.0f}.h5'.format(time))
