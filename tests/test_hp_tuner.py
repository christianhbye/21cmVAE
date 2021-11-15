import numpy as np
import os
from tensorflow.keras.models import load_model
from VeryAccurateEmulator import hyperparameter_tuner as hpt
from VeryAccurateEmulator.training_tools import em_loss_fcn


def test_gen_hp():
    """
    Test that function hpt.generate_hp() returns hps in desired range and of desired type
    :return: None
    """
    min_vals, step_sizes, max_steps = np.random.randint(1000, size=(3, 100))
    assert all(isinstance(mv, (int, np.integer)) for mv in min_vals)
    hpslist = np.empty(100)
    for i in range(100):
        min_val, step_size, max_step = min_vals[i], step_sizes[i], max_steps[i]
        hp = hpt.generate_hp(min_val, step_size, max_step)
        hpslist[i] = hp
        assert isinstance(hp, (int, np.integer))
        max_val = min_val + max_step * step_size
        assert min_val <= hp <= max_val
    assert not np.allclose(hpslist[0]*np.ones_like(hpslist), hpslist)

def test_gen_layers():
    """
    Test that function hpt.generate_layers_hps() returns the expected result
    :return: None
    """
    x = np.random.randint(1, 1000, size=(3, 2))
    no_hidden_layers, hidden_dims = x.T
    hdims_arr = hpt.generate_layer_hps(no_hidden_layers, hidden_dims)
    assert len(np.shape(hdims_arr)) == 1
    max_number_of_layers = no_hidden_layers[0] + no_hidden_layers[1] * no_hidden_layers[2]
    assert np.shape(hdims_arr)[0] <= max_number_of_layers
    max_dim = hidden_dims[0] + hidden_dims[1] * hidden_dims[2]
    assert 0 < all(hdims_arr) <= max_dim


def test_gen_0layers():
    """
    Test that we get the expected output if we try to generate 0 layers
    """
    hidden_dims = np.random.randint(1, 1000, size=3)  # can be arbitrary in this test
    hdims_arr = hpt.generate_layer_hps((0, 0, 0), hidden_dims)
    assert np.shape(hdims_arr) == (0,)

def test_run_tuner():
    files_before = os.listdir()
    nfiles = len(files_before)  # ensures all files created here gets deleted
    max_trials = 6
    epochs = 5
    x = np.random.randint(1, 50, size=(3, 2))
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
    files_now = os.listdir()
    if len(files_now) > nfiles:
        print("Not all files got deleted")
        print(np.diff(files_before, files_now))
        raise AssertionError
