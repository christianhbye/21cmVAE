def preproc(signal, signal_train):
    """
    Function that preprocesses all the signals in train/validation/test set.
    :param signal: array of signals to preprocess
    :param signal_train: array of the training set signals
    :return: preprocessed signals
    """
    proc_signal = signal.copy()
    proc_signal -= np.mean(signal_train, axis=0)  # subtract mean
    proc_signal /= np.std(signal_train)  # divide by standard deviation
    return proc_signal


def unpreproc(signal, signal_train):
    """
    Inverse of preproc function
    :return: unpreprocessed signals
    """
    proc_signal = signal.copy()
    proc_signal *= np.std(signal_train)
    proc_signal += np.mean(signal_train, axis=0)
    return proc_signal
