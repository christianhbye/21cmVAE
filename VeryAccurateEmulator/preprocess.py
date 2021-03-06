import numpy as np


def preproc(signal: np.ndarray, signal_train: np.ndarray) -> np.ndarray:
    """
    Preprocess all the signals in a dataset.

    Parameters
    ----------
    signal : np.ndarray
        Array of signals to preprocess.
    signal_train : np.ndarray
        Array of the training set signals.

    Returns
    -------
    proc_signal : np.ndarray
        The preprocessed signals.

    """
    proc_signal = signal.copy()
    proc_signal -= np.mean(signal_train, axis=0)  # subtract mean
    proc_signal /= np.std(signal_train)  # divide by standard deviation
    return proc_signal


def unpreproc(signal: np.ndarray, signal_train: np.ndarray) -> np.ndarray:
    """
    Inverse of preproc function.

    Parameters
    ----------
    signal : np.ndarray
        Array of preprocesed signals to unpreprocess.
    signal_train : np.ndarray
        Array of the training set signals used for preprocessing.

    Returns
    --------
    unproc_signal : np.ndarray
        Array of the unpreprocessed signals.

    """
    unproc_signal = signal * np.std(signal_train)
    unproc_signal += np.mean(signal_train, axis=0)
    return unproc_signal


def par_transform(
    parameters: np.ndarray, params_train: np.ndarray
) -> np.ndarray:
    """
    Preprocess a set of parameters the same way that the training
    set parameters are processed:
    that is, take log of first three columns and apply a linear map that makes
    all the training set parameters be in the range [-1, 1]. Note that this
    map will not send other sets of parameters to [-1, 1].

    Parameters
    ----------
    parameters : np.ndarray
        Array of parameters.
    params_train : np.ndarray
        The parameters used to train the model.

    Returns
    -------
    newparams : np.ndarray
        The processed parameters.
    """
    if len(np.shape(parameters)) == 1:
        parameters = np.expand_dims(parameters, axis=0)
    # first copy the parameters and take log of first three
    cols12 = parameters[:, :2].copy()  # fstar and Vc
    fx = parameters[:, 2].copy()  # fx
    fx[fx == 0] = 10 ** (-6)  # to avoid -inf in cases where fx == 0
    newcols12 = np.log10(cols12)  # log of fstar and Vc
    newfx = np.log10(fx)  # log of fx

    # initialize arrays with processed parameters:
    newparams = np.empty(parameters.shape)
    newparams[:, :2] = newcols12  # copy the log of fstar and Vc
    newparams[:, 2] = newfx  # the log of fx
    newparams[:, 3:] = parameters[
        :, 3:
    ].copy()  # copy the remaining parameters

    # do the same for the training params
    cols12_tr = params_train[:, :2].copy()  # fstar and Vc
    fx_tr = params_train[:, 2].copy()  # fx
    fx_tr[fx_tr == 0] = 10 ** (-6)  # to avoid -inf in cases where fx == 0
    newcols12_tr = np.log10(cols12_tr)  # log of fstar and Vc
    newfx_tr = np.log10(fx_tr)  # log of fx
    newparams_tr = np.empty(params_train.shape)
    newparams_tr[:, :2] = newcols12_tr  # copy the log of fstar and Vc
    newparams_tr[:, 2] = newfx_tr  # the log of fx
    newparams_tr[:, 3:] = params_train[:, 3:].copy()  # remaining parameters

    # get the max and min values of each parameter in the training set
    maximum = np.max(newparams_tr, axis=0)
    minimum = np.min(newparams_tr, axis=0)

    # subtract min, divide by (max-min), multiply by 2 and subtract 1 to get
    # parameters in the range [-1, 1] for the case of the training set
    newparams -= minimum  # subtract min to get the range [0, max-min]
    newparams /= maximum - minimum  # divide by (max-min) to get [0, 1]
    newparams *= 2
    newparams -= 1  # multiply by 2, subtract 1 to get [-1, 1]

    return newparams
