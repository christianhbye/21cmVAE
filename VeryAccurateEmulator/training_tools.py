from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.losses import mse
import tensorflow as tf
import numpy as np
import VeryAccurateEmulator.preprocess as pp


def _relative_mse_loss(signal_train):
    """
    The square of the FoM in the paper, in units of standard deviation
    since the signals are preproccesed. We need a wrapper function to be able to
    pass signal_train as an input param.
    :param signal_train: numpy array of training signals
    :param y_true: array, the true signal concatenated with the amplitude
    :param y_pred: array, the predicted signal (by the emulator)
    :return: callable, the loss function
    """
    def loss_function(y_true, y_pred):
        # unpreproc signal to get the ampltiude
        signal = y_true + tf.convert_to_tensor(np.mean(signal_train, axis=0))
        # get amplitude in units of standard deviation of signals
        reduced_amp = tf.math.reduce_max(tf.abs(signal), axis=1, keepdims=True) 
        loss = mse(y_pred, y_true)  # loss is mean squared error ...
        loss /= K.square(reduced_amp)  # ... divided by square of amplitude
        return loss
    return loss_function


def em_loss_fcn(signal_train):
    """
    Wrapper function to be imported by other modules
    """
    return _relative_mse_loss(signal_train)


def _compile_model(model, lr, relative_mse=False, signal_train=None):
    """
    Function that compiles a model with a given learning rate and loss function
    and the Adam optimizer
    :param model: keras model object, model to compile
    :param lr: float, initial learning rate of model
    :param relative_mse: bool, the loss function is mse/amplitude if True (like
    the FOM in the paper) and mse if False (necessary for the autoencoder-based
    emulator). In general, you'll want True unless the model is the autoencoder-
    based emulator.
    :param signal_train: numpy array of training signals (needed to get the
    amplitudes), needed when relative_mse=True
    :return: None
    """
    if relative_mse and signal_train is None:
        raise KeyError(
                "relative_mse=True but signal_train=None so amplitudes cannot" \
                        "be computed!"
                        )
    optimizer = optimizers.Adam(learning_rate=lr)
    if relative_mse:
        loss_fcn = _relative_mse_loss(signal_train)
    else:
        loss_fcn = mse
    model.compile(optimizer=optimizer, loss=loss_fcn)
    print('Model Compiled: ' + model.name)


def train_emulator(
        emulator,
        signal_train,
        signal_val,
        par_train,
        par_val,
        epochs,
        initial_lr,
        lr_factor,
        lr_patience,
        lr_min_delta,
        min_lr,
        es_delta,
        es_patience
        ):
    """
    Trains the emulator.
    :param emulator: keras model object, the uncompiled emulator
    :param signal_train: numpy array with global signals in training set
    :param signal_val: numpy array with global signals in validation set
    :param par_train: numpy array with parameters corresponding to signals in
    signal_train
    :param par_val: numpy array with parameters corresponding to signals in
    signal_val
    :param epochs: int, number of epcohs to train model for
    :param initial_lr: float, initial learning rate of emulator
    :param lr_factor: float, factor to multiply learning rate by in LR scheduler
    :param lr_patience: int, number of epochs to wait before reducing learning
    rate
    :param lr_min_delta: float, minimum reduction in validation loss before
    reducing learning rate
    :param min_lr: float, minimum allowed learning rate
    :param es_delta: float, like lr_min_delta but for early stopping
    :param es_patience: int, like lr_patience but for early stopping
    
    See https://keras.io/api/callbacks/reduce_lr_on_plateau/ and
    https://keras.io/api/callbacks/early_stopping/ for more information about 
    the LR scheduler and the Early Stopping callback.
    
    :return: tuple of lists with the form (training loss, validation loss),
    giving the loss at each epoch
    """
    _compile_model(
            emulator,
            initial_lr,
            relative_mse=True,
            signal_train=signal_train
            )  # compile the emulator with the given LR
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=lr_patience,
            min_delta=lr_min_delta,
            min_lr=min_lr
            )  # callback for the LR scheduler
    early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=es_delta,
            patience=es_patience,
            restore_best_weights=True
            )  # early stopping
    y_train = pp.preproc(signal_train, signal_train)  # the target variables
    y_val = pp.preproc(signal_val, signal_train)  # validation target
    X_train = pp.par_transform(par_train, par_train) # input train variables
    X_val = pp.par_transform(par_val, par_train)  # input validation variables
    validation_set = (X_val, y_val)
    hist = emulator.fit(
            x=X_train,
            y=y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=validation_set,
            validation_batch_size=256, 
            callbacks=[reduce_lr, early_stop]
            )  # train emulator
    loss, val_loss = hist.history['loss'], hist.history['val_loss']
    return loss, val_loss


def _train_autoencoder(
        autoencoder,
        signal_train,
        signal_val,
        epochs,
        initial_lr,
        lr_factor,
        lr_patience,
        lr_min_delta,
        min_lr,
        es_delta,
        es_patience
        ):
    """
    Trains the autoencoder.
    :param autoencoder: keras model object, the uncompiled autoencoder
    :param signal_train: numpy array with global signals in training set
    :param signal_val: numpy array with global signals in validation set
    :param epochs: int, number of epcohs to train model for
    :param initial_lr: float, initial learning rate of emulator
    :param lr_factor: float, factor to multiply learning rate by in LR scheduler
    :param lr_patience: int, number of epochs to wait before reducing LR
    :param lr_min_delta: float, minimum reduction in validation loss before
    reducing learning rate
    :param min_lr: float, minimum allowed learning rate
    :param es_delta: float, like lr_min_delta but for early stopping
    :param es_patience: int, like lr_patience but for early stopping
    
    See https://keras.io/api/callbacks/reduce_lr_on_plateau/ and
    https://keras.io/api/callbacks/early_stopping/ for more information about
    the LR scheduler and the Early Stopping callback.
    
    :return: tuple of lists with the form (training loss, validation loss),
    giving the loss at each epoch
    """
    _compile_model(
            autoencoder,
            initial_lr,
            relative_mse=True,
            signal_train=signal_train
            )  # compiles the autoencoder with the given LR
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=lr_patience,
            min_delta=lr_min_delta,
            min_lr=min_lr
            )  # LR schedule callback
    early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=es_delta,
            patience=es_patience,
            restore_best_weights=True
            )  # early stopping
    y_train = pp.preproc(signal_train, signal_train)
    y_val = pp.preproc(signal_val, signal_train)
    validation_set = (y_val, y_val)
    hist = autoencoder.fit(
            x=y_train,
            y=y_train,
            batch_size=256,
            epochs=epochs,
            callbacks=[reduce_lr, early_stop],
            validation_data=validation_set,
            validation_batch_size=256
            )  # train autoencoder
    loss, val_loss = hist.history['loss'], hist.history['val_loss']
    return loss, val_loss


def train_ae_based_emulator(
        emulator,
        encoder,
        signal_train,
        signal_val,
        par_train,
        par_val,
        epochs,
        initial_lr,
        lr_factor,
        lr_patience,
        lr_min_delta,
        min_lr,
        es_delta,
        es_patience
        ):
    """
    Trains the autoencoder-based emulator (described in Appendix A).
    :param emulator: keras model object, the uncompiled emulator
    :param encoder: keras model object, the encoder part of the autoencoder
    :param signal_train: numpy array with global signals in training set
    :param signal_val: numpy array with global signals in validation set
    :param par_train: numpy array with parameters corresponding to signals in
    signal_train
    :param par_val: numpy array with parameters corresponding to signals in
    signal_val
    :param epochs: int, number of epcohs to train model for
    :param initial_lr: float, initial learning rate of emulator
    :param lr_factor: float, factor to multiply learning rate by in LR scheduler
    :param lr_patience: int, number of epochs to wait before reducing LR
    :param lr_min_delta: float, minimum reduction in validation loss before
    reducing learning rate
    :param min_lr: float, minimum allowed learning rate
    :param es_delta: float, like lr_min_delta but for early stopping
    :param es_patience: int, like lr_patience but for early stopping
    
    See https://keras.io/api/callbacks/reduce_lr_on_plateau/ and
    https://keras.io/api/callbacks/early_stopping/ for more information about
    the LR scheduler and the Early Stopping callback.
    
    :return: tuple of lists with the form (training loss, validation loss),
    giving the loss at each epoch
    """
     # compile the emulator with the given LR
    _compile_model(emulator, initial_lr, relative_mse=False, signal_train=None) 
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=lr_patience,
            min_delta=lr_min_delta,
            min_lr=min_lr
            )  # LR schedule callback
    early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=es_delta,
            patience=es_patience,
            restore_best_weights=True
            )  # early stopping
    y_train = encoder.predict(pp.preproc(signal_train, signal_train))  
    y_val = encoder.predict(pp.preproc(signal_val, signal_train))
    X_train = pp.par_transform(par_train, par_train)
    X_val = pp.par_transform(par_val, par_train)
    validation_set = (X_val, y_val)
    hist = emulator.fit(
            x=X_train,
            y=y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=validation_set,
            validation_batch_size=256,
            callbacks=[reduce_lr, early_stop]
            )  # train emulator
    loss, val_loss = hist.history['loss'], hist.history['val_loss']
    return loss, val_loss


def train_ae_emulator(
        autoencoder,
        encoder,
        emulator,
        signal_train,
        signal_val,
        par_train,
        par_val,
        epochs,
        ae_lr,
        em_lr,
        ae_lr_factor,
        ae_lr_patience,
        ae_lr_min_delta,
        ae_min_lr,
        ae_es_delta,
        ae_es_patience,
        em_lr_factor,
        em_lr_patience,
        em_lr_min_delta,
        em_min_lr,
        em_es_delta,
        em_es_patience
        ):
    """
    Trains the auteoncoder and the emulator based on it.
    :param autoencoder: keras model object, the uncompiled autoencoder
    :param encoder: keras model object, the encoder part of the autoencoder
    :param emulator: keras model object, the uncompiled emulator
    :param signal_train: numpy array with global signals in training set
    :param signal_val: numpy array with global signals in validation set
    :param par_train: numpy array with parameters corresponding to signals in
    signal_train
    :param par_val: numpy array with parameters corresponding to signals in
    signal_val
    :param epochs: int, number of epcohs to train model for
    :param ae_lr: float, initial learning rate of autoencoder
    :param em_lr: float, initial learning rate of emulator
    :param ae_lr_factor: float, factor to multiply learning rate by in LR
    scheduler of the autoencoder
    :param ae_lr_patience: int, number of epochs to wait before reducing
    learning rate of the autoencoder
    :param ae_lr_min_delta: float, minimum reduction in validation loss before
    reducing learning rate of the autoencoder
    :param ae_min_lr: float, minimum allowed learning rate of the autoencoder
    :param ae_es_delta: float, like lr_min_delta but for early stopping of the
    autoencoder training
    :param ae_es_patience: int, like lr_patience but for early stopping of the
    autoencoder training
    :param em_lr_factor: float, factor to multiply learning rate by in LR
    scheduler of the emulator
    :param em_lr_patience: int, number of epochs to wait before reducing
    learning rate of the emulator
    :param em_lr_min_delta: float, minimum reduction in validation loss before
    reducing learning rate of the emulator
    :param em_min_lr: float, minimum allowed learning rate of the emulator
    :param em_es_delta: float, like lr_min_delta but for early stopping of the
    emualtor training
    :param em_es_patience: int, like lr_patience but for early stopping of the
    emulator training
    
    See https://keras.io/api/callbacks/reduce_lr_on_plateau/ and
    https://keras.io/api/callbacks/early_stopping/ for more information about
    the LR scheduler and the Early Stopping callback.
    
    :return: tuple of lists with the form (autoencoder training loss,
    autoencoder validation loss, emulator training loss, emulator validation
    loss), giving the loss at each epoch
    """
    print('Train Autoencoder')
    ae_loss, ae_val_loss = _train_autoencoder(
            autoencoder,
            signal_train,
            signal_val,
            epochs,
            ae_lr,
            ae_lr_factor,
            ae_lr_patience,
            ae_lr_min_delta,
            ae_min_lr,
            ae_es_delta,
            ae_es_patience
            )
    if len(ae_loss) < epochs:
        print('Early Stopping')
    print('Train Emulator')
    em_loss, em_val_loss = train_ae_based_emulator(
            emulator,
            encoder,
            signal_train,
            signal_val,
            par_train,
            par_val,
            epochs,
            em_lr,
            em_lr_factor,
            em_lr_patience,
            em_lr_min_delta,
            em_min_lr,
            em_es_delta,
            em_es_patience
            )
    if len(em_loss) < epochs:
        print('Early Stopping')
    return ae_loss, ae_val_loss, em_loss, em_val_loss
