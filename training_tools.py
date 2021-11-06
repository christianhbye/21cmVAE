from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.losses import mse
import tensorflow as tf
import numpy as np
import preprocess as pp


def _relative_mse_loss(signal_train):
    """
    The square of the FoM in the paper, in units of standard deviation
    since the signals are preproccesed. We need a wrapper function to be able to pass signal_train as an input param.
    :param signal_train: numpy array of training signals
    :param y_true: array, the true signal concatenated with the amplitude
    :param y_pred: array, the predicted signal (by the emulator)
    :return: the loss
    """
    def loss_function(y_true, y_pred):
        signal = y_true[:, 0:-1]  # the signal
        amplitude = y_true[:, -1]/tf.math.reduce_std(signal_train)  # amplitude
        loss = mse(y_pred, signal)  # loss is mean squared error ...
        loss /= K.square(amplitude)  # ... divided by square of amplitude
        return loss
    return loss_function


def em_loss_fcn(signal_train):
    return _relative_mse_loss(signal_train)


def _compile_model(model, lr, relative_mse=False, signal_train=None):
    """
    Function that compiles a model with a given learning rate and loss function and the Adam optimizer
    :param model: keras model object, model to compile
    :param lr: float, initial learning rate of model
    :param relative_mse: bool, the loss function is mse/amplitude if True (like the FOM in the paper) and mse if False
    (necessary for the autoencoder based emulator). In general, you'll want True unless your model is the autoencoder-
    based emulator.
    :param signal_train: numpy array of training signals (needed to get the amplitudes), needed when relative_mse=True
    :return: None
    """
    if relative_mse and signal_train is None:
        raise KeyError("relative_mse=True but signal_train=None so amplitudes cannot be computed!")
    optimizer = optimizers.Adam(learning_rate=lr)
    if relative_mse:
        loss_fcn = _relative_mse_loss(signal_train)
    else:
        loss_fcn = mse
    model.compile(optimizer=optimizer, loss=loss_fcn)


def _plateau_check(patience, max_factor, em_loss_val):
    """
    Helper function for reduce_lr(). Checks if the validation loss has stopped
    decreasing as defined by the parameters.
    :param patience: max number of epochs loss has not decreased
    :param max_factor: max_factor * current loss is the max acceptable loss
    :param em_loss_val: list of emulator validation losses
    :return: boolean, True (reduce LR) or False (don't reduce LR)
    """
    loss_list = em_loss_val

    if not len(loss_list) > patience:  # there is not enough training to compare
        return False

    max_loss = max_factor * loss_list[-(1 + patience)]  # max acceptable loss

    count = 0
    while count < patience:
        if loss_list[-(1 + count)] > max_loss:
            count += 1
            continue
        else:
            break
    if count == patience:  # the last [patience] losses are all too large: reduce lr
        return True
    else:
        return False


def _reduce_lr(model, factor, min_lr):
    """
    Manual implementation of https://keras.io/api/callbacks/reduce_lr_on_plateau/.
    :param model: keras model object
    :param factor: factor * old LR is the new LR
    :param min_lr: minimum allowed LR
    :return: None
    """
    assert min_lr >= 0, "min_lr must be non-negative"
    old_lr = K.get_value(model.optimizer.learning_rate)  # get current LR
    if old_lr * factor <= min_lr < old_lr:
        new_lr = min_lr
        print('Reached min_lr, lr will not continue to decrease! {}_lr = {:.3e}'.format(model.name, new_lr))
    elif old_lr == min_lr:
        pass
    else:
        new_lr = old_lr * factor
        print('Reduce learning rate, {}_lr = {:.3e}'.format(model.name, new_lr))
        K.set_value(model.optimizer.learning_rate, new_lr)


def _early_stop(patience, max_factor, em_loss_val):
    """
    Manual implementation of https://keras.io/api/callbacks/early_stopping/.
    :param patience: max number of epochs loss has not decreased
    :param max_factor: max_factor * current loss is the max acceptable loss
    :param vae_loss_val: list of vae validation losses
    :param em_loss_val: list of emulator validation losses
    :return: boolean, True (keep going) or False (stop early)
    """
    if not len(em_loss_val) > patience:  # there is not enough training to compare
        return True

    em_max_loss = em_loss_val[-(1 + patience)] * max_factor  # the max acceptable loss

    count = 0
    while count < patience:
        if em_loss_val[-(1 + count)] > em_max_loss:
            count += 1
            continue
        else:
            break
    if count == patience:  # the last [patience] losses are all too large: stop training
        print("Early stopping!")
        return False  # keep_going = False, i.e. stop early
    else:
        return True  # keep_going = True, continue


def _create_batch(x_train, y_train, amplitudes):
    """
    Create minibatches.
    :param x_train: training/validation parameters
    :param y_train: training/validation signals
    :param amplitudes: amplitude of training signals / np.std(signal_train)
    :return: minibatches for training or validation
    """
    batch_size = 256
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, amplitudes)).shuffle(1000)
    # Combines consecutive elements of this dataset into batches.
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    # Creates a Dataset that prefetches elements from this dataset
    return dataset


def train_direct_emulator(direct_emulator, em_lr, signal_train, signal_val, par_train, par_val, epochs, em_lr_factor,
                          em_min_lr, em_lr_patience, lr_max_factor, es_patience, es_max_factor):
    """
    Function that trains the direct emulator
    :param direct_emulator: Keras model object, the direct emulator
    :param em_lr: float, initial direct emulator learning rate
    :param signal_train: numpy array of training signals
    :param dataset: batches from training dataset
    :param val_dataset: batches from validation dataset
    :param epochs: max number of epochs to train for, early stopping may stop it before
    :param em_lr_factor: factor * old LR (learning rate) is the new LR for the emulator
    :param em_min_lr: minimum allowed LR for emulator
    :param em_lr_patience: max number of epochs loss has not decreased for the emulator before reducing LR
    :param lr_max_factor: max_factor * current loss is the max acceptable loss, a larger loss means that the counter
    is added to, when it reaches the 'patience', the LR is reduced
    :param es_patience: max number of epochs loss has not decreased before early stopping
    :param es_max_factor: max_factor * current loss is the max acceptable loss, a larger loss for either the VAE or the
    emulator means that the counter is added to, when it reaches the 'patience', early stopping is applied
    :return tuple, two lists of losses as they change with epoch for the emulator (training and validation)
    """
    # initialize lists of training losses and validation losses
    em_loss = []
    em_loss_val = []

    # Did the model loss plateau?
    plateau_em = False
    em_reduced_lr = 0  # epochs since last time lr was reduced

    # compile the models
    _compile_model(direct_emulator, em_lr, True, signal_train)

    X_train, X_val = pp.par_transform(par_train, par_train), pp.par_transform(par_val, par_train)
    y_train, y_val = pp.preproc(signal_train, signal_train), pp.preproc(signal_val, signal_train)
    train_amplitudes = np.max(np.abs(signal_train), axis=-1)
    val_amplitudes = np.max(np.abs(signal_val), axis=-1)
    dataset = _create_batch(X_train, y_train, train_amplitudes)
    val_dataset = _create_batch(X_val, y_val, val_amplitudes)

    @tf.function
    def run_train_step(batch):
        """
        Function that trains the VAE and emulator for one batch. Returns the losses
        for that specific batch.
        """
        params = batch[0]
        signal = batch[1]
        amp_raw = batch[2]  # amplitudes, raw because we need to reshape
        amplitudes = tf.expand_dims(amp_raw, axis=1)  # reshape amplitudes
        signal_amplitudes = tf.concat((signal, amplitudes), axis=1)  # both signal and amplitude
        with tf.GradientTape() as tape:
            em_pred = direct_emulator(params)
            loss_function = _relative_mse_loss(signal_train)
            _em_batch_loss = loss_function(signal_amplitudes, em_pred)
        em_gradients = tape.gradient(_em_batch_loss, direct_emulator.trainable_weights)
        direct_emulator.optimizer.apply_gradients(zip(em_gradients, direct_emulator.trainable_weights))
        return _em_batch_loss

    # the training loop
    for i in range(epochs):
        epoch = int(i + 1)
        print("\nEpoch {}/{}".format(epoch, epochs))

        # reduce lr if necessary
        if plateau_em and em_reduced_lr >= 5:
            _reduce_lr(direct_emulator, em_lr_factor, em_min_lr)
            em_reduced_lr = 0

        em_batch_losses = []
        val_em_batch_losses = []

        # loop through the batches and train the models on each batch
        for batch in dataset:
            em_batch_loss = run_train_step(batch)
            em_batch_losses.append(em_batch_loss)  # append emulator train loss for this batch

        # loop through the validation batches, we are not training on them but
        # just evaluating and tracking the performance
        for batch in val_dataset:
            param_val = batch[0]
            signal_val = batch[1]
            amp_val = tf.expand_dims(batch[2], axis=1)
            val_signal_amplitudes = tf.concat((signal_val, amp_val), axis=1)
            val_em_batch_loss = direct_emulator.test_on_batch(param_val, val_signal_amplitudes)
            val_em_batch_losses.append(val_em_batch_loss)

        em_loss_epoch = K.mean(tf.convert_to_tensor(em_batch_losses))  # average emulator train loss
        print('Emulator train loss: {:.4f}'.format(em_loss_epoch))

        # in case a loss is NaN
        # this is unusal, but not a big deal, just restart the training
        # (otherwise the loss just stays NaN)
        if np.isnan(em_loss_epoch):
            print("Loss is NaN, restart training")
            break

        # save each epoch loss to a list with all epochs
        em_loss.append(em_loss_epoch)

        em_loss_epoch_val = np.mean(val_em_batch_losses)  # average emulator train loss
        em_loss_val.append(em_loss_epoch_val)
        print('Emulator val loss: {:.4f}'.format(em_loss_epoch_val))

        # save weights
        if epoch == 1:  # save first epoch
            direct_emulator.save('checkpoints/best_direct_em')
        elif em_loss_val[-1] < np.min(em_loss_val[:-1]):  # performance is better than prev epoch
            direct_emulator.save('checkpoints/best_direct_em')

        # early stopping?
        keep_going = _early_stop(es_patience, es_max_factor, em_loss_val)
        if not keep_going:
            break

        # check if loss stopped decreasing
        plateau_em = _plateau_check(em_lr_patience, lr_max_factor, em_loss_val)

        em_reduced_lr += 1

    return em_loss, em_loss_val


def _train_autoencoder(autoencoder, signal_train, signal_val, epochs, lr_factor, lr_patience, lr_min_delta, min_lr,
                       es_delta, es_patience):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_delta=lr_min_delta,
                                     min_lr=min_lr)
    early_stop = EarlyStopping(monitor="val_loss", min_delta=es_delta, patience=es_patience, restore_best_weights=True)
    y_train, y_val = pp.preproc(signal_train, signal_train), pp.preproc(signal_val, signal_train)
    validation_set = (y_val, y_val)
    hist = autoencoder.fit(x=y_train, y=y_train, batch_size=256, epochs=epochs, callbacks=[reduce_lr, early_stop],
                           validation_data=validation_set, validation_batch_size=256)
    loss, val_loss = hist.history['loss'], hist.history['val_loss']
    return loss, val_loss


def _train_emulator(emulator, encoder, signal_train, signal_val, par_train, par_val, epochs, lr_factor, lr_patience,
                    lr_min_delta, min_lr, es_delta, es_patience):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_delta=lr_min_delta,
                                  min_lr=min_lr)
    early_stop = EarlyStopping(monitor="val_loss", min_delta=es_delta, patience=es_patience, restore_best_weights=True)
    y_train = encoder.predict(pp.preproc(signal_train, signal_train))
    y_val = encoder.predict(pp.preproc(signal_val, signal_train))
    X_train, X_val = pp.par_transform(par_train, par_train), pp.par_transform(par_val, par_train)
    validation_set = (X_val, y_val)
    hist = emulator.fit(x=X_train, y=y_train, batch_size=256, epochs=epochs, validation_data=validation_set,
                        validation_batch_size=256, callbacks=[reduce_lr, early_stop])
    loss, val_loss = hist.history['loss'], hist.history['val_loss']
    return loss, val_loss


def train_ae_emulator(autoencoder, encoder, emulator, signal_train, signal_val, par_train, par_val, epochs,
                      ae_lr_factor, ae_lr_patience, ae_lr_min_delta, ae_min_lr, ae_es_delta, ae_es_patience,
                      em_lr_factor, em_lr_patience, em_lr_min_delta, em_min_lr, em_es_delta, em_es_patience):
    ae_loss, ae_val_loss = _train_autoencoder(autoencoder, signal_train, signal_val, epochs, ae_lr_factor,
                                              ae_lr_patience, ae_lr_min_delta, ae_min_lr, ae_es_delta, ae_es_patience)
    em_loss, em_val_loss = _train_emulator(emulator, encoder, signal_train, signal_val, par_train, par_val, epochs,
                                           em_lr_factor, em_lr_patience, em_lr_min_delta, em_min_lr, em_es_delta,
                                           em_es_patience)
    return ae_loss, ae_val_loss, em_loss, em_val_loss
