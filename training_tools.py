from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np


def compile_VAE(vae, vae_lr, sampling_dim, hps, annealing_param, reconstruction_loss, kl_loss):
    # get hyperparameters:
    gamma = hps['gamma']
    beta = hps['beta']
    vae_loss_fcn = (sampling_dim * reconstruction_loss + beta * annealing_param * kl_loss) / (sampling_dim * gamma)
    vae.add_loss(vae_loss_fcn)  # add the loss function to the model
    vae_optimizer = optimizers.Adam(learning_rate=vae_lr)
    vae.compile(optimizer=vae_optimizer)  # compile the model with the optimizer


def em_loss(y_true, y_pred):
    """
    The emulator loss function, that is, the square of the FoM in the paper, in units of standard deviation
    since the signals are preproccesed
    :param y_true: array, the true signal concatenated with the amplitude
    :param y_pred: array, the predicted signal (by the emulator)
    :return: the loss
    """
    signal = y_true[:, 0:-1]  # the signal
    amplitude = y_true[:, -1]/tf.math.reduce_std(signal_train)  # amplitude
    loss = mse(y_pred, signal)  # loss is mean squared error ...
    loss /= K.square(amplitude)  # ... divided by square of amplitude
    return loss


def compile_emulator(emulator, em_lr):
    em_optimizer = optimizers.Adam(learning_rate=em_lr)
    emulator.compile(optimizer=em_optimizer, loss=em_loss)


# KL annealing
hp_lambda = K.variable(1)  # initialize the annealing parameter


def anneal_schedule(epoch):
    """
    Annealing schedule, linear increase from 0 to 1 over 10 epochs
    :param epoch: Current training epoch
    :return: Value of the annealing parameter
    """
    return min(epoch * 0.1, 1.)


class AnnealingCallback(callbacks.Callback):
    def __init__(self, schedule, variable):
        super(AnnealingCallback, self).__init__()
        self.schedule = schedule
        self.variable = variable

    def on_epoch_begin(self, epoch, logs={}):
        value = self.schedule(epoch)
        assert type(value) == float, ('The output of the "schedule" function should be float.')
        K.set_value(self.variable, value)


# instantiate the AnnealingCallback class
kl_annealing = AnnealingCallback(anneal_schedule, hp_lambda)


def plateau_check(model, patience, max_factor, vae_loss_val, em_loss_val):
    """
    Helper function for reduce_lr(). Checks if the validation loss has stopped
    decreasing as defined by the parameters.
    :param model: string, 'vae' or 'emulator'
    :param patience: max number of epochs loss has not decreased
    :param max_factor: max_factor * current loss is the max acceptable loss
    :param vae_loss_val: list of vae validation losses
    :param em_loss_val: list of emulator validation losses
    :return: boolean, True (reduce LR) or False (don't reduce LR)
    """
    if model == "vae":
        loss_list = vae_loss_val
    elif model == "emulator":
        loss_list = em_loss_val
    else:
        print('Invalid input parameter "model". Must be "vae" or "emulator".')

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


def reduce_lr(model, factor, min_lr):
    """
    Manual implementation of https://keras.io/api/callbacks/reduce_lr_on_plateau/.
    :param model: string, 'vae' or 'emulator'
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


def early_stop(patience, max_factor, vae_loss_val, em_loss_val):
    """
    Manual implementation of https://keras.io/api/callbacks/early_stopping/.
    :param patience: max number of epochs loss has not decreased
    :param max_factor: max_factor * current loss is the max acceptable loss
    :param vae_loss_val: list of vae validation losses
    :param em_loss_val: list of emulator validation losses
    :return: boolean, True (keep going) or False (stop early)
    """
    if not len(vae_loss_val) > patience:  # there is not enough training to compare
        return True

    vae_max_loss = vae_loss_val[-(1 + patience)] * max_factor  # the max acceptable loss
    em_max_loss = em_loss_val[-(1 + patience)] * max_factor  # the max acceptable loss

    count = 0
    while count < patience:
        if vae_loss_val[-(1 + count)] > vae_max_loss and em_loss_val[-(1 + count)] > em_max_loss:
            count += 1
            continue
        else:
            break
    if count == patience:  # the last [patience] losses are all too large: stop training
        print("Early stopping!")
        return False  # keep_going = False, i.e. stop early
    else:
        return True  # keep_going = True, continue


def create_batch(x_train, y_train, amplitudes):
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


def train_models(vae, emulator, reconstruction_loss, kl_loss, em_lr, vae_lr, hps, dataset, val_dataset, epochs,
                 vae_lr_factor, em_lr_factor, vae_min_lr, em_min_lr, vae_lr_patience, em_lr_patience, lr_max_factor,
                 es_patience, es_max_factor):
    """
    Function that train the models simultaneously
    :param vae: Keras model object, the VAE
    :param emulator: Keras model object, the emulator
    :param dataset: batches from training dataset
    :param val_dataset: batches from validation dataset
    :param epochs: max number of epochs to train for, early stopping may stop it before
    :param vae_lr_factor: factor * old LR (learning rate) is the new LR for the VAE
    :param em_lr_factor: factor * old LR (learning rate) is the new LR for the emulator
    :param vae_min_lr: minimum allowed LR for VAE
    :param em_min_lr: minimum allowed LR for emulator
    :param vae_lr_patience: max number of epochs loss has not decreased for the VAE before reducing LR
    :param em_lr_patience: max number of epochs loss has not decreased for the emulator before reducing LR
    :param lr_max_factor: max_factor * current loss is the max acceptable loss, a larger loss means that the counter
    is added to, when it reaches the 'patience', the LR is reduced
    :param es_patience: max number of epochs loss has not decreased before early stopping
    :param es_max_factor: max_factor * current loss is the max acceptable loss, a larger loss for either the VAE or the
    emulator means that the counter is added to, when it reaches the 'patience', early stopping is applied
    :return tuple, four lists of losses as they change with epoch for the VAE (training loss and validation loss)
    and emulator (training and validation) in that order
    """
    # initialize lists of training losses and validation losses
    vae_loss = []
    vae_loss_val = []
    em_loss = []
    em_loss_val = []

    # Did the model loss plateau?
    plateau_vae = False
    plateau_em = False
    vae_reduced_lr = 0  # epochs since last time lr was reduced
    em_reduced_lr = 0  # epochs since last time lr was reduced

    @tf.function
    def run_train_step(batch):
        '''
        Function that trains the VAE and emulator for one batch. Returns the losses
        for that specific batch.
        '''
        params = batch[0]
        signal = batch[1]
        amp_raw = batch[2]  # amplitudes, raw because we need to reshape
        amplitudes = tf.expand_dims(amp_raw, axis=1)  # reshape amplitudes
        signal_amplitudes = tf.concat((signal, amplitudes), axis=1)  # both signal and amplitude
        with tf.GradientTape() as tape:
            vae_pred = vae(signal)  # apply vae to input signal
            vae_batch_loss = vae.losses  # get the loss
        # back-propagate losses for the VAE
        vae_gradients = tape.gradient(vae_batch_loss, vae.trainable_weights)
        vae.optimizer.apply_gradients(zip(vae_gradients, vae.trainable_weights))
        # same procedure for emulator
        with tf.GradientTape() as tape:
            em_pred = emulator(params)
            em_batch_loss = em_loss_fcn(signal_amplitudes, em_pred)
        em_gradients = tape.gradient(em_batch_loss, emulator.trainable_weights)
        emulator.optimizer.apply_gradients(zip(em_gradients, emulator.trainable_weights))
        return vae_batch_loss, em_batch_loss

    # the training loop
    for i in range(epochs):
        epoch = int(i + 1)
        print("\nEpoch {}/{}".format(epoch, epochs))

        # reduce lr if necessary
        if plateau_vae and vae_reduced_lr >= 5:
            reduce_lr(vae, vae_lr_factor, vae_min_lr)
            vae_reduced_lr = 0
        if plateau_em and em_reduced_lr >= 5:
            reduce_lr(emulator, em_lr_factor, em_min_lr)
            em_reduced_lr = 0

        vae_batch_losses = []
        val_vae_batch_losses = []
        em_batch_losses = []
        val_em_batch_losses = []

        # KL annealing, updates hp_lambda
        kl_annealing.on_epoch_begin(epoch)

        # compile the models
        sampling_dim = 0
        for batch in dataset:
            typical_signal = batch[1]
            sampling_dim = np.shape(typical_signal)[-1]
            break
        compile_VAE(vae, vae_lr, sampling_dim, hps, hp_lambda, reconstruction_loss, kl_loss)
        compile_emulator(emulator, em_lr)

        # loop through the batches and train the models on each batch
        for batch in dataset:
            vae_batch_loss, em_batch_loss = run_train_step(batch)
            vae_batch_losses.append(vae_batch_loss)  # append VAE train loss for this batch
            em_batch_losses.append(em_batch_loss)  # append emulator train loss for this batch

        # loop through the validation batches, we are not training on them but
        # just evaluating and tracking the performance
        for batch in val_dataset:
            param_val = batch[0]
            signal_val = batch[1]
            amp_val = tf.expand_dims(batch[2], axis=1)
            val_signal_amplitudes = tf.concat((signal_val, amp_val), axis=1)
            val_em_batch_loss = emulator.test_on_batch(param_val, val_signal_amplitudes)
            val_vae_batch_loss = vae.test_on_batch(signal_val, signal_val)
            val_vae_batch_losses.append(val_vae_batch_loss)
            val_em_batch_losses.append(val_em_batch_loss)

        vae_loss_epoch = K.mean(tf.convert_to_tensor(vae_batch_losses))  # average VAE train loss over this epoch
        em_loss_epoch = K.mean(tf.convert_to_tensor(em_batch_losses))  # average emulator train loss
        print('VAE train loss: {:.4f}'.format(vae_loss_epoch))
        print('Emulator train loss: {:.4f}'.format(em_loss_epoch))

        # in case a loss is NaN
        # this is unusal, but not a big deal, just restart the training
        # (otherwise the loss just stays NaN)
        if np.isnan(vae_loss_epoch) or np.isnan(em_loss_epoch):
            print("Loss is NaN, restart training")
            break

        # save each epoch loss to a list with all epochs
        vae_loss.append(vae_loss_epoch)
        em_loss.append(em_loss_epoch)

        vae_loss_epoch_val = np.mean(val_vae_batch_losses)  # average VAE train loss over this epoch
        em_loss_epoch_val = np.mean(val_em_batch_losses)  # average emulator train loss
        vae_loss_val.append(vae_loss_epoch_val)
        em_loss_val.append(em_loss_epoch_val)
        print('VAE val loss: {:.4f}'.format(vae_loss_epoch_val))
        print('Emulator val loss: {:.4f}'.format(em_loss_epoch_val))

        # save weights
        if epoch == 1:  # save first epoch
            vae.save('checkpoints/best_vae')
            emulator.save('checkpoints/best_em')
        elif em_loss_val[-1] < np.min(em_loss_val[:-1]):  # performance is better than prev epoch
            vae.save('checkpoints/best_vae')
            emulator.save('checkpoints/best_em')

        # early stopping?
        keep_going = early_stop(es_patience, es_max_factor, vae_loss_val, em_loss_val)
        if not keep_going:
            break

        # check if loss stopped decreasing
        plateau_vae = plateau_check("vae", vae_lr_patience, lr_max_factor, vae_loss_val, em_loss_val)
        plateau_em = plateau_check("emulator", em_lr_patience, lr_max_factor, vae_loss_val, em_loss_val)

        vae_reduced_lr += 1
        em_reduced_lr += 1

    return vae_loss, vae_loss_val, em_loss, em_loss_val


