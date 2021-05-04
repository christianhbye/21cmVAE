# KL annealing
hp_lambda = K.variable(1)  # initialize the annealing parameter


def anneal_schedule(epoch):
    """
    Annealing schedule, linear increase from 0 to 1 over 10 epochs
    :param epoch: Current training epoch
    :return: Value of the annealing parameter
    """
    return min(epoch * 0.1, 1.)


class AnealingCallback(callbacks.Callback):
    def __init__(self, schedule, variable):
        super(AneelingCallback, self).__init__()
        self.schedule = schedule
        self.variable = variable

    def on_epoch_begin(self, epoch, logs={}):
        value = self.schedule(epoch)
        assert type(value) == float, ('The output of the "schedule" function should be float.')
        K.set_value(self.variable, value)


# instantiate the AnealingCallback class
kl_annealing = AnealingCallback(anneal_schedule, hp_lambda)


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


# create minibatches
batch_size = 256


def create_batch(x_train, y_train, amplitudes):
    """
    Create minibatches.
    :param x_train: training/validation parameters
    :param y_train: training/validation signals
    :param amplitudes: amplitude of training signals / np.std(signal_train)
    :return: minibatches for training or validation
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, amplitudes)).shuffle(1000)
    # Combines consecutive elements of this dataset into batches.
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    # Creates a Dataset that prefetches elements from this dataset
    return dataset


