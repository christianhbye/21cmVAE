from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow as tf


def sampling(args):
    """
    Function that samples from the Gaussian distributions to the latent layer
    :param args: mean and log of variance of the Gaussians
    :return: sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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


def build_models(hps, layer_hps, vae_lr=0.01, em_lr=0.01, activation_func='relu'):
    """
    Function that build the two neural networks.
    :param hps: hyperparameters, a dictionary with values for the latent layer dimensionality, beta, and gamma
    :param layer_hps: the hyperparameters controlling the number of layers and their dimensionalities,
    packed into nested lists
    :param vae_lr: float, initial VAE learning rate (will be reduced by reduce_lr() during training)
    :param em_lr: float, initial emulator learning rate (will be reduced by reduce_lr() during training)
    :param activation_func: str, name of a keras recognized activation function or a tf.keras.activations instance
    (see https://keras.io/api/layers/activations/)
    :return: the VAE and the emulator as keras model objects
    """
    encoding_hidden_dims = layer_hps[0]  # the layers of the encoder
    vae_input = Input(shape=(signal_train.shape[1],))  # input layer for VAE
    # loop over the number of layers and build the encoder part of the VAE
    # with layers of the given dimensionalities
    for i, dim in enumerate(encoding_hidden_dims):
        if i == 0:
            input_layer = vae_input  # the first layer takes input from the input layer
        else:
            input_layer = x  # subsequent layers take input from the previous layer
        # using dense (fully connected) layers
        x = Dense(dim, activation=activation_func, name='encoder_hidden_layer_' + str(i))(input_layer)
    latent_dim = hps['latent_dim']  # dimensionality of latent layer
    z_mean = Dense(latent_dim, name='z_mean')(x)  # mean values for the Gaussians
    # in the latent layer
    z_log_var = Dense(latent_dim, name='z_log_var')(x)  # log variance of Gaussians
    # sample points from the Gaussians with given mean and variance
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # now, the same procedure for the decoder as for the encoder
    decoding_hidden_dims = layer_hps[1]
    decoder_layers = []  # list of decoder layers
    for i, dim in enumerate(decoding_hidden_dims):
        if i == 0:
            input_layer = z
        else:
            input_layer = x
        decoder_layer = Dense(dim, activation=activation_func, name='decoder_hidden_layer_' + str(i))
        decoder_layers.append(decoder_layer)  # add to decoder layer list
        x = decoder_layer(input_layer)

    decoder_output_layer = Dense(signal_train.shape[1])  # output of decoder
    decoder_layers.append(decoder_output_layer)
    decoded = decoder_output_layer(x)
    vae_output = decoded

    # create the VAE
    vae = Model(vae_input, vae_output, name='VAE')

    # the VAE loss function
    reconstruction_loss = mse(vae_input, vae_output)  # mean squared error
    # get the unpreprocessed signal to get the amplitude
    unproc_signal = vae_input + K.constant(np.mean(signal_train, axis=0) / np.std(signal_train))
    amplitude = K.max(K.abs(unproc_signal), axis=-1)
    reconstruction_loss /= K.square(amplitude)
    # KL-part of the loss:
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    orig_dim = signal_train.shape[-1]
    # get hyperparameters:
    gamma = hps['gamma']
    beta = hps['beta']
    vae_loss_fcn = (orig_dim * reconstruction_loss + beta * hp_lambda * kl_loss) / (orig_dim * gamma)
    vae.add_loss(vae_loss_fcn)  # add the loss function to the model
    vae_optimizer = optimizers.Adam(learning_rate=vae_lr)
    vae.compile(optimizer=vae_optimizer)  # compile the model with the optimizer

    # make the emulator in the same way as the encoder
    em_input_par = Input(shape=(X_train.shape[1],), name='em_input')
    em_hidden_dims = layer_hps[2]
    for i, dim in enumerate(em_hidden_dims):
        if i == 0:
            input_layer = em_input_par
        else:
            input_layer = x
        x = Dense(dim, activation=activation_func, name='em_hidden_layer_' + str(i))(input_layer)

    # the latent layer of the emulator
    autoencoder_par = Dense(latent_dim, name='em_autoencoder')(x)

    # the decoder layers are shared so we just get them from the list made above
    for i, decoder_layer in enumerate(decoder_layers):
        if i == 0:
            output_par_in = autoencoder_par
        else:
            output_par_in = output_par
        output_par = decoder_layer(output_par_in)

    # compile the emulator
    em_output = output_par
    emulator = Model(em_input_par, em_output, name='emulator')
    em_optimizer = optimizers.Adam(learning_rate=em_lr)
    emulator.compile(optimizer=em_optimizer, loss=em_loss)

    return vae, emulator
