import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow as tf


def build_direct_emulator(layer_hps, signal_train, par_train, activation_func='relu'):
    """
    Function that builds the emulator.
    :param layer_hps: list, the hyperparameter controlling the number of layers and their dimensionalities
    :param signal_train: numpy array of training signals
    :param par_train: numpy array of training parameters
    :param activation_func: str, name of a keras recognized activation function or a tf.keras.activations instance
    (see https://keras.io/api/layers/activations/)
    :return: the emulator as a keras model object
    """
    em_input_par = Input(shape=(par_train.shape[1],), name='em_input')
    em_hidden_dims = layer_hps
    for i, dim in enumerate(em_hidden_dims):
        if i == 0:
            input_layer = em_input_par
        else:
            input_layer = x
        x = Dense(dim, activation=activation_func, name='em_hidden_layer_' + str(i))(input_layer)
    output_par = Dense(signal_train.shape[1])(x)
    emulator = Model(em_input_par, output_par, name="Emulator")
    return emulator


def build_autoencoder(layer_hps, activation_func='relu'):
    encoding_hidden_dims = layer_hps[0]  # the layers of the encoder
    ae_input = Input(shape=(signal_train.shape[1],))  # input layer for autoencoder
    # loop over the number of layers and build the encoder with layers of the given dimensions
    for i, dim in enumerate(encoding_hidden_dims):
        if i == 0:
            input_layer = vae_input  # the first layer takes input from the input layer
        else:
            input_layer = x  # subsequent layers take input from the previous layer
        # using dense (fully connected) layers
        x = Dense(dim, activation=activation_func, name='encoder_hidden_layer_' + str(i))(input_layer)
    latent_dim = layer_hps[1]  # dimensionality of latent layer
    z = Dense(latent_dim, name='z_mean')(x)  # vanilla autoencoder

    # create the encoder
    encoder = Model(ae_input, z)

    # now, the same procedure for the decoder as for the encoder
    decoding_hidden_dims = layer_hps[2]
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
    ae_output = decoded

    # create the AutoEncoder
    autoencoder = Model(ae_input, ae_output, name='AutoEncoder')

    # create the decoder
    encoded_input = Input(shape=(latent_dim,))
    for i, layer in enumerate(decoder_layers):
        if i == 0:
            input = encoded_input
        else:
            input = y
        y = layer(input)
    decoder = Model(encoded_input, y)

    return autoencoder, encoder, decoder


def build_ae_emulator(layer_hps, activation_func='relu'):
    em_input_par = Input(shape=(X_train.shape[1],), name='em_input')
    em_hidden_dims = layer_hps[3]
    for i, dim in enumerate(em_hidden_dims):
        if i == 0:
            input_layer = em_input_par
        else:
            input_layer = x
        x = Dense(dim, activation=activation_func, name='em_hidden_layer_' + str(i))(input_layer)

    latent_dim = layer_hps[1]
    # the latent layer of the emulator
    autoencoder_par = Dense(latent_dim, name='em_autoencoder')(x)

    return emulator
