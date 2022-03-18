import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def _build_model(input_layer, out_dim, hidden_dims, activation_func):
    """
    Function that builds a Keras model.

    Parameters
    ----------
    input_layer : keras Layer obect
        Input layer to model.
    out_dim : int
        Dimension of output layer.
    hidden_dims : list of ints
        Dimensions of hidden layers.
    activation_func : st or tf.keras.activations object
        Name of a keras recognized activation function or an instance of a
        tf.keras.activations.
    
    Returns
    -------
    emulator : tf.keras Model object
        A keras model object.

    """
    x = input_layer
    for dim in hidden_dims:
        x = Dense(dim, activation=activation_func)(x)
    output_par = Dense(out_dim)(x)
    model = Model(input_layer, output_par)
    return model


def build_direct_emulator(
        par_train,
        signal_train,
        hidden_dims,
        activation_func="relu"
    ):
    """
    Function that builds the emulator.

    Parameters
    ----------
    par_train : numpy.ndarray
        Training parameters.
    signal_train : numpy.ndarray
        Training signals.
    hidden_dims : list of ints
        Dimensions of hidden layers.
    activation_func : st or tf.keras.activations object
        Name of a keras recognized activation function or an instance of a
        tf.keras.activations. Default 'relu'.
    
    Returns
    -------
    emulator : tf.keras Model object
        The emulator as a keras model object

    """

    em = _build_model(
        Input(shape=(par_train.shape[-1],)),
        signal_train.shape[-1],
        hidden_dims,
        activation_func
    )
    return em

def build_autoencoder(
        latent_dim,
        signal_train,
        enc_hidden_dims,
        dec_hidden_dims,
        enc_activation_func="relu",
        dec_actication_func="relu"
    ):
    """
    Function that builds the emulator.

    Parameters
    ----------
    latent_dim : int
        Dimensions of latent space.
    signal_train : numpy.ndarray
        Training signals
    enc_hidden_dims : list of ints
        Dimensions of hidden layers of encoder.
    dec_hidden_dims : list of ints
        Dimensions of hidden layers of decoder.
    enc_activation_func : st or tf.keras.activations object
        Name of a keras recognized activation function or an instance of a
        tf.keras.activations for the encoder. Default 'relu'.
    dec_activation_func : st or tf.keras.activations object
        Name of a keras recognized activation function or an instance of a
        tf.keras.activations for the decoder. Default 'relu'.
    
    Returns
    -------
    autoencoder, encoder, decoder : tf.keras Model objects
        The autoencoder and the corresponding encoder and decoder as keras
        model objects

    """

    encoder = _build_model(
        Input(shape=(signal_train.shape[-1],)),
        latent_dim,
        enc_hidden_dims,
        enc_activation_func
    )

    ae_decoder = _build_model(
        encoder.layers[-1],
        signal_train.shape[-1],
        dec_hidden_dims,
        dec_activation_func
    )

    decoder = _build_model(
        Input(shape=(latent_dim,)),
        signal_train.shape[-1],
        dec_hidden_dims,
        dec_activation_func
    )
    autoencoder = _build_model(
        Input


    return em

def build_autoencoder(layer_hps, signal_train, activation_func='relu'):
    """
    Function that builds the autoencoder.
    :param layer_hps: list, the hyperparameter controlling the number of layers
    and their dimensionalities
    :param signal_train: numpy array of training signals
    :param activation_func: str, name of a keras recognized activation function
    or a tf.keras.activations instance (see
    https://keras.io/api/layers/activations/)
    :return: the autoencoder as a keras model object
    """
    assert len(layer_hps) == 3,\
            "Layer hps should have one element for each of encoder, decoder," \
            "latent dim"
    encoding_hidden_dims = layer_hps[0]  # the layers of the encoder
    ae_input = Input(shape=(signal_train.shape[1],))  # autoencoder input layer

    # loop over the number of layers and build the encoder with layers of the
    # given dimensions
    layers = [ae_input]
    for i, dim in enumerate(encoding_hidden_dims):
        input_layer = layers[-1]
        # using dense (fully connected) layers
        x = Dense(
                dim,
                activation=activation_func,
                name='encoder_hidden_layer_' + str(i)
                )(input_layer)
        layers.append(x)
    latent_dim = layer_hps[1]  # dimensionality of latent layer
    z = Dense(latent_dim, name='z_mean')(layers[-1])  # vanilla autoencoder
    layers.append(z)
    # create the encoder
    encoder = Model(ae_input, z, name='Encoder')

    # now, the same procedure for the decoder as for the encoder
    decoding_hidden_dims = layer_hps[2]
    decoder_layers = []
    for i, dim in enumerate(decoding_hidden_dims):
        input_layer = layers[-1]
        decoder_layer = Dense(
                dim,
                activation=activation_func,
                name='decoder_hidden_layer_' + str(i)
                )
        decoder_layers.append(decoder_layer)
        x = decoder_layer(input_layer)
        layers.append(x)

    decoder_output_layer = Dense(signal_train.shape[1])  # output of decoder
    decoder_layers.append(decoder_output_layer)
    decoded = decoder_output_layer(layers[-1])
    ae_output = decoded

    # create the AutoEncoder
    autoencoder = Model(ae_input, ae_output, name='AutoEncoder')

    # create the decoder
    encoded_input = Input(shape=(latent_dim,))
    enc_layers = [encoded_input]
    for i, layer in enumerate(decoder_layers):
        layer_in = enc_layers[-1]
        y = layer(layer_in)
        layer_in.append(y)
    decoder = Model(encoded_input, layer_in[-1], name='Decoder')
    return autoencoder, encoder, decoder


def build_ae_emulator(layer_hps, par_train, activation_func='relu'):
    """
    Function that builds the autoencoder-based emulator.
    :param layer_hps: list, the hyperparameter controlling the number of layers
    and their dimensionalities
    :param par_train: numpy array of training parameters
    :param activation_func: str, name of a keras recognized activation function
    or a tf.keras.activations instance
    (see https://keras.io/api/layers/activations/)
    :return: the emulator as a keras model object
    """
    assert len(layer_hps) == 2,\
            "Layer hps should have one element for each of emulator, latent dim"
    em_input_par = Input(shape=(par_train.shape[1],), name='em_input')
    em_hidden_dims = layer_hps[1]
    layers = [em_input_par]
    for i, dim in enumerate(em_hidden_dims):  # add hidden layers one by one
        input_layer = layers[-1]
        x = Dense(
                dim,
                activation=activation_func,
                name='em_hidden_layer_' + str(i)
                )(input_layer)
        layers.append(x)

    latent_dim = layer_hps[0]  # the latent layer of the emulator
    autoencoder_par = Dense(latent_dim, name='em_autoencoder')(layers[-1])
    emulator = Model(em_input_par, autoencoder_par, name='AE_Emulator')
    return emulator
