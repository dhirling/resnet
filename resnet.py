import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


def bn_relu(x):
    """
Performs BN and ReLU activation sequentially.
"""
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(x, n_filters1, n_filters2, n_blocks, layer, initializer=keras.initializers.HeNormal()):
    """
Basic building blocks for the ResNet-50 architecture.

Parameters
----------
x : tensor
  input tensor to conv block.
n_filters1 : int
  channel size during the first convolution part.
n_filters2 : int
  output channel size.
n_blocks : int
  defines how many times we should apply the "triple convolution".
layer : int
  specifies which layer we are on according to the original article.
initializer: keras initializer object
  specifies the kernel initialization method.

Returns
----------
x : tensor
  tensor after the convolution process
"""

    # store the original input
    identity = tf.identity(x)

    for n in range(n_blocks):
        # after layer 2, at the beginning of each layer, we want to downsample the input with 1x1 convolutions,
        # and strides of 2
        start_strides = 2 if (layer > 2 and n == 0) else 1
        x = keras.layers.Conv2D(filters=n_filters1, kernel_size=(1, 1), strides=start_strides, padding='same',
                                kernel_initializer=initializer)(x)
        x = bn_relu(x)
        # in every other case, strides are 1
        x = keras.layers.Conv2D(filters=n_filters1, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)(
            x)
        x = bn_relu(x)
        x = keras.layers.Conv2D(filters=n_filters2, kernel_size=(1, 1), padding='same', kernel_initializer=initializer)(
            x)
        x = keras.layers.BatchNormalization()(x)

        # if we are at the beginning of the block, the channel dimension of the original input should match the dim.
        # of our current x
        if n == 0:
            identity = keras.layers.Conv2D(filters=n_filters2, kernel_size=(1, 1), strides=start_strides,
                                           padding='same', kernel_initializer=initializer)(identity)
            identity = keras.layers.BatchNormalization()(identity)

        # add x and identity (skip connection) and apply ReLU
        x = keras.layers.Add()([x, identity])
        x = keras.layers.Activation('relu')(x)

    return x


def resnet(input_size, n_classes):
    """
  Builds the ResNet-50 architecture the same way as it was specified in the article table.

  Parameters
  ----------
  input_size : tuple
    input shape. Currently only supports original ImageNet shapes ((224,224,3)).
  n_classes : int
    how many classes we want to distinguish.

  Returns
  ----------
  model : keras model
    resnet-50 model.
    """

    # He weight initialization according to the article
    initializer = keras.initializers.HeNormal()

    # currently only supports the ImageNet input shapes
    assert (input_size == (224, 224, 3))

    # specify input shapes
    inputs = keras.layers.Input(input_size)

    # conv1_x
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer=initializer)(
        inputs)
    x = bn_relu(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    # conv2_x
    x = conv_block(x, 64, 256, n_blocks=3, layer=2)
    # conv3_x
    x = conv_block(x, 128, 512, n_blocks=4, layer=3)
    # conv4_x
    x = conv_block(x, 256, 1024, n_blocks=6, layer=4)
    # conv5_x
    x = conv_block(x, 512, 2048, n_blocks=3, layer=5)

    # this will flatten x
    x = keras.layers.GlobalAveragePooling2D()(x)

    # classification layer
    x = keras.layers.Dense(units=n_classes, activation='softmax', kernel_initializer=initializer)(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model
