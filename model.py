import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LSTM, Reshape, Dropout, Input, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def upsample(filters, size, kernel_regularizer=None):
    """
    Custom upsampling layer with L2 regularization.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    return result

def build_CNN_RNN_AE_model() -> Model:
    """
    Create CNN + RNN autoencoder model with regularization to prevent overfitting.
    
    Returns:
        (Model): Keras model.
    """
    # Define the base MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(input_shape=[32, 32, 12], include_top=False, weights=None)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 32x32
        'block_3_expand_relu',   # 16x16
        'block_6_expand_relu',   # 8x8
        'block_13_expand_relu',  # 4x4
        'block_16_project',      # 1x1
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = True

    # Define the upsampling stack with L2 regularization
    up_stack = [
        upsample(512, 3, kernel_regularizer=l2(1e-4)),  # 2x2 -> 4x4
        upsample(256, 3, kernel_regularizer=l2(1e-4)),  # 4x4 -> 8x8
        upsample(128, 3, kernel_regularizer=l2(1e-4)),  # 8x8 -> 16x16
        upsample(64, 3, kernel_regularizer=l2(1e-4)),   # 16x16 -> 32x32
    ]

    # Define model input
    inputs = tf.keras.layers.Input(shape=[32, 32, 12])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]  # Bottleneck output
    skips = reversed(skips[:-1])

    # Add RNN block at the bottleneck
    x = tf.keras.layers.Reshape((1, 320))(x)  # Flatten spatial dimensions for RNN
    x = tf.keras.layers.LSTM(512, return_sequences=True, activation='tanh', dropout=0.3)(x)
    x = tf.keras.layers.LSTM(512, activation='tanh', dropout=0.3)(x)
    x = tf.keras.layers.Reshape((1, 1, 512))(x)  # Reshape back to spatial dimensions

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Dropout(0.3)(x)  # Add dropout to the upsampled features
        x = Concatenate()([x, skip])

    # Final layer to restore original spatial resolution
    last = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2, padding='same'
    )  # 64x64 -> 128x128

    x = last(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create the segmentation model
segmentation_model = build_CNN_RNN_AE_model()
segmentation_model.summary()
