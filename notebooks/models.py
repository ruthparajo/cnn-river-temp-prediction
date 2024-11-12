import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.layers import concatenate
import numpy as np


from tensorflow.keras import layers, models, regularizers

def build_cnn_baseline(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + Batch Normalization + ReLU + Max Pooling
    model.add(layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 2: Convolucional + Batch Normalization + ReLU + Max Pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 3: Convolucional + Batch Normalization + ReLU + Max Pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling para reducir las dimensiones
    model.add(layers.GlobalAveragePooling2D())

    # Capas densas con Dropout
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))  # Regularización para evitar sobreajuste

    # Capa de salida con una sola neurona para la predicción escalar
    model.add(layers.Dense(1, activation='linear'))

    return model


def build_cnn_model(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + Activación ReLU + Max Pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 2: Convolucional + Activación ReLU + Max Pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 3: Convolucional + Activación ReLU + Max Pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 4: Convolucional + Activación ReLU
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Capa de aplanamiento
    model.add(layers.Flatten())

    # Capa densa
    model.add(layers.Dense(256, activation='relu'))

    # Capa de salida con activación lineal (para predicciones de temperatura)
    model.add(layers.Dense(256 * 256, activation='linear'))

    # Reshape de la salida a la forma (256, 256)
    model.add(layers.Reshape((256, 256)))

    return model

def build_img_2_img_model(input_shape):
    model = models.Sequential()

    # Encoder: downsampling using Conv2D + MaxPooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Bottleneck
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Decoder: upsampling using Conv2DTranspose
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2D(1, (3, 3), activation='linear', padding='same'))  # 1 channel output for temperature

    return model

def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)

    # Decoder with skip connections
    u4 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)

    u5 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='linear')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def build_transfer_model(input_shape, base_model_trainable=False):
    """
    Creates a transfer learning model based on VGG16 with custom top layers.
    
    Parameters:
    - input_shape: tuple representing the input shape, e.g., (256, 256, 3)
    - base_model_trainable: bool, whether to make the base VGG16 model trainable or not

    Returns:
    - transfer_model: the created Keras model
    """
    
    # Load VGG16 pre-trained on ImageNet without the top fully-connected layers
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = base_model_trainable  # Set whether the base model layers are trainable

    # Create custom top layers
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(256 * 256, activation='linear')(x)
    outputs = layers.Reshape((256, 256))(outputs)

    # Create the full model
    transfer_model = models.Model(inputs, outputs)
    
    return transfer_model

# Función de error cuadrático medio
def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def conservation_energy_loss(y_true, y_pred, model_input_batch, alpha=0.5, beta=0.5, threshold=5.0):
    # Calcular el promedio de los primeros tres canales de model_input_batch
    lst_batch = tf.reduce_mean(tf.cast(model_input_batch[0][:, :, :, :3], 'float32'))#, axis=-1)

    # Calcular Q_in y Q_out
    Q_in = tf.maximum(lst_batch - y_pred, 0)  # Solo calor hacia el agua cuando LST > predicción
    Q_out = tf.maximum(y_pred - lst_batch, 0) # Solo calor hacia afuera cuando predicción > LST

    differences = np.abs(lst_batch - y_pred)
    threshold = np.percentile(differences, 80)
    
    # Penalizar diferencias extremas: solo cuando Q_in o Q_out superen el umbral
    extreme_penalty = tf.where(tf.abs(Q_in - Q_out) > threshold, tf.abs(Q_in - Q_out), 0)
    physics_loss = tf.reduce_mean(extreme_penalty)

    # Pérdida basada en datos (RMSE)
    data_loss = root_mean_squared_error(y_true, y_pred)

    # Pérdida total combinada
    return alpha * data_loss + beta * physics_loss

def conservation_energy_loss_v0(y_true, y_pred, model_input_batch, alpha=0.5, beta=0.5):
    # Calcular el promedio de los primeros tres canales de model_input_batch
    lst_batch = tf.reduce_mean(tf.cast(model_input_batch[:, :, :, :3], 'float32'), axis=-1)

    # Calcular Q_in y Q_out
    Q_in = tf.reduce_mean(tf.maximum(lst_batch - y_pred, 0))
    Q_out = tf.reduce_mean(tf.maximum(y_pred - lst_batch, 0))

    # Pérdida de conservación de energía
    physics_loss = tf.reduce_mean(tf.abs(Q_in - Q_out))
    
    # Pérdida basada en datos (RMSE)
    data_loss = root_mean_squared_error(y_true, y_pred)
    print("Data Loss (RMSE):", data_loss)
    print("Physics Loss (Energy Conservation):", physics_loss)

    # Pérdida total combinada
    return alpha * data_loss + beta * physics_loss

def build_simplified_cnn_model_improved(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + BatchNormalization + ReLU + Max Pooling + Dropout
    model.add(layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0005), input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Capa 2: Convolucional + BatchNormalization + ReLU + Max Pooling + Dropout
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Capa de aplanamiento
    model.add(layers.Flatten())

    # Capa densa + BatchNormalization + Dropout
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    # Capa de salida con activación lineal (para predicciones de temperatura)
    model.add(layers.Dense(256 * 256, activation='linear'))

    # Reshape de la salida a la forma (256, 256)
    model.add(layers.Reshape((256, 256)))

    return model


def build_cnn_model_features(input_shape, additional_inputs_shape):
    # Image input (temperature)
    image_input = Input(shape=input_shape, name="Image_Input")

    # Convolutional Layer 1 + Batch Normalization + ReLU + Max Pooling
    x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Layer 2 + Batch Normalization + ReLU + Max Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Layer 3 + Batch Normalization + ReLU + Max Pooling
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global Average Pooling to reduce dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Additional features input (for scalar features like month sine/cosine, latitude, longitude)
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")

    # Concatenate the CNN output with additional features (month sin/cos, lat, lon)
    x = concatenate([x, additional_features_input])

    # Dense layer after concatenation with Dropout for regularization
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)

    # Output layer with a single neuron for scalar prediction
    output = layers.Dense(1, activation='linear', name="Scalar_Output")(x)

    # Final model with image and additional features inputs
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model
'''
def build_cnn_model_features(input_shape, additional_inputs_shape):
    # Image input (temperature)
    image_input = Input(shape=input_shape, name="Image_Input")

    # Convolutional Layer 1 + ReLU Activation + Max Pooling
    x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Layer 2 + ReLU Activation + Max Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten the convolutional output
    x = layers.Flatten()(x)

    # Combined input for month sine/cosine, latitude, and longitude
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")

    # Concatenate the CNN output with river label and additional features (month sin/cos, lat, lon)
    x = concatenate([x, additional_features_input])

    # Dense layer after concatenation
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)

    # Output layer with linear activation (for scalar temperature prediction)
    output = layers.Dense(1, activation='linear', name="Scalar_Output")(x)

    # Final model with three inputs
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model'''
