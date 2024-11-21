import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.layers import concatenate
import numpy as np

############################# Models #############################

def build_cnn_baseline(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 2: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 3: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D()) # try flatten

    # Capas densas con Dropout reducido
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.2))  # Dropout reducido

    # Capa de salida
    model.add(layers.Dense(1, activation='linear'))

    return model


def build_cnn_baseline_8x8(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))  # Reduce de 8x8 a 4x4

    # Capa 2: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))  # Reduce de 4x4 a 2x2

    # Opcional: Otra capa convolucional si se requiere más profundidad
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())

    # Global Average Pooling en lugar de Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Capas densas con Dropout reducido
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.2))  # Dropout reducido

    # Capa de salida
    model.add(layers.Dense(1, activation='linear'))

    return model


def build_cnn_model_features(input_shape, additional_inputs_shape):
    # Entrada de imagen (7 canales, como en tu configuración actual)
    image_input = Input(shape=input_shape, name="Image_Input")

    # Rama de la imagen: Capas convolucionales
    x = layers.Conv2D(16, (3, 3), kernel_regularizer=regularizers.l2(0.0001))(image_input) 
    # x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global Average Pooling para reducir la dimensionalidad de la imagen
    x = layers.GlobalAveragePooling2D()(x)

    # Entrada de características adicionales (vectores)
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")

    # Procesamiento de los inputs adicionales
    y = layers.Dense(32, activation='relu')(additional_features_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)  # Regularización

    # Concatenación de las dos ramas
    combined = concatenate([x, y])

    # Capas densas finales después de la concatenación
    combined = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(combined)
    combined = layers.Dropout(0.2)(combined)

    # Capa de salida con una sola neurona para la predicción escalar
    output = layers.Dense(1, activation='linear')(combined)

    # Definición del modelo con ambas entradas
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model

def build_cnn_model_features_8x8(input_shape, additional_inputs_shape):
    # Entrada de imagen (7 canales, como en tu configuración actual)
    image_input = Input(shape=input_shape, name="Image_Input")

    # Rama de la imagen: Capas convolucionales
    x = layers.Conv2D(16, (3, 3),  padding='same')(image_input) 
    # x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global Average Pooling para reducir la dimensionalidad de la imagen
    x = layers.GlobalAveragePooling2D()(x)

    # Entrada de características adicionales (vectores)
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")

    # Procesamiento de los inputs adicionales
    y = layers.Dense(32, activation='relu')(additional_features_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)  # Regularización

    # Concatenación de las dos ramas
    combined = concatenate([x, y])

    # Capas densas finales después de la concatenación
    combined = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(combined)
    combined = layers.Dropout(0.2)(combined)

    # Capa de salida con una sola neurona para la predicción escalar
    output = layers.Dense(1, activation='linear')(combined)

    # Definición del modelo con ambas entradas
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model


def build_cnn_baseline(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 2: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 3: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D()) # try flatten

    # Capas densas con Dropout reducido
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))  # Dropout reducido

    # Capa de salida
    model.add(layers.Dense(1, activation='linear'))

    return model

def build_cnn_model_features(input_shape, additional_inputs_shape):
    # Entrada de imagen (7 canales, como en tu configuración actual)
    image_input = Input(shape=input_shape, name="Image_Input")

    # Rama de la imagen: Capas convolucionales
    x = layers.Conv2D(16, (3, 3), kernel_regularizer=regularizers.l2(0.0001))(image_input) 
    # x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global Average Pooling para reducir la dimensionalidad de la imagen
    x = layers.GlobalAveragePooling2D()(x)

    # Entrada de características adicionales (vectores)
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")

    # Procesamiento de los inputs adicionales
    y = layers.Dense(32, activation='relu')(additional_features_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)  # Regularización

    # Concatenación de las dos ramas
    combined = concatenate([x, y])

    # Capas densas finales después de la concatenación
    combined = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(combined)
    combined = layers.Dropout(0.2)(combined)

    # Capa de salida con una sola neurona para la predicción escalar
    output = layers.Dense(1, activation='linear')(combined)

    # Definición del modelo con ambas entradas
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model

############################# Loss functions #############################

def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def conservation_energy_loss(y_true, y_pred, model_input_batch, alpha=0.5, beta=0.5, threshold=5.0):
    # Calcular el promedio de los primeros tres canales de model_input_batch
    lst_batch = tf.reduce_mean(tf.cast(model_input_batch[:, :, :, :3], 'float32'))#, axis=-1)

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

def conservation_energy_loss_v0(y_true, y_pred, lst, alpha=0.5, beta=0.5):
    # Calcular el promedio de los primeros tres canales de model_input_batch
    lst_batch = tf.reduce_mean(tf.cast(lst, 'float32'), axis=-1)

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

