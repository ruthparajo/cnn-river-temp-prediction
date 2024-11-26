import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.layers import concatenate
import numpy as np
from tensorflow.keras.applications import ResNet50

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

def build_cnn_model_features2(input_shape, additional_inputs_shape):
    # Image input branch
    image_input = Input(shape=input_shape, name="Image_Input")

    # Convolutional layers with fewer pooling layers to accommodate smaller resolution
    x = layers.Conv2D(16, (3, 3), kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Reduces resolution to 32x32
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Reduces resolution to 16x16
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Reduces resolution to 8x8
    x = layers.Dropout(0.4)(x)

    # Global Average Pooling for dimensionality reduction
    x = layers.Flatten()(x)  # Output becomes a vector

    # Additional features input branch
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")
    y = layers.Dense(64, activation='relu')(additional_features_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # Combine both branches
    combined = layers.concatenate([x, y])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
    combined = layers.Dropout(0.3)(combined)

    # Output layer
    output = layers.Dense(1, activation='linear')(combined)

    # Define the model
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model

def build_cnn3(input_shape):
    model = models.Sequential()

    # Capa 1: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(16, (3, 3),kernel_regularizer=regularizers.l2(0.0001), input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 2: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa 3: Convolucional + Batch Normalization + Leaky ReLU + Max Pooling
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0001),))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(layers.Flatten()) # try flatten

    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.3))  # Increased Dropout
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(layers.Dense(1, activation='linear'))

    return model

def build_cnn_model_features3(input_shape, additional_inputs_shape):
    # Image input branch
    image_input = Input(shape=input_shape, name="Image_Input")

    # Convolutional layers with fewer pooling layers to accommodate smaller resolution
    x = layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Reduces resolution to 32x32
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Reduces resolution to 16x16
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Reduces resolution to 8x8
    x = layers.Dropout(0.4)(x)

    # Global Average Pooling for dimensionality reduction
    x = layers.Flatten()(x)  # Output becomes a vector

    # Additional features input branch
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")
    y = layers.Dense(64, activation='relu')(additional_features_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # Combine both branches
    combined = layers.concatenate([x, y])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
    combined = layers.Dropout(0.3)(combined)

    # Output layer
    output = layers.Dense(1, activation='linear')(combined)

    # Define the model
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model

def resnet_block(x, filters, kernel_size=3, stride=1):
    """A basic ResNet block with skip connections."""
    shortcut = x

    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Add skip connection
    if stride > 1 or x.shape[-1] != shortcut.shape[-1]:  # Match dimensions
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)

    return x

def build_resnet(input_shape, additional_inputs_shape):
    # Image input branch
    image_input = Input(shape=input_shape, name="Image_Input")

    # Initial Conv Layer
    x = layers.Conv2D(64, (7, 7), strides=2, padding="same")(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Residual Blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)  # Downsample
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, stride=2)  # Downsample
    x = resnet_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)

    # Additional features input branch
    additional_features_input = Input(shape=(additional_inputs_shape,), name="Additional_Features_Input")
    y = layers.Dense(64, activation='relu')(additional_features_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # Combine both branches
    combined = layers.concatenate([x, y])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)

    # Output layer
    output = layers.Dense(1, activation='linear')(combined)

    # Define the model
    model = models.Model(inputs=[image_input, additional_features_input], outputs=output)

    return model

def build_simple_resnet(input_shape):
    """Build a ResNet-like model for image inputs."""
    # Input layer
    image_input = Input(shape=input_shape, name="Image_Input")

    # Initial convolutional layer
    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Residual blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)  # Downsample
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, stride=2)  # Downsample
    x = resnet_block(x, 256)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense output layer for scalar prediction
    output = layers.Dense(1, activation="linear")(x)

    # Define the model
    model = models.Model(inputs=image_input, outputs=output)

    return model



def build_pretrained_resnet(input_shape):
    """
    Build a ResNet50 model that dynamically adapts to the number of input channels.
    Args:
        input_shape: Tuple of the input shape (H, W, C), where C is variable.
    Returns:
        A Keras Model.
    """
    print('A CONSTRUIR',input_shape)
    # Load the ResNet50 model without the top layers
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Get the first convolutional layer
    first_conv_layer = next(layer for layer in base_model.layers if isinstance(layer, layers.Conv2D))
    original_weights = first_conv_layer.get_weights()
    original_kernel, original_bias = original_weights

    # Dynamically adjust kernel weights for the input channels
    input_channels = input_shape[-1]
    if input_channels == 3:
        # If input is 3 channels, keep the original weights
        new_kernel = original_kernel
    else:
        # Average original weights across channels and tile to match input channels
        new_kernel = np.mean(original_kernel, axis=-2, keepdims=True)  # Average weights
        new_kernel = np.tile(new_kernel, (1, 1, input_channels, 1))  # Tile to match new channels

    # Define a new input layer for variable channels
    new_input = layers.Input(shape=input_shape, name="Input_Variable_Channels")

    # Replace the first convolutional layer with the modified one
    custom_first_layer = layers.Conv2D(
        filters=first_conv_layer.filters,
        kernel_size=first_conv_layer.kernel_size,
        strides=first_conv_layer.strides,
        padding=first_conv_layer.padding,
        activation=first_conv_layer.activation,
        name="custom_conv_channels",
        kernel_regularizer=regularizers.l2(0.001)
    )
    x = custom_first_layer(new_input)
    custom_first_layer.set_weights([new_kernel, original_bias])
    print(x.shape)

    first_layer_index = base_model.layers.index(first_conv_layer)

    # Pass the output through the remaining ResNet layers
    for i, layer in enumerate(base_model.layers[first_layer_index + 1:]):
        try:
            x = layer(x)
            print(f"Layer {i} ({layer.name}) output shape: {x.shape}")
        except Exception as e:
            print(f"Error in layer {i} ({layer.name}): {e}")
            break
        

    # Add custom layers for regression
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation='linear', name="Output")(x)

    # Build the model
    model = models.Model(inputs=new_input, outputs=output)
    return model

def build_pretrained_resnet_features(input_shape, scalar_shape):
    """
    Build a ResNet50 model with additional scalar inputs that dynamically adapts to the number of input channels.
    Args:
        input_shape: Tuple of the input shape (H, W, C), where C is variable.
        scalar_shape: Tuple representing the shape of the scalar inputs.
    Returns:
        A Keras Model.
    """
    # Load the ResNet50 model without the top layers
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Get the first convolutional layer
    first_conv_layer = next(layer for layer in base_model.layers if isinstance(layer, layers.Conv2D))
    original_weights = first_conv_layer.get_weights()
    original_kernel, original_bias = original_weights

    # Dynamically adjust kernel weights for the input channels
    input_channels = input_shape[-1]
    if input_channels == 3:
        # If input is 3 channels, keep the original weights
        new_kernel = original_kernel
    else:
        # Average original weights across channels and tile to match input channels
        new_kernel = np.mean(original_kernel, axis=-2, keepdims=True)  # Average weights
        new_kernel = np.tile(new_kernel, (1, 1, input_channels, 1))  # Tile to match new channels

    # Define a new input layer for variable channels
    image_input = layers.Input(shape=input_shape, name="Image_Input")

    # Replace the first convolutional layer with the modified one
    custom_first_layer = layers.Conv2D(
        filters=first_conv_layer.filters,
        kernel_size=first_conv_layer.kernel_size,
        strides=first_conv_layer.strides,
        padding=first_conv_layer.padding,
        activation=first_conv_layer.activation,
        name="Custom_Conv_Channels",
        kernel_regularizer=regularizers.l2(0.001)
    )
    x = custom_first_layer(image_input)
    custom_first_layer.set_weights([new_kernel, original_bias])

    first_layer_index = base_model.layers.index(first_conv_layer)

    # Pass the output through the remaining ResNet layers
    for i, layer in enumerate(base_model.layers[first_layer_index + 1:]):
        try:
            x = layer(x)
        except Exception as e:
            print(f"Error in layer {i} ({layer.name}): {e}")
            break

    # Add global average pooling for image features
    image_features = layers.GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)

    # Add a new input layer for scalar variables
    scalar_input = layers.Input(shape=scalar_shape, name="Scalar_Input")

    # Concatenate image features with scalar inputs
    combined_features = layers.concatenate([image_features, scalar_input], name="Concat_Image_Scalars")

    # Add fully connected layers for regression
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined_features)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation='linear', name="Output")(x)

    # Build the model
    model = models.Model(inputs=[image_input, scalar_input], outputs=output)
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

