from functions import *
from models import *
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import gc
import time
from datetime import datetime
import numpy as np
import argparse
from collections import Counter
import os
from multiprocessing import Pool
import tensorflow as tf
import io
from contextlib import redirect_stdout
import logging
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import pickle

logging.getLogger('tensorflow').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def run_experiment(data_folder, model_name, batch_size, epochs, W=256, inputs=None, loss_type=None, filt_alt=None, num=0, \
                   augment= False, split_num = 0):

    print(f"Running experiment with model={model_name}, batch_size={batch_size}, epochs={epochs}, inputs = {inputs}")
    
    with open('../data/external/cos_to_month.pkl', 'rb') as file:
        cos_to_month = pickle.load(file)
        
    train_dir = os.path.join(data_folder, 'train')
    validation_dir = os.path.join(data_folder, 'validation')
    test_dir = os.path.join(data_folder, 'test')
    
    # Load train data
    train_input = np.load(os.path.join(train_dir, 'input_data.npy'))
    train_target = np.load(os.path.join(train_dir, 'target_data.npy'))
    additional_inputs_train = np.load(os.path.join(train_dir, 'additional_inputs.npy'))
    
    # Load validation data
    validation_input = np.load(os.path.join(validation_dir, 'input_data.npy'))
    validation_target = np.load(os.path.join(validation_dir, 'target_data.npy'))
    additional_inputs_validation = np.load(os.path.join(validation_dir, 'additional_inputs.npy'))
    
    # Load test data
    test_input = np.load(os.path.join(test_dir, 'input_data.npy'))
    test_target = np.load(os.path.join(test_dir, 'target_data.npy'))
    additional_inputs_test = np.load(os.path.join(test_dir, 'additional_inputs.npy'))
    input_shape = train_input.shape[1:]

    # Define the variables and their positions or channels
    var_channels = {'lst': 0, 'ndvi': 1, 'slope': 2, 'altitude': 3, 'direction': 4}
    var_position = {'month': [0, 1], 'coords': [2, 3], 'discharge': 4}
    
    # Initialize inputs
    train_image_inputs = []  # To store image-based inputs
    train_vector_inputs = []  # To store vector-based inputs
    
    val_image_inputs = []
    val_vector_inputs = []
    
    test_image_inputs = []
    test_vector_inputs = []
    
    # Build the inputs based on the variables in the list `inputs`
    for inp in inputs:
        if inp in var_channels:
            # Add image channels
            channel = var_channels[inp]
            train_image_inputs.append(train_input[..., channel:channel + 1])  # Slicing to preserve dimensions
            val_image_inputs.append(validation_input[..., channel:channel + 1])
            test_image_inputs.append(test_input[..., channel:channel + 1])
        elif inp in var_position:
            # Add vector-based inputs from `additional_inputs`
            position = var_position[inp]
            if isinstance(position, list):
                train_vector_inputs.append(additional_inputs_train[:, position])
                val_vector_inputs.append(additional_inputs_validation[:, position])
                test_vector_inputs.append(additional_inputs_test[:, position])
            else:
                train_vector_inputs.append(additional_inputs_train[:, position:position + 1])
                val_vector_inputs.append(additional_inputs_validation[:, position:position + 1])
                test_vector_inputs.append(additional_inputs_test[:, position:position + 1])
    
    # Combine image inputs along the last axis
    train_image_inputs = np.concatenate(train_image_inputs, axis=-1) if train_image_inputs else None
    val_image_inputs = np.concatenate(val_image_inputs, axis=-1) if val_image_inputs else None
    test_image_inputs = np.concatenate(test_image_inputs, axis=-1) if test_image_inputs else None
    
    # Combine vector inputs along the last axis
    train_vector_inputs = np.concatenate(train_vector_inputs, axis=-1) if train_vector_inputs else None
    val_vector_inputs = np.concatenate(val_vector_inputs, axis=-1) if val_vector_inputs else None
    test_vector_inputs = np.concatenate(test_vector_inputs, axis=-1) if test_vector_inputs else None
    
    # Adjust input_args and model_input format
    if train_vector_inputs is not None:  # If vector inputs are present
        input_args = (train_image_inputs.shape[1:], train_vector_inputs.shape[1])
        train_model_input = [train_image_inputs, train_vector_inputs]
        val_model_input = [val_image_inputs, val_vector_inputs]
        test_model_input = [test_image_inputs, test_vector_inputs]
    else:  # Only image inputs
        input_args = train_image_inputs.shape[1:]
        train_model_input = train_image_inputs
        val_model_input = val_image_inputs
        test_model_input = test_image_inputs
    
    # Print shapes for verification
    if isinstance(train_model_input, list):
        print(f"Train model input shapes: {[x.shape for x in train_model_input]}")
    else:
        print(f"Train model input shape: {train_model_input.shape}")


    
    # Start model
    print('INPUT SHAPE',input_args)
    model = build_model_map(model_name, input_args, W)
    start_time = time.time()
        
    summary_file = f"../models/{model_name}_summary.txt"
    with open(summary_file, "w") as f:
        with redirect_stdout(f):
            model.summary()
    
    # Create batch dataset
    conditioned = len(train_model_input)==2
    if conditioned:
        dataset = tf.data.Dataset.from_tensor_slices(((train_model_input[0], train_model_input[1]), train_target))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((train_model_input, train_target))
        
    for inputss, label in dataset.take(1):
        print("Image Stack Shape:", inputss[0].shape)  # (64, 64, 5)
        print("Scalars Shape:",  inputss[1].shape)     # (5,)
        print("Label Shape:", label.shape) 
    
    if augment:
        # Duplicate the dataset for augmentation
        if conditioned:
            augmented_dataset = (
            dataset.map(lambda inputs, label: augment_data(inputs[0], label, inputs[1]), num_parallel_calls=tf.data.AUTOTUNE)  # Augmentation
        )
        else:
            augmented_dataset = (
                dataset.map(lambda image, label: augment_data(image, label), num_parallel_calls=tf.data.AUTOTUNE)  # Augmentation
            )
        
        # Combine original and augmented datasets
        dataset = dataset.concatenate(augmented_dataset)
    
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        
    dataset_val = tf.data.Dataset.from_tensor_slices((*val_model_input, validation_target) if conditioned else (val_model_input, validation_target))
    dataset_val = dataset_val.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    if augment:
        sample_image, sample_label = next(iter(dataset))  # Take one batch
        augmented_images = [
                            augment_data(sample_image[0], sample_label[0])[0].numpy()
                            if not conditioned
                            else augment_data(sample_image[0][0], sample_label[0])[0].numpy()
                            for _ in range(10)
                        ]
        # Save a grid plot of augmented images
        plt.figure(figsize=(15, 5))
        for i, aug_img in enumerate(augmented_images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(aug_img[:, :, :3])  # Show only the RGB channels
            plt.title(f"Augmented {i+1}")
            plt.axis("off")
        plt.tight_layout()
        # Save the plot
        output_plot_path = "../plots/augmented_examples_plot.png"
        plt.savefig(output_plot_path, dpi=300)
    
        output_file = "../official_results/augmentation_report.txt"
        # Count images in original and augmented datasets
        original_images_count = count_images(dataset)  # Assuming `dataset` is your original dataset
        augmented_images_count = count_images(augmented_dataset)  # Assuming `augmented_dataset` contains augmented images
        
        # Write the counts to a text file
        with open(output_file, "w") as f:
            f.write("Augmentation Report\n")
            f.write("===================\n")
            f.write(f"Original images: {original_images_count}\n")
            f.write(f"Augmented images: {augmented_images_count}\n")
            f.write(f"Total images (original + augmented): {original_images_count + augmented_images_count}\n")

    

    #initial_lr = 0.01
    #lr_schedule = ExponentialDecay(initial_lr, decay_steps=50, decay_rate=0.96, staircase=True)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    #optimizer = tf.keras.optimizers.SGD()
    optimizer = tf.keras.optimizers.Adam()
    errors_log = {"epoch": [], "month": [], "error": []}
    loss_per_epoch = []
    val_loss_per_epoch = []

    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
     
    
    # Early Stopping parameters
    patience = 30  # Number of epochs with no improvement before stopping
    min_delta = 1e-4  # Minimum improvement required to consider progress
    best_val_loss = float('inf')  # Best observed validation loss
    wait = 0  # Counter for epochs without improvement

    # Train model
    for epoch in range(epochs):
        epoch_loss = 0  
        num_batches = 0
        start_idx = 0
        dataset = dataset.shuffle(buffer_size=len(dataset))
        
        for batch in dataset:
            if conditioned:
                model_input_batch = batch[:-1]  
                target_batch = batch[-1]        
            else:
                model_input_batch, target_batch = batch  # Direct unpacking for a single dataset
            
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model([*model_input_batch], training=True) if conditioned else model(model_input_batch, training=True)
                
                # Compute loss based on the selected method
                if loss_type == 'Physics_guided':
                    lst_batch = model_input_batch[0][:, :, :, :3] if conditioned else model_input_batch[:, :, :, :3]
                    loss = conservation_energy_loss(target_batch, y_pred, lst_batch, alpha=0.5, beta=0.5)
                elif loss_type == 'RMSE_sensitive':
                    loss = rmse_extreme_sensitive(target_batch, y_pred, k1=0.01, k2=1.0, alpha=1.0)
                elif loss_type == 'RMSE_focal':
                    loss = rmse_focal(target_batch, y_pred, gamma=1.0)
                else:
                    loss = root_mean_squared_error(target_batch, y_pred) 
            
            # Calculate gradients and apply optimization
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
            epoch_loss += loss.numpy()
            num_batches += 1
            
            # Log variables values and error
            y_true = tf.cast(target_batch, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            current_batch_size = y_true.shape[0]
            batch_cosine_values = additional_inputs_train[:, 0][start_idx:start_idx + current_batch_size]
        
            # Log RMSE values for each prediction
            for cos, pred, true in zip(batch_cosine_values, y_pred, y_true):
                squared_error = tf.square(pred - true) 
                rmse_sample = tf.sqrt(squared_error)  # RMSE 
                rmse_value = rmse_sample.numpy()
                errors_log["epoch"].append(epoch + 1)
                errors_log["month"].append(cos_to_month[cos])
                errors_log["error"].append(rmse_value)
            start_idx += current_batch_size
                
        avg_epoch_loss = epoch_loss / num_batches
        loss_per_epoch.append(avg_epoch_loss)
        
        # Validation loss
        val_loss = 0
        val_batches = 0
        
        for val_batch in dataset_val:
            if conditioned:
                val_input_batch = val_batch[:-1]  
                val_target_batch = val_batch[-1]        
            else:
                val_input_batch, val_target_batch = val_batch
            
            val_pred = model([*val_input_batch], training=False) if conditioned else model(val_input_batch, training=False)
            val_loss += root_mean_squared_error(val_target_batch, val_pred).numpy()
            val_batches += 1
            
        avg_val_loss = val_loss / val_batches
        val_loss_per_epoch.append(avg_val_loss)
    
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
        # Early Stopping Logic
        if avg_val_loss < best_val_loss - min_delta:
            # Update the best validation loss and reset patience counter
            best_val_loss = avg_val_loss
            wait = 0  
            print(f"Validation loss improved to {best_val_loss:.4f}.")
        else:
            # Increment patience counter
            wait += 1
            print(f"No improvement in validation loss for {wait} epochs.")
            if wait >= patience:
                # Stop training if patience threshold is exceeded
                print(f"Stopping early at epoch {epoch + 1}.")
                break
        
        gc.collect()  # Free up memory after each epoch
    


    # Convert the error log to a DataFrame for further analysis
    errors_df = pd.DataFrame(errors_log)
    errors_df["error"] = errors_df["error"].apply(lambda x: x[0] if len(x) == 1 else x)
    print('Done training!')

    errors_df.to_csv(f'../official_results/error_logs/{model_name}_exp_{num}_split_{split_num}.csv',index=False)

    # Plot training curve
    num_final_epochs = epoch + 1
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_final_epochs + 1), loss_per_epoch, label='Training Loss')
    plt.plot(range(1, num_final_epochs + 1), val_loss_per_epoch, label='Validation Loss')
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fig1 = plt.gcf()
    fig1.savefig(f"../official_results/learning_curves/{model_name}_exp_{num}_split_{split_num}.png", dpi=100) 
    plt.close(fig1)
    
    # Evaluate results
    print('\nComputing result metrics...')
    test_prediction = model.predict(test_model_input)
    rmse_test = mean_squared_error(test_target, test_prediction, squared=False)
    print('RMSE:',rmse_test,'\n')
    
    # Get experiment data
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Save model results
    if isinstance(optimizer, tf.keras.optimizers.Adam):
        opt = 'Adam'
    elif isinstance(optimizer, tf.keras.optimizers.SGD):
        opt = 'SGD'
        
    variables = '' if inputs == None else ', '.join(inputs)
    samples_str = f'{count_images(dataset)} images + {augmented_images_count} augmented' if augment else f'{count_images(dataset)} images'

    details = {'Experiment':num,'RMSE':rmse_test,'Variables':variables, 'Split_id': split_num,'Optimizer': opt, \
               'nº samples': samples_str, 'Batch size': batch_size, \
               'Epochs': f'{num_final_epochs} of {epochs}','Date':current_date,'Time':current_time, 'Duration': duration, \
               'Loss':  loss_type, 'Resolution':W}
     
    file_path = f"../official_results/{model_name}_results.xlsx"
    save_excel(file_path, details, excel = 'Results')
    model.save(f'../models/{model_name}.h5')
    print(model.summary())
    if conditioned:
        print(f"Experiment {model_name} with batch_size={batch_size} and epochs={epochs} completed and {input_shape, additional_inputs.shape} inputs .\n")
    else:
    
        print(f"Experiment {model_name} with batch_size={batch_size} and epochs={epochs} completed and {input_shape} inputs .\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to run experiments with deep learning models.")

    # Define command-line arguments
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32],
                        help="List of batch sizes for training the model. Example: --batch_sizes 16 32")
    parser.add_argument('--epochs_list', nargs='+', type=int, default=[10, 50, 100],
                        help="List of epochs for training the model. Example: --epochs_list 10 50 100")
    parser.add_argument('--model_names', nargs='+', type=str, default=['img_wise_CNN', 'UNet', 'CNN', 'img_2_img'],

                        help="List of model names. Example: --model_names baseline_CNN UNet CNN img_2_img")
    parser.add_argument('--inputs', nargs='+', type=str, required=True,
                        help="List of inputs to include . Example: --inputs lst ndvi")

    parser.add_argument('--resolution', nargs='+', type=int, required=True,
                        help="Image Resolution. Example: --resolution 64")

    parser.add_argument('--loss_type', nargs='+', type=str, required=True,
                        help="Type of loss . Example: --loss_type RMSE, RMSE_sensitive, Physics_guided")

    # Parse arguments
    args = parser.parse_args()

    # Extract argument values
    batch_sizes = args.batch_sizes
    epochs_list = args.epochs_list
    model_names = args.model_names
    inputs = args.inputs
    W = args.resolution[0]
    loss_type = args.loss_type[0]
    
    print(inputs)
    if inputs[0] == 'full features':
        inputs = ['lst', 'ndvi','slope','altitude','direction','month','coords','discharge']
    elif inputs[0] == 'image features':
        inputs = ['lst', 'ndvi','slope','altitude','direction']
        
        
    filt_alt = False 
    augment = False
    
    # Run experiments with parameter combinations
    for model_name in model_names:
        exp_num = get_next_experiment_number(f'../official_results/{model_name}_results.xlsx')
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                #for c in [True, False]:
                for split in range(1,6):
                    data_folder = f'../data/processed_data/{W}x{W}/{split}'
                    run_experiment(data_folder, model_name, batch_size, epochs, W=W, inputs=inputs, \
                                   loss_type=loss_type, filt_alt=filt_alt, num = exp_num, augment = augment, split_num = split)
                