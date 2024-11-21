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


logging.getLogger('tensorflow').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def run_experiment(data, model_name, batch_size, epochs, W=256, conditioned=False, inputs=None, split=None,\
                   physics_guided=None, filt_alt=None, num=0, split_num=0):

    print(f"Running experiment with model={model_name}, batch_size={batch_size}, epochs={epochs}, inputs = {inputs}")

    # Gather input and target data 
    total_data, total_times, data_targets, labels = data
    cosine_months, sine_months, cos_to_month = get_months_vectorized(total_times['lst'])
    lat, lon = get_lat_lon(labels)
    discharge = get_discharge(labels, total_times['lst'])
    additional_inputs = np.column_stack((cosine_months, sine_months, lat, lon, discharge))

    
    grad_output_folder = f'../plots/grad_cam/{model_name}/exp_{num}'
    os.makedirs(grad_output_folder, exist_ok=True)

    # Calculate global min and max for each variable
    global_ranges = {}
    for inp in inputs:  # 'inputs' is the list of variable names
        all_images = total_data[inp]  # Shape: (n, H, W) or (n, H, W, C) for the variable
        global_min = np.min(all_images)
        global_max = np.max(all_images)
        global_ranges[inp] = (global_min, global_max)
        print(f"Variable: {inp}, Min: {global_min}, Max: {global_max}")
    
    # Adapt input shapes and normalize
    expanded_images = []
    for inp in inputs:  # Loop through each variable
        all_images = total_data[inp]  # Get all images for the variable
        min_val, max_val = global_ranges[inp]  # Get global min and max
        
        # Normalize all images for the current variable
        normalized_images = normalize_min_max(all_images, min_val, max_val)
        
        # Adjust shape if necessary
        if normalized_images.ndim == 3:  # Case where images are (n, H, W) (single-channel)
            normalized_images = np.expand_dims(normalized_images, axis=-1)  # Add a channel dimension
        
        expanded_images.append(normalized_images)
    
    # Combine all normalized inputs along the last axis (channels)
    combined_input = np.concatenate(expanded_images, axis=-1)  # Concatenate along channel axis
    input_data = combined_input

    # Split data
    print('split', split)
    train_index, validation_index, test_index = get_split_index(split[0], input_data, data_targets, labels, split_num, filt_alt)
    validation_input = input_data[validation_index, :] 
    validation_target = data_targets[validation_index]
    test_input = input_data[test_index, :]
    test_target = data_targets[test_index]
    train_input = input_data[train_index, :]
    train_target = data_targets[train_index]
    
    additional_inputs_train = additional_inputs[train_index, :]
    additional_inputs_validation = additional_inputs[validation_index, :]
    additional_inputs_test = additional_inputs[test_index, :]

    # Adapt input to condition
    if len(train_input.shape) == 3:
        input_shape = train_input.shape[1:]+(1,)
    else:
        input_shape = train_input.shape[1:]

    if conditioned:
        input_args = (input_shape, additional_inputs.shape[1])
        model_input = [train_input, additional_inputs_train]
        val_model_input = [validation_input, additional_inputs_validation]
        test_model_input = [test_input, additional_inputs_test]
    else:
        input_args = input_shape
        model_input = train_input
        val_model_input = validation_input
        test_model_input = test_input

    
    # Start model
    model = build_model_map(model_name, input_args, conditioned, W, train_input)
    start_time = time.time()
    '''
    if model_name == "baseline_CNN":
        if conditioned:
            model = build_cnn_model_features(input_args[0], input_args[1])
        elif W == 8 or W == 16:
            model = build_cnn_baseline_8x8(input_args)
        else:
            model = build_cnn_baseline(input_args)
    elif model_name == 'CNN':
        model = build_cnn_model(input_args)
    elif model_name == 'img_2_img':
        model = build_img_2_img_model(input_args)
    elif model_name == 'UNet':
        model = build_unet(input_args)
    elif model_name == 'transfer_learning_VGG16':
        train_input = train_input[:, :, :, :3]
        model = build_transfer_model((W, W, 3))
    elif model_name == "img_wise_CNN_improved":
        model = build_simplified_cnn_model_improved(input_args)'''
        
    summary_file = f"../models/{model_name}_summary.txt"
    with open(summary_file, "w") as f:
        with redirect_stdout(f):
            model.summary()
    
    # Create batch dataset
    dataset = tf.data.Dataset.from_tensor_slices((*model_input, train_target) if conditioned else (model_input, train_target))
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    dataset_val = tf.data.Dataset.from_tensor_slices((*val_model_input, validation_target) if conditioned else (val_model_input, validation_target))
    dataset_val = dataset_val.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    #initial_lr = 0.01
    #lr_schedule = ExponentialDecay(initial_lr, decay_steps=50, decay_rate=0.96, staircase=True)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
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
                if physics_guided:
                    lst_batch = model_input_batch[0][:, :, :, :3] if conditioned else model_input_batch[:, :, :, :3]
                    loss = conservation_energy_loss(target_batch, y_pred, lst_batch, alpha=0.5, beta=0.5)
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
    opt = 'Adam'#'SGD w/ dynamic-lr, momentum 0.9 & nesterov'#'Adam' #SGD
    laabeel = 'month, discharge, lat, lon' if conditioned else None
    var_inputs = '' if inputs == None else ', '.join(inputs)
    variables = ', '.join([var_inputs, laabeel]) if conditioned else var_inputs
    details = {'Experiment':num,'RMSE':rmse_test,'Variables':variables,'Input': f'{len(np.unique(labels))} cells', 'Split': split[0], \
               'Split_id': split_num,'Optimizer': opt,'nº samples': len(data_targets), 'Batch size': batch_size, \
               'Epochs': f'{num_final_epochs} of {epochs}','Date':current_date,'Time':current_time, 'Duration': duration, \
               'Loss':  'Physics-guided' if physics_guided else 'RMSE', 'Resolution':W}
    
    file_path = f"../official_results/{model_name}_results.xlsx"
    save_excel(file_path, details, excel = 'Results')
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

    parser.add_argument('--split', nargs='+', type=str, required=True,
                        help="Type of data split . Example: --split random, time, stratified")

    parser.add_argument('--resolution', nargs='+', type=int, required=True,
                        help="Type of data split . Example: --split random, time, stratified")

    # Parse arguments
    args = parser.parse_args()

    # Extract argument values
    batch_sizes = args.batch_sizes
    epochs_list = args.epochs_list
    model_names = args.model_names
    inputs = args.inputs
    split = args.split
    W = args.resolution[0]
    
    if W==128:
        data_folder = '../data/preprocessed/'
    else:
        data_folder = f'../data/preprocessed/{W}x{W}/'
    filt_alt = False 
    
    data = load_all_data(
    source_folder='../data/external/shp/river_cells_oficial',
    source_path=data_folder,
    data_paths= inputs,
    filter_altitude=filt_alt,
    W=W,
    time_split=True if split=='time' else False)
    

    # Run experiments with parameter combinations
    for model_name in model_names:
        exp_num = get_next_experiment_number(f'../official_results/{model_name}_results.xlsx')
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                #run_experiment(data, model_name, batch_size, epochs, W=W, conditioned=False, inputs=inputs, split=split, physics_guided=True, \
                               #filt_alt=filt_alt, num = exp_num)
                for c in [True, False]:
                    run_experiment(data, model_name, batch_size, epochs, W=W, conditioned=c, inputs=inputs, split=split, physics_guided=False, \
                                  filt_alt=filt_alt, num = exp_num, split_num = 1)
                    run_experiment(data, model_name, batch_size, epochs, W=W, conditioned=c, inputs=inputs, split=split, physics_guided=False, \
                                  filt_alt=filt_alt, num = exp_num+1, split_num = 2)
                    run_experiment(data, model_name, batch_size, epochs, W=W, conditioned=c, inputs=inputs, split=split, physics_guided=False, \
                                  filt_alt=filt_alt, num = exp_num+2, split_num = 3)
                    