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
    
tf.config.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)
print("Using GPU:", gpus[1])




def run_experiment(data_folder, model_name, batch_size, epochs, W=256, inputs=None, loss_type=None, filt_alt=None, num=0, \
                   augment= False, split_num = 0, mode = None):

    print(f"Running experiment with model={model_name}, batch_size={batch_size}, epochs={epochs}, inputs = {inputs}, resolution = {W}")

    # Collect datasets
    
    var_channels = {'lst': 0, 'ndvi': 1, 'slope': 2, 'altitude': 3, 'direction': 4}
    var_position = {'month': [0, 1], 'coords': [2, 3], 'discharge': 4}
    with open('../data/external/cos_to_month.pkl', 'rb') as file:
        cos_to_month = pickle.load(file)
    print(inputs)
    train_model_input, train_additional_inputs, train_target = load_set(data_folder, inputs, 'train', var_channels, var_position)
    val_model_input, validation_additional_inputs, validation_target = load_set(data_folder, inputs, 'validation', var_channels, var_position)
    test_model_input, test_additional_inputs, test_target = load_set(data_folder, inputs, 'test', var_channels, var_position)

    months = [cos_to_month[val] for val in train_additional_inputs[..., var_position['month'][0]]]
    reference = (months, train_additional_inputs[..., var_position['coords']]) 
    conditioned = len(train_model_input) == 2
    
    
    if augment:
        dest_dir = "../data/processed_data/augmented_all_5x"
        train_dataset = load_numpy_dataset(f"{dest_dir}/{split_num}/train_dataset")
        channels_to_keep = [ch for k, ch in var_channels.items() if k in inputs] 
        train_dataset = select_channels(train_dataset, channels_to_keep)

    else:
        train_dataset = prepare_dataset(train_model_input, train_target, conditioned, W, reference, augment=augment)
        
    val_dataset = prepare_dataset(val_model_input, validation_target, conditioned, W, augment=False)

    
    print('que',len(train_model_input), conditioned, train_model_input[0].shape,train_model_input[1].shape)
        
    # Finalize the datasets
    count_im = len(train_dataset)
    train_dataset = train_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # Start model
    if conditioned:
        input_args = (train_model_input[0].shape[1:], train_model_input[1].shape[1])
    else:
        input_args = train_model_input.shape[1:]
        
    model = build_model_map(model_name, input_args, W)
    start_time = time.time()
        
    summary_file = f"../models/{model_name}_summary.txt"
    with open(summary_file, "w") as f:
        with redirect_stdout(f):
            model.summary()

    # Set hyperparmeters variables
    #initial_lr = 0.01
    #lr_schedule = ExponentialDecay(initial_lr, decay_steps=50, decay_rate=0.96, staircase=True)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    #optimizer = tf.keras.optimizers.SGD()
    optimizer = tf.keras.optimizers.Adam()
    errors_log = {"epoch": [], "month": [], "error": []}
    loss_per_epoch = []
    val_loss_per_epoch = []
    
    # Early Stopping parameters
    patience = 30  # Number of epochs with no improvement before stopping
    min_delta = 1e-4  # Minimum improvement required to consider progress
    best_val_loss = float('inf')  # Best observed validation loss
    wait = 0  # Counter for epochs without improvement

    # Train model
    for epoch in range(epochs):
        epoch_loss = 0  
        num_batches = 0
        train_dataset = train_dataset.shuffle(buffer_size=len(train_model_input)).prefetch(tf.data.AUTOTUNE)
        
        for batch in train_dataset:
            if augment:
                if conditioned:
                    model_input_batch = batch[1][0]  
                    target_batch = batch[1][1]
                    idx = batch[0]
                else:
                    model_input_batch = batch[1][0][0] 
                    target_batch = batch[1][1]
                    idx = batch[0]
            else:
                if conditioned:
                    model_input_batch = batch[1][0]
                    target_batch = batch[1][1]  
                    idx = batch[0]
                else:
                    idx, batch_data = batch
                    model_input_batch = batch_data[1][0][0]
                    target_batch = batch_data[1][1]
                
            
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model([*model_input_batch], training=True) if conditioned else model(model_input_batch, training=True)
                
                # Compute loss based on the selected method
                if loss_type == 'Physics_guided':
                    lst_batch = model_input_batch[0][:, :, :, :0] if conditioned else model_input_batch[:, :, :, 0]
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
            batch_cosine_values = train_additional_inputs[:, 0][idx]
        
            # Log RMSE values for each prediction
            for cos, pred, true in zip(batch_cosine_values, y_pred, y_true):
                squared_error = tf.square(pred - true) 
                rmse_sample = tf.sqrt(squared_error)  # RMSE 
                rmse_value = rmse_sample.numpy()
                errors_log["epoch"].append(epoch + 1)
                errors_log["month"].append(cos_to_month[cos])
                errors_log["error"].append(rmse_value)
                    
        avg_epoch_loss = epoch_loss / num_batches
        loss_per_epoch.append(avg_epoch_loss)
        
        # Validation loss
        val_loss = 0
        val_batches = 0
        
        for val_batch in val_dataset:
            if conditioned:
                val_input_batch = val_batch[1][0]  
                val_target_batch = val_batch[1][1]  
                idx = val_batch[0]
            else:
                idx, batch_data = val_batch
                val_input_batch = batch_data[:-1]
                val_target_batch = batch_data[-1]
                
            
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
    
    if mode == 'Save':
        # Convert the error log to a DataFrame for further analysis
        errors_df = pd.DataFrame(errors_log)
        errors_df["error"] = errors_df["error"].apply(lambda x: x[0] if len(x) == 1 else x)
        print('Done training!')
    
        errors_df.to_csv(f'../official_results/error_logs/{model_name}_exp_{num}_split_{split_num}.csv',index=False)
    
        fig1.savefig(f"../official_results/learning_curves/{model_name}_exp_{num}_split_{split_num}.png", dpi=100) 
        plt.close(fig1)
    elif mode == 'Show':
        fig1.show()
        
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
    samples_str = f'{count_im} images, all augmented 5x' #count_images(train_dataset,augment)

    details = {'Experiment':num,'RMSE':rmse_test,'Variables':variables, 'Split_id': split_num,'Optimizer': opt, \
               'nº samples': samples_str, 'Batch size': batch_size, \
               'Epochs': f'{num_final_epochs} of {epochs}','Date':current_date,'Time':current_time, 'Duration': duration, \
               'Loss':  loss_type, 'Resolution':W}

    if mode == 'Save':
        print('SAVING MODEL AND RESULTS!')
        file_path = f"../official_results/{model_name}_results.xlsx"
        save_excel(file_path, details, excel = 'Results')
        model.save(f'../models/{model_name}.h5')
    elif mode == 'Show':
        print(details)

    input_shape = train_model_input[0].shape
    
    if conditioned:
        print(f"Experiment {model_name} with batch_size={batch_size} and epochs={epochs} completed and {input_shape, train_additional_inputs.shape} inputs .\n")
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
    resolutions = args.resolution
    loss_type = args.loss_type[0]
    
    print(inputs)
    if inputs[0] == 'full features':
        inputs = ['lst', 'ndvi','slope','altitude','direction','month','coords','discharge']
    elif inputs[0] == 'image features':
        inputs = ['lst', 'ndvi','slope','altitude','direction']
        
        
    filt_alt = False 
    augment = True
    epochs=300
    
    # Run experiments with parameter combinations
    for model_name in model_names:
        exp_num = get_next_experiment_number(f'../official_results/{model_name}_results.xlsx')
        for batch_size in batch_sizes:
            for W in resolutions:
                #for c in [True, False]:
                for split in range(1,6):
                    data_folder = f'../data/processed_data/{W}x{W}/{split}'
                    run_experiment(data_folder, model_name, batch_size, epochs, W=W, inputs=inputs, \
                                       loss_type=loss_type, filt_alt=filt_alt, num = exp_num, augment = augment, split_num = split, mode = 'Save')

    # python experiments_augmented.py --batch_sizes 128 --epochs_list 300 --model_names CNN_2 CNN_3 Resnet transfer_resnet --inputs 'full features' --resolution 128 --loss_type RMSE            
    # python experiments_augmented.py --batch_sizes 128 --epochs_list 300 --model_names CNN_2 CNN_3 Resnet transfer_resnet --inputs 'image features' --resolution 128 --loss_type RMSE 