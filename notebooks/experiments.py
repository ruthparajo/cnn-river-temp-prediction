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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# -------------------------------- Load data --------------------------------
# Set variables
source_folder = '../data/external/raster_masks'
rivers = {}
source_path = '../data/preprocessed/'
data_paths = ['lst','slope', 'discharge', 'ndvi','altitude']#, 'ndvi', 'wt', 'masked','discharge', 'slope']#, 'wt_interpolated']
dir_paths = [os.path.join(source_path,p) for p in data_paths]
all_dir_paths = {k:[] for k in data_paths}    
total_data = {}
total_times = {}
complete_rivers = []
filter_river = None
W=256
time_split = False

# Load rivers
for subdir, dirs, files in os.walk(source_folder):
    for i,file in enumerate(files):
        r,m = load_raster(os.path.join(subdir, file), False)
        name = file.split('.')[0].split('bw_')[-1]
        rivers[name] = r

# Load input paths
for i,dir_p in enumerate(dir_paths):
    if data_paths[i] != 'discharge' and data_paths[i] != 'slope' and data_paths[i] != 'altitude':
        imgs_per_river = Counter()
    for subdir, dirs,files in os.walk(dir_p):
        if subdir != dir_p and not subdir.endswith('masked') and not subdir.endswith('.ipynb_checkpoints'): 
            all_dir_paths[data_paths[i]].append(subdir)
        elif subdir.endswith('masked') and 'masked' in data_paths:
            all_dir_paths['masked'].append(subdir)
        elif dir_p.endswith('altitude'):
            all_dir_paths[data_paths[i]].extend(files)
            
# Load input data
for k,v in all_dir_paths.items():
    if filter_river != None:
        v = [v[i] for i in filter_river]
    
    if k != 'discharge' and k != 'slope' and k != 'altitude':
        if k == 'lst' or k == 'masked':
            list_rgb = [True]*len(v)
        else:
            list_rgb = [False]*len(v)
            
        data, times = load_data(v,W,list_rgb)
        if k!='masked':
            labels = []
            for ki,value in data.items():
                labels+=[ki.split('/')[-1]]*len(value)
        
        data_values = [np.array(img) for sublist in list(data.values()) for img in sublist]
        times_list = [t for sublist in times for t in sublist]
   
        if time_split:
            dates = [datetime.strptime(date, '%Y-%m') for date in times_list]
            pairs = sorted(zip(dates, data_values, labels), key=lambda x: x[0])
            sorted_dates, data_values, labels = zip(*pairs)
            times_list = [date.strftime('%Y-%m') for date in sorted_dates]
            
        total_data[k] = np.array(data_values)
        total_times[k] = times_list
        print(k,':' ,total_data[k].shape)


for k,v in all_dir_paths.items():
    total = []
    if k == 'discharge' or k == 'slope' or k == 'altitude':
        imgss = {}
        for i,lab in enumerate(labels):
            for file in v:
                if lab == file.split('/')[-1] or lab == file.split('.')[0]:
                    if lab not in imgss.keys():
                        if k != 'altitude':
                            file_path = os.path.join(file,os.listdir(file)[0])
                        else:
                            file_path = os.path.join('../data/preprocessed/altitude', file)
                        
                        r,m = load_raster(file_path, False)
                        var = resize_image(r, W,W)
                        var = np.where(np.isnan(var), 0.0, var)
                        imgss[lab] = var
                    else:
                        var = imgss[lab]
                    total.append(var)
                        
        total_data[k] = np.array(total)
        print(k, np.array(total).shape)

# Load cos and sin variables
str_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# Load target variable
water_temp = pd.read_csv('../data/preprocessed/wt/water_temp.csv', index_col=0)
times_ordered = total_times['lst']
wt_temp=[]
for cell, date in zip(labels,times_ordered):
    temp = water_temp[(water_temp["Cell"] == cell) & (water_temp["Date"] == date)]["WaterTemp"]
    if not temp.empty:
        wt_temp.append(temp.values[0])
data_targets = np.array(wt_temp)


# -------------------------------- Finish loading data --------------------------------

def get_results(test_target, test_prediction, rivers, labels, test_index):
    mean_results = {k:[] for k in results.keys()}
    # Loop through each sample and compute the MSE for that sample
    for i in range(test_target.shape[0]):
        # Flatten the true and predicted values for this sample
        riv = rivers[labels[test_index[i]]].flatten()
        y_true_flatten = test_target[i].flatten()
        y_true_mask = y_true_flatten[riv != 0]
        y_pred_flatten = test_prediction[i].flatten()
        y_pred_mask = y_pred_flatten[riv != 0]
        # Calculate metrics
        res = evaluate_model(y_true_mask, y_pred_mask)
        for k,v in res.items():
            mean_results[k].append(v)
    for key in mean_results:
        mean_results[key] = np.mean(mean_results[key])
    return mean_results

def get_months_vectorized(times):
    month_cosine_dict = {}
    month_sinus_dict = {}
    cos_to_month = {}
    # Fill the dictionary with cosine values for each month
    for month in range(1, 13):
        # Scale the month from 0 to 1 (January as 0, December as 11/12)
        month_scaled = (month - 1) / 12
        # Convert the scaled month to radians and apply the cosine function
        month_cosine = np.cos(month_scaled * 2 * np.pi)
        month_sinus = np.sin(month_scaled * 2 * np.pi)
        # Store it in the dictionary
        month_cosine_dict[month] = month_cosine
        month_sinus_dict[month] = month_sinus
        cos_to_month[month_cosine]= month
       
    
    times_dt = pd.to_datetime(times)
    cosine_months = []
    sine_months = []
    for time_dt in times_dt:
        m = time_dt.month
        cos = month_cosine_dict[m]
        sin = month_sinus_dict[m]
        cosine_months.append(cos)
        sine_months.append(sin)
        
    cosine_months = np.array(cosine_months)
    sine_months = np.array(sine_months)
    
    return cosine_months, sine_months, cos_to_month
    


def run_experiment(model_name, batch_size, epochs, W=256, conditioned=False, inputs=None, stratified=None, physics_guided=None, time_split=None):
    print(f"Running experiment with model={model_name}, batch_size={batch_size}, epochs={epochs}, inputs = {inputs}")
    
    # Choose inputs
    inputs_d = [total_data[inp] for inp in inputs]
    # List to store the processed additional images
    expanded_images = []
    # Expand dimensions for single-channel images, leave multi-channel images as they are
    for img in inputs_d:
        if img.ndim == 3:  # Case where image is (n, 256, 256) (single-channel)
            expanded_images.append(np.expand_dims(img, axis=-1))  # Expand to add an extra channel
        elif img.ndim == 4:  # Case where image already has multiple channels (n, 256, 256, c)
            expanded_images.append(img)  # Leave the image as it is
    # Concatenate all images along the last axis (channels)
    combined_input = np.concatenate(expanded_images, axis=-1)
    # The final combined input is stored in input_data
    input_data = combined_input
    
    cosine_months, sine_months, cos_to_month = get_months_vectorized(total_times['lst'])
    additional_inputs = np.column_stack((cosine_months, sine_months))

    if time_split:
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        
        # Calcular el tamaño de cada conjunto
        total_images = len(input_data)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        indices = np.arange(total_images)
        
        train_index = indices[:train_size]                       # Primeros índices para entrenamiento
        validation_index = indices[train_size:train_size + val_size]    # Siguientes índices para validación
        test_index = indices[train_size + val_size:]             # Últimos índices para prueba
       
    elif stratified:
        train_index, validation_index, test_index = split_data_stratified(input_data, data_targets, labels)
    else:
        train_index, validation_index, test_index = split_data(input_data, data_targets)
            
    validation_input = input_data[validation_index, :] / 255.0  # Normalize inputs
    validation_target = data_targets[validation_index]
    test_input = input_data[test_index, :] / 255.0  # Normalize inputs
    test_target = data_targets[test_index]
    train_input = input_data[train_index, :] / 255.0  # Normalize inputs
    train_target = data_targets[train_index]
    
    additional_inputs_train = additional_inputs[train_index, :]
    additional_inputs_validation = additional_inputs[validation_index, :]
    additional_inputs_test = additional_inputs[test_index, :]

    
    if len(train_input.shape) == 3:
        input_shape = train_input.shape[1:]+(1,)
    else:
        input_shape = train_input.shape[1:]

    
    # Adapt input to condition
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
    start_time = time.time()
    if model_name == "baseline_CNN":
        if conditioned:
            model = build_cnn_model_features(input_args[0], input_args[1])
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
        model = build_simplified_cnn_model_improved(input_args)

    
    # Create batch dataset
    dataset = tf.data.Dataset.from_tensor_slices((*model_input, train_target) if conditioned else (model_input, train_target))
    dataset = dataset.batch(batch_size)

    dataset_val = tf.data.Dataset.from_tensor_slices((*val_model_input, validation_target) if conditioned else (val_model_input, validation_target))
    dataset_val = dataset_val.batch(batch_size)
    
    #optimizer = tf.keras.optimizers.SGD()
    optimizer = tf.keras.optimizers.Adam()
    errors_log = {"epoch": [], "month": [], "error": []}
    loss_per_epoch = []
    val_loss_per_epoch = []
    
    # Train model
    for epoch in range(epochs):
        epoch_loss = 0  
        num_batches = 0
        for batch in dataset:
            if conditioned:
                model_input_batch = batch[:-1]  
                target_batch = batch[-1]        
            else:
                model_input_batch, target_batch = batch  # Desempaquetado directo para un solo dataset
    
            with tf.GradientTape() as tape:
                y_pred = model([*model_input_batch], training=True) if conditioned else model(model_input_batch, training=True)
                if physics_guided:
                    loss = conservation_energy_loss(target_batch, y_pred, model_input_batch, alpha=0.5, beta=0.5)
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
            for i, (value1, value2) in enumerate(model_input_batch[1].numpy()):  
                error = tf.abs(y_pred[i] - y_true[i]).numpy()
                errors_log["epoch"].append(epoch + 1)
                errors_log["month"].append(cos_to_month[value1])
                errors_log["error"].append(error)
                
        avg_epoch_loss = epoch_loss / num_batches
        loss_per_epoch.append(avg_epoch_loss)
        val_loss = 0
        val_batches = 0
        for val_batch in dataset_val:
            if conditioned:
                val_input_batch = batch[:-1]  
                val_target_batch = batch[-1]        
            else:
                val_input_batch, val_target_batch = batch
            val_pred = model([*val_input_batch], training=False) if conditioned else model(val_input_batch, training=False)
            val_loss += root_mean_squared_error(val_target_batch, val_pred).numpy()
            val_batches += 1
            
        avg_val_loss = val_loss / val_batches
        val_loss_per_epoch.append(avg_val_loss)
    
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f}")



    # Convert the error log to a DataFrame for further analysis
    errors_df = pd.DataFrame(errors_log)
    errors_df["error"] = errors_df["error"].apply(lambda x: x[0] if len(x) == 1 else x)
    print('Done training!')

    num = len(os.listdir('../results/error_logs'))
    errors_df.to_csv(f'../results/error_logs/{model_name}_exp_{num}.csv',index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_per_epoch, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss_per_epoch, label='Validation Loss')
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fig1 = plt.gcf()
    fig1.savefig(f"../results/learning_curves/{model_name}_exp_{num}.png", dpi=100) 
    
    # Evaluate results
    test_prediction = model.predict(test_model_input)
    rmse_test = mean_squared_error(test_target, test_prediction, squared=False)
    
    print('\nComputing result metrics...')
    
    # Get experiment data
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Save model results
    laabeel = 'label, month' if conditioned else 'no label'
    var_inputs = '' if inputs == None else ', '.join(inputs)
    variables = ', '.join([var_inputs, laabeel])
    details = {'Experiment':num,'RMSE':rmse_test,'Variables':variables,'Input': f'{len(np.unique(labels))} rivers', 'Output': 'wt', \
               'Resolution': W, 'nº samples': len(data_targets), 'Batch size': batch_size, 'Epochs': epochs, 'Date':current_date, \
               'Time':current_time, 'Duration': duration, 'Loss':  'Physics-guided' if physics_guided else 'RMSE'}
    
    file_path = f"../results/{model_name}_results.xlsx"
    save_excel(file_path, details, excel = 'Results')
    
    print('RMSE:',rmse_test,'\n')
    print(f"Experiment {model_name} with batch_size={batch_size} and epochs={epochs} completed.\n")



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
    
    # Parse arguments
    args = parser.parse_args()

    # Extract argument values
    batch_sizes = args.batch_sizes
    epochs_list = args.epochs_list
    model_names = args.model_names
    inputs = args.inputs
    

    # Run experiments with parameter combinations
    for model_name in model_names:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                run_experiment(model_name, batch_size, epochs, W=256, conditioned=True, inputs=inputs, stratified=False, physics_guided=False, time_split = time_split)
                

