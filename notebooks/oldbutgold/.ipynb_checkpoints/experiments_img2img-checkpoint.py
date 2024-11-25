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


# -------------------------------- Load data --------------------------------
# Set variables
source_folder = '../data/external/raster_masks'
rivers = {}
source_path = '../data/preprocessed/'
data_paths = ['lst','wt_interpolated','masked','slope', 'discharge', 'ndvi','altitude']#, 'ndvi', 'wt', 'masked','discharge', 'slope']#, 'wt_interpolated']
dir_paths = [os.path.join(source_path,p) for p in data_paths]
all_dir_paths = {k:[] for k in data_paths}    
total_data = {}
total_times = {}
complete_rivers = []
filter_river = None
W=256
time_split = True

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
                        imgss[lab] = var
                    else:
                        var = imgss[lab]
                    total.append(var)
                        
        total_data[k] = np.array(total)
        print(k, np.array(total).shape)

# Hot encoding
encoder = OneHotEncoder(sparse_output=False)
river_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1))
data_targets = total_data['wt_interpolated']
results = {'MAE':0,'MSE':0,'RMSE':0,'R²':0,'MAPE (%)':0,'MSE sample-wise':0}
str_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

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
    return cosine_months, sine_months
    



def run_experiment(model_name, batch_size, epochs, W=256, conditioned=False, inputs=None, stratified=None, physics_guided=None):
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
    
    cosine_months, sine_months = get_months_vectorized(total_times['lst'])

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
    validation_target = data_targets[validation_index, :]
    validation_rivers = river_encoded[validation_index, :]
    test_input = input_data[test_index, :] / 255.0  # Normalize inputs
    test_target = data_targets[test_index, :]
    test_rivers = river_encoded[test_index, :]
    train_input = input_data[train_index, :] / 255.0  # Normalize inputs
    train_target = data_targets[train_index, :]
    train_rivers = river_encoded[train_index, :]
    
    train_cos_months = cosine_months[train_index] 
    train_sin_months = sine_months[train_index]
    val_cos_months = cosine_months[validation_index]
    val_sin_months = sine_months[validation_index]
    test_cos_months = cosine_months[test_index]
    test_sin_months = sine_months[test_index]

    
    if len(train_input.shape) == 3:
        input_shape = train_input.shape[1:]+(1,)
    else:
        input_shape = train_input.shape[1:]

    
    # Adapt input to condition
    if conditioned:
        input_args = (input_shape, river_encoded.shape[1])
        model_input = [train_input, train_rivers, train_cos_months, train_sin_months]
        val_model_input = [validation_input, validation_rivers, val_cos_months, val_sin_months]
        test_model_input = [test_input, test_rivers,  test_cos_months, test_sin_months]
    else:
        input_args = input_shape
        model_input = train_input
        val_model_input = validation_input
        test_model_input = test_input

    
    # Start model
    start_time = time.time()
    if model_name == "img_wise_CNN":
        if conditioned:
            model = build_simplified_cnn_model_label(input_args[0], input_args[1])
        else:
            model = build_simplified_cnn_model(input_args)
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
        
    
    
    # Train the model
    if not physics_guided:
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        history = model.fit(model_input, train_target, batch_size=batch_size, epochs=epochs, validation_data=(val_model_input, validation_target))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((*model_input, train_target) if isinstance(model_input, tuple) else (model_input, train_target))

        dataset = dataset.batch(batch_size)
        optimizer = tf.keras.optimizers.Adam()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for batch in dataset:
                # Handle batch based on whether model_input is a tuple or a single dataset
                if isinstance(model_input, tuple):
                    model_input_batch = batch[:-1]  # All except the last element (target_batch)
                    target_batch = batch[-1]        # Last element is target_batch
                else:
                    model_input_batch, target_batch = batch  # Direct unpacking for single dataset

                with tf.GradientTape() as tape:
                    y_pred = model(*model_input_batch, training=True) if isinstance(model_input_batch, tuple) else model(model_input_batch, training=True)
                    loss = conservation_energy_loss(target_batch, y_pred, model_input_batch, alpha=0.5, beta=0.5)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    # Evaluate results
    #validation_prediction = model.predict(val_model_input)
    test_prediction = model.predict(test_model_input)
    mean_results = get_results(test_target, test_prediction, rivers, labels, test_index)

    # Get experiment data
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    
    # Save model results
    laabeel = 'label' if conditioned else 'no label'
    var_inputs = '' if inputs == None else ', '.join(inputs)
    variables = ', '.join([var_inputs, laabeel])
    details = {'RMSE':mean_results['RMSE'],'Variables':variables,'Input': f'{len(np.unique(labels))} rivers', 'Output': 'wt', \
               'Resolution': W, 'nº samples': len(data_targets), 'Batch size': batch_size, 'Epochs': epochs, 'Date':current_date, \
               'Time':current_time, 'Duration': duration, 'Loss': 'Physics-guided' if physics_guided else 'RMSE'}
    
    file_path = f"../results/{model_name}_results.xlsx"
    save_excel(file_path, details, excel = 'Results')
    
    mean_results['Model'] = model_name
    file_path = f"../results/all_results.xlsx"
    save_excel(file_path, mean_results, excel = 'Results')
    
    print(f"Experiment {model_name} with batch_size={batch_size} and epochs={epochs} completed.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to run experiments with deep learning models.")

    # Define command-line arguments
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32],
                        help="List of batch sizes for training the model. Example: --batch_sizes 16 32")
    parser.add_argument('--epochs_list', nargs='+', type=int, default=[10, 50, 100],
                        help="List of epochs for training the model. Example: --epochs_list 10 50 100")
    parser.add_argument('--model_names', nargs='+', type=str, default=['img_wise_CNN', 'UNet', 'CNN', 'img_2_img'],
                        help="List of model names. Example: --model_names img_wise_CNN UNet CNN img_2_img")
    parser.add_argument('--inputs', nargs='+', type=str, required=True,
                        help="List of inputs to include . Example: --inputs lst ndvi")

    # Parse arguments
    args = parser.parse_args()

    # Extract argument values
    batch_sizes = args.batch_sizes
    epochs_list = args.epochs_list
    model_names = args.model_names
    inputs = args.inputs

    # Filter inputs as needed
    #inputs = [d for d in inputs if d not in ['wt', 'masked']]

    # Run experiments with parameter combinations
    for model_name in model_names:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                run_experiment(model_name, batch_size, epochs, W=256, conditioned=False, inputs=inputs, stratified=False, physics_guided=True)
                # Additional condition for 'img_wise_CNN'
                if model_name == 'img_wise_CNN':
                    run_experiment(model_name, batch_size, epochs, W=256, conditioned=True, inputs=inputs, stratified=False, physics_guided=True)

'''
if '__main__':
    batch_sizes = [16, 32]
    epochs_list = [10, 50]
    model_names = ['UNet', 'CNN', 'img_2_img','img_wise_CNN','img_wise_CNN_improved']
    inputs = [d for d in data_paths if d not in ['wt_interpolated', 'masked','slope', 'discharge', 'altitude']]
    inputs_comb = [[d for d in data_paths if d not in ['wt_interpolated', 'masked','slope', 'discharge']],[d for d in data_paths if d not in ['wt_interpolated', 'masked','ndvi']],[d for d in data_paths if d not in ['wt_interpolated', 'masked']], [d for d in data_paths if d not in ['wt_interpolated', 'masked','slope', 'discharge','ndvi']]]
    model_name = 'img_wise_CNN'
    
    for model_name in model_names:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                #for ph in [True, False]:
                run_experiment(model_name, batch_size, epochs, W=256, conditioned=False, inputs=inputs, stratified = False, physics_guided = False)
                if model_name == 'img_wise_CNN':
                    run_experiment(model_name, batch_size, epochs, W=256, conditioned=True, inputs=inputs, stratified = False, physics_guided = False)'''
                

