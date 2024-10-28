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


# -------------------------------- Load data --------------------------------
# Set variables
source_folder = '../data/external/raster_masks'
rivers = {}
source_path = '../data/preprocessed/'
data_paths = ['lst','wt','masked','slope', 'discharge']#, 'ndvi', 'wt', 'masked','discharge', 'slope']#, 'wt_interpolated']
dir_paths = [os.path.join(source_path,p) for p in data_paths]
all_dir_paths = {k:[] for k in data_paths}    
total_data = {}
total_times = {}
complete_rivers = []
filter_river = None
W=256

# Load rivers
for subdir, dirs, files in os.walk(source_folder):
    for i,file in enumerate(files):
        r,m = load_raster(os.path.join(subdir, file), False)
        name = file.split('.')[0].split('bw_')[-1]
        rivers[name] = r

# Load input paths
for i,dir_p in enumerate(dir_paths):
    for subdir, dirs,files in os.walk(dir_p):
        if subdir != dir_p and not subdir.endswith('masked') and not subdir.endswith('.ipynb_checkpoints'): 
            all_dir_paths[data_paths[i]].append(subdir)
        elif subdir.endswith('masked'):
            all_dir_paths['masked'].append(subdir)

# Load input data
for k,v in all_dir_paths.items():
    if filter_river != None:
        v = [v[i] for i in filter_river]
    
    if k != 'discharge' and k != 'slope':
        if k == 'lst' or k == 'masked':
            list_rgb = [True]*len(v)
        else:
            list_rgb = [False]*len(v)

            labels = []
            for ki,value in data.items():
                labels+=[ki.split('/')[-1]]*len(value)
        
        data, times = load_data(v,W,list_rgb)
        
        filtered = [arr for arr in data.values() if arr.size > 0]

        total_data[k] = np.concatenate(filtered, axis=0)
        total_times[k] = times
        print(k,':' ,total_data[k].shape)

    elif k == 'discharge' or k == 'slope':
        total = []
        for p in v:
            for file in os.listdir(p):
                file_path = os.path.join(p, file)
                r,m = load_raster(file_path, False)
                var = resize_image(r, W,W)
                img_river = labels.count(p.split("/")[-1])
                var_input = np.tile(var, (img_river, 1, 1))
                total.append(var_input)
        
        total_data[k] = np.concatenate(total, axis=0)
        print(k,':' ,total_data[k].shape)

# Hot encoding
encoder = OneHotEncoder(sparse_output=False)
river_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1))
data_targets = total_data['wt']
results = {'MAE':0,'MSE':0,'RMSE':0,'R²':0,'MAPE (%)':0,'MSE sample-wise':0}

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


def run_experiment(model_name, batch_size, epochs, W=256, conditioned=False, inputs=None, stratified=None, physics_guided=None):
    print(f"Running experiment with model={model_name}, batch_size={batch_size}, epochs={epochs}")
    
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

    
    # Split data    
    if stratified:
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
    
    
    if len(train_input.shape) == 3:
        input_shape = train_input.shape[1:]+(1,)
    else:
        input_shape = train_input.shape[1:]

    
    # Adapt input to condition
    if conditioned:
        input_args = (input_shape, river_encoded.shape[1])
        model_input = [train_input, train_rivers]
        val_model_input = [validation_input, validation_rivers]
        test_model_input = [test_input, test_rivers]
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
               'Time':current_time, 'Duration': duration, 'Loss': 'Physics-guided'}
    
    file_path = f"../results/{model_name}_results.xlsx"
    save_excel(file_path, details, excel = 'Results')
    
    mean_results['Model'] = model_name
    file_path = f"../results/all_results.xlsx"
    save_excel(file_path, mean_results, excel = 'Results')
    
    print(f"Experiment {model_name} with batch_size={batch_size} and epochs={epochs} completed.\n")


if '__main__':
    batch_sizes = [16, 32]
    epochs_list = [10, 50, 100]
    model_names = ['img_wise_CNN', 'UNet', 'CNN', 'img_2_img']
    inputs = [d for d in data_paths if d not in ['wt', 'masked']]

    for model_name in model_names:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                run_experiment(model_name, batch_size, epochs, W=256, conditioned=False, inputs=inputs, stratified = False, physics_guided = True)
                #if model_name == 'img_wise_CNN':
                 #   run_experiment(model_name, batch_size, epochs, W=256, conditioned=True, inputs=inputs, stratified = True, physics_guided = True)
                

