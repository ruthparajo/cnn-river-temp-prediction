import rasterio
import pandas as pd
import os
import re
import subprocess
import numpy as np
from skimage.transform import resize
import openpyxl
from openpyxl import load_workbook
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rasterio.transform import from_origin
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import geopandas as gpd
import tensorflow as tf
import cv2
import json


def load_raster(filepath,rgb = True):
  try:
      with rasterio.open(filepath) as src:
        if rgb:
          red = src.read(1)
          green = src.read(2)
          blue = src.read(3)
          rgb = np.dstack((red, green, blue))
          return rgb, src
        else:
          return src.read(1), src
  except rasterio.errors.RasterioIOError:
    print(f"{filepath} no es pot obrir com a TIFF.")
      
def resize_image(image, new_width, new_height):
    return resize(image, (new_height, new_width), preserve_range=True)

def extract_year_month_from_filename(filename):
    # Function to extract the year and month from the filename
    # Adjust regex to match the YYYY-MM format
    match = re.search(r'(\d{4})-(\d{2})', filename)
    if match:
        return match.group(0)  # Return the matched year and month as a string (YYYY-MM)
    return None

def load_data(dir_paths, W=None, rgb=[], show=None):
    data = {}
    time_slots = []
    for i, path in enumerate(dir_paths):
        image_files = os.listdir(path)

        # Filter out files without a valid date in the filename
        valid_image_files = [img for img in image_files if extract_year_month_from_filename(img) is not None]

        # Sort the valid image files by year and month extracted from the filename
        valid_image_files = sorted(valid_image_files, key=lambda x: extract_year_month_from_filename(x))
        
        times = []
        imgs_data = []
        for img_file in valid_image_files:
            imagen_path = os.path.join(path, img_file)
            try:
                img_data, meta = load_raster(imagen_path, rgb[i])  # Load raster data
    
                if W:  # Resize the image if W is provided
                    img_data = resize_image(img_data, W, W)
    
                imgs_data.append(img_data)
                times.append(extract_year_month_from_filename(img_file))
            except:
                pass
            
        data[path] = np.array(imgs_data)
        time_slots.append(times)

    return data, time_slots

def split_data(X, y, validation_split=0.1, test_split=0.1):
    """
    Splits the data into training, validation, and test sets.

    Parameters:
    X (np.array): Input data (e.g., images).
    y (np.array): Target data (e.g., temperature values).
    validation_split (float): Proportion of data to use for validation set.
    test_split (float): Proportion of data to use for test set.

    Returns:
    train_input (np.array): Training data inputs normalized.
    train_target (np.array): Training data targets.
    validation_input (np.array): Validation data inputs normalized.
    validation_target (np.array): Validation data targets.
    test_input (np.array): Test data inputs normalized.
    test_target (np.array): Test data targets.
    """

    n = X.shape[0]  # Number of samples
    rem_index = np.arange(n)  # Array of all indices

    # Select validation indices
    validation_index = np.random.choice(rem_index, int(validation_split * n), replace=False)
    rem_index = list(set(rem_index) - set(validation_index))

    # Select test indices from remaining
    test_index = np.random.choice(rem_index, int(test_split * n), replace=False)
    train_index = list(set(rem_index) - set(test_index))  # Remaining for training

    return train_index, validation_index, test_index#train_input, train_target, validation_input, validation_target, test_input, test_target

def evaluate_model(y_true, y_pred):
    """
    Returns useful metrics for evaluating a linear regression model for temperature prediction.

    Parameters:
    y_true: array-like, true temperature values.
    y_pred: array-like, predicted temperature values.

    Returns:
    dict containing all useful metrics.
    """

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R-squared (R²)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE)
    epsilon = 1e-10  # Small value to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # Mean Squared Error (MSE) sample-wise
    sample_wise_mse = []
    # Loop through each sample and compute the MSE for that sample
    for i in range(y_true.shape[0]):
        # Flatten the true and predicted values for this sample
        y_true_flatten = y_true[i].flatten()
        y_pred_flatten = y_pred[i].flatten()
        # Calculate MSE for this sample
        mse = mean_squared_error(y_true_flatten, y_pred_flatten)
        sample_wise_mse.append(mse)
    
    average_mse = np.mean(sample_wise_mse)

    # Create a dictionary with the metrics
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
        'MSE sample-wise': average_mse
    }

    return metrics

def update_excel_with_results(file_path, model_name, metrics, details):
    # Check if the Excel file exists
    if not os.path.exists(file_path):
        # Create a new Excel workbook and add a sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Model Results"
        # Write the header based on the metric keys
        header = ["Model Name"] + list(metrics.keys()) + ["Details"]
        sheet.append(header)
    else:
        # Load existing Excel workbook
        workbook = load_workbook(file_path)
        sheet = workbook.active

    # Find the next empty row
    next_row = sheet.max_row + 1
    
    # Prepare the row with model name, metrics, and details
    row = [model_name] + list(metrics.values()) + [details]
    
    # Append the row data to the next available row
    for col_num, value in enumerate(row, start=1):
        sheet.cell(row=next_row, column=col_num, value=value)
    
    # Save the Excel file
    workbook.save(file_path)

def save_excel(file_path, dicc, excel=None):
    # Check if the Excel file exists
    if not os.path.exists(file_path):
        # Create a new Excel workbook and add a sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        if excel == 'Results':
            sheet.title = "Model Results"
        else:
            sheet.title = "Model Details"
        # Write the header based on the metric keys
        header = list(dicc.keys())
        sheet.append(header)
    else:
        # Load existing Excel workbook
        workbook = load_workbook(file_path)
        sheet = workbook.active

    # Find the next empty row
    next_row = sheet.max_row + 1
    
    # Prepare the row with model name, metrics, and details
    row = list(dicc.values())
    
    # Append the row data to the next available row
    for col_num, value in enumerate(row, start=1):
        sheet.cell(row=next_row, column=col_num, value=value)
    
    # Save the Excel file
    workbook.save(file_path)

def split_data_df(X, y, validation_split=0.1, test_split=0.1):
    n = X.shape[0]  # Número de muestras
    rem_index = np.arange(n)  # Array de todos los índices

    # Seleccionar índices de validación
    validation_index = np.random.choice(rem_index, int(validation_split * n), replace=False)
    rem_index = list(set(rem_index) - set(validation_index))

    # Seleccionar índices de prueba de los restantes
    test_index = np.random.choice(rem_index, int(test_split * n), replace=False)
    train_index = list(set(rem_index) - set(test_index))  # El resto para entrenamiento

    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    validation_input = X.iloc[validation_index, :].values / 255.0  # Normalizar inputs si es necesario
    validation_target = y.iloc[validation_index].values

    test_input = X.iloc[test_index, :].values / 255.0  # Normalizar inputs
    test_target = y.iloc[test_index].values

    train_input = X.iloc[train_index, :].values / 255.0  # Normalizar inputs
    train_target = y.iloc[train_index].values

    return train_input, train_target, validation_input, validation_target, test_input, test_target

def plot_image(image, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def clear_directory(directory):
    # Function to clear the contents of a directory
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}. Reason: {e}')

def save_raster(raster_array,filepath,shp):
    resolution = 30
    x_min, y_min, x_max, y_max = shp.total_bounds
    transform = from_origin(x_min, y_max, resolution, resolution)

    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=raster_array.shape[0],
        width=raster_array.shape[1],
        count=len(raster_array.shape),
        dtype=raster_array.dtype,
        crs=shp.crs.to_string(),  # Ensure correct CRS
        transform=transform,
        nodata=0.0
    ) as dst:
        if len(raster_array.shape) == 3:
            dst.write(raster_array[:, :, 0], 1)  # Red channel
            dst.write(raster_array[:, :, 1], 2)  # Green channel
            dst.write(raster_array[:, :, 2], 3)  # Blue channel
        else:
            dst.write(raster_array, 1)

def distance_matrix(x0, y0, x1, y1):
    """
    Calculate distance matrix.
    Note: from <http://stackoverflow.com/questions/1871536>
    """

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    # calculate hypotenuse
    return np.hypot(d0, d1)

def simple_idw(x, y, z, xi, yi, beta=2, dist_matrix=None):
    """
    Simple inverse distance weighted (IDW) interpolation
    x`, `y`,`z` = known data arrays containing coordinates and data used for interpolation
    `xi`, `yi` =  two arrays of grid coordinates
    `beta` = determines the degree to which the nearer point(s) are preferred over more distant points.
            Typically 1 or 2 (inverse or inverse squared relationship)
    """
    if dist_matrix is None:
        dist_matrix = distance_matrix(x, y, xi, yi)
    
    # Calculate weights using inverse distance with exponent `beta`
    weights = dist_matrix ** (-beta)
    weights /= weights.sum(axis=0) + 1e-12  # Normalizing weights

    weights = np.nan_to_num(weights, nan=0.0)
    z = np.nan_to_num(z, nan=0.0)

    return np.dot(weights.T, z)


    '''dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    # weights = 1.0/(dist+1e-12)**power
    weights = dist ** (-beta)
    
    # Make weights sum to one
    weights /= weights.sum(axis=0)

    weights = np.nan_to_num(weights, nan=0.0)
    z = np.nan_to_num(z, nan=0.0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)'''

def project_linestrings_to_points(gdf,show=None):
    x_coords = []
    y_coords = []
    z_coords = []

    # Extract coordinates from LineStrings and store them as points
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            for coord in row.geometry.coords:
                x_coords.append(coord[0])
                y_coords.append(coord[1])
                if len(coord) == 3:
                    z_coords.append(coord[2])

        elif geom.geom_type == 'MultiLineString':
            # Si es MultiLineString, recorre cada LineString
            for linestring in geom.geoms:
                for coord in linestring.coords:
                    x_coords.append(coord[0])
                    y_coords.append(coord[1])
                    if len(coord) == 3:
                        z_coords.append(coord[2])

    if show:
        # Plot the points on a 2D grid
        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, color='blue', s=10, label='Coordinates')
    
        # Customize the plot
        plt.title('Projected Coordinates (Points) in EPSG:2056', fontsize=15)
        plt.xlabel('Easting (meters)', fontsize=12)
        plt.ylabel('Northing (meters)', fontsize=12)
    
        # Add gridlines
        plt.grid(True)
    
        # Show the plot
        plt.legend()
        plt.show()
    return x_coords, y_coords, z_coords

def split_data_stratified(inputs, data_targets, labels):
    # 1. Obtener las etiquetas únicas
    unique_labels, label_indices = np.unique(labels, return_inverse=True)
    
    # 2. Hacer una división estratificada en función de las etiquetas únicas
    train_label_idx, temp_label_idx = train_test_split(unique_labels, test_size=0.4, random_state=42)
    
    # Dividir el conjunto temporal (val y test)
    val_label_idx, test_label_idx = train_test_split(temp_label_idx, test_size=0.5, random_state=42)
    
    # 3. Crear máscaras booleanas en base a las etiquetas
    train_indices = np.where(np.isin(labels, train_label_idx))[0]
    val_indices = np.where(np.isin(labels, val_label_idx))[0]
    test_indices = np.where(np.isin(labels, test_label_idx))[0]


    return train_indices, val_indices, test_indices

def load_river_raster(filepath):
    r, m = load_raster(filepath, False)
    name = filepath.split('/')[-1].split('.')[0].split('bw_')[-1]
    return name, r

def get_rivers_altitude(source_folder, filt):
    rivs=[]
    all_x =[]
    all_y=[]
    all_z=[]
    with rasterio.open('../../../../../data/simon.walther/swissAltitude/swissAltitude.vrt') as src:
        for f in os.listdir(source_folder):
            if os.path.join(source_folder, f).endswith('shp'):
                river = gpd.read_file(os.path.join(source_folder, f))
                x_coords, y_coords, z_coords = project_linestrings_to_points(river)
                
                if z_coords == []:
                    dir_altitudes = '../data/external/altitudes'
                    
                    for i in range(len(x_coords)):
                        lon, lat = x_coords[i], y_coords[i]
                        row, col = src.index(lon, lat)
                        window = Window(col_off=col, row_off=row, width=1, height=1)
                        pixel_value = src.read(1, window=window)
                        z_coords.append(pixel_value[0][0])
                       
                if filt and np.mean(z_coords) <= 800:
                    rivs.append(f.split('station_')[-1].split('.')[0])
                    all_x.append(x_coords)
                    all_y.append(y_coords)
                    all_z.append(z_coords)
                elif not filt:
                    rivs.append(f.split('station_')[-1].split('.')[0])
                    all_x.append(x_coords)
                    all_y.append(y_coords)
                    all_z.append(z_coords)
                    
                
    return rivs, all_x, all_y, all_z

# Function to compute Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap.
    Args:
        img_array: Input image array (batch of shape (1, height, width, channels)).
        model: The model to analyze.
        last_conv_layer_name: The name of the last convolutional layer.
        pred_index: Index of the target class for Grad-CAM.
    Returns:
        A heatmap of the Grad-CAM.
    """
    print('holaaaa',img_array.shape)
    
    last_deep_layer = model.layers[-1].name
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.get_layer(last_deep_layer).output]
    )
    print(grad_model.summary())

    # Record gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        target_output = preds[:,0]
    print('Intento predir', target_output)
    # Compute the gradients
    grads = tape.gradient(target_output, last_conv_layer_output)
    print(grads)
    # Compute the mean intensity of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by the pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to visualize Grad-CAM overlay
def display_grad_cam(img, heatmap, alpha=0.4):
    """
    Combines the original image with the generated heatmap.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

def save_and_display_gradcam(img, heatmap, filename, alpha=0.4):
    """
    Saves and displays a Grad-CAM image.
    Args:
        img: The original image (H x W x C).
        heatmap: The heatmap generated by Grad-CAM.
        filename: File path to save the output image.
        alpha: Transparency factor for overlay.
    """
    # Resize heatmap to match input image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # Save the output image
    cv2.imwrite(filename, superimposed_img)
    print(f"Saved Grad-CAM to {filename}")

def save_grad_map(val_input_gradients, image_path,impact_channel):
    grad_map = val_input_gradients[0, :, :, impact_channel].numpy()  # Extraer gradiente de un canal
    
    plt.imshow(grad_map, cmap="bwr")
    plt.colorbar()
    plt.title(f"Impacto del canal {impact_channel} en validación")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_all_data(
    source_folder='../data/external/shp/river_cells_oficial',
    source_path='../data/preprocessed/',
    data_paths=['lst', 'slope', 'discharge', 'ndvi', 'altitude'],
    filter_altitude=None,
    W=256,
    time_split=False,
):
    
    dir_paths = [os.path.join(source_path, p) for p in data_paths]
    
    all_dir_paths = {k: [] for k in data_paths}
    total_data = {}
    total_times = {}

    rivers = get_rivers_altitude(source_folder,filter_altitude)[0]
    
    # Cargar rutas de entrada
    for i, dir_p in enumerate(dir_paths):
        for subdir, dirs, files in os.walk(dir_p):
            if subdir != dir_p and not subdir.endswith('masked') and not subdir.endswith('.ipynb_checkpoints') and subdir.split('/')[-1] in rivers:
                all_dir_paths[data_paths[i]].append(subdir)
            elif dir_p.endswith('altitude'):
                all_dir_paths[data_paths[i]].extend([f for f in files if f.split('.')[0] in rivers])
      
    # Cargar datos de entrada
    
    for k, v in all_dir_paths.items():
        if k not in ['direction', 'slope', 'altitude']:
            labels = []
            list_rgb = [True] * len(v) if k in ['lst', 'masked'] else [False] * len(v)
            data, times = load_data(v, W, list_rgb)
            if k != 'masked':
                for ki in data.keys():
                    labels += [ki.split('/')[-1]] * len(data[ki])

            data_values = [np.array(img) for sublist in list(data.values()) for img in sublist]
            times_list = [t for sublist in times for t in sublist]

            if time_split:
                dates = [datetime.strptime(date, '%Y-%m') for date in times_list]
                pairs = sorted(zip(dates, data_values, labels), key=lambda x: x[0])
                sorted_dates, data_values, labels = zip(*pairs)
                times_list = [date.strftime('%Y-%m') for date in sorted_dates]

            total_data[k] = np.array(data_values)
            total_times[k] = times_list
            print(f"{k} : {total_data[k].shape}")

    # Cargar variables adicionales
    for k, v in all_dir_paths.items():
        if k in ['direction', 'slope', 'altitude']:
            imgss = {}
            total = []
            for i, lab in enumerate(labels):
                for file in v:
                    if lab in file.split('/')[-1] or lab in file.split('.')[0]:
                        if lab not in imgss:
                            file_path = os.path.join(file, os.listdir(file)[0]) if k != 'altitude' else os.path.join('../data/preprocessed/altitude', file)
                            r, m = load_raster(file_path, False)
                            var = resize_image(r, W, W)
                            var = np.where(np.isnan(var), 0.0, var)
                            imgss[lab] = var
                        else:
                            var = imgss[lab]
                            
                total.append(var)

            total_data[k] = np.array(total)
            print(f"{k}: {np.array(total).shape}")

    # Cargar variable objetivo
    water_temp = pd.read_csv('../data/preprocessed/wt/water_temp.csv', index_col=0)
    times_ordered = total_times['lst']
    wt_temp = []
    for cell, date in zip(labels, times_ordered):
        temp = water_temp[(water_temp["Cell"] == cell) & (water_temp["Date"] == date)]["WaterTemp"]
        if not temp.empty:
            wt_temp.append(temp.values[0])
    data_targets = np.array(wt_temp)

    return total_data, total_times, data_targets, labels

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
    """
    Calculate the cosine and sine values for each month in a given list of dates.
    Additionally, return a dictionary that maps each unique cosine value to its corresponding month.
    
    Parameters:
    ----------
    times : list or array-like
        A list of date strings or datetime objects from which month information is extracted.
        
    Returns:
    -------
    cosine_months : np.ndarray
        An array of cosine values corresponding to each month in the `times` input.
        
    sine_months : np.ndarray
        An array of sine values corresponding to each month in the `times` input.
        
    cos_to_month : dict
        A dictionary where each key is a unique cosine value and the corresponding value is the month (1-12) 
        associated with that cosine value."""
    
    month_cosine_dict = {month: np.cos((month - 1) / 12 * 2 * np.pi) for month in range(1, 13)}
    month_sinus_dict = {month: np.sin((month - 1) / 12 * 2 * np.pi) for month in range(1, 13)}
    
    cos_to_month = {cos_val: month for month, cos_val in month_cosine_dict.items()}
    
    def cosine_for_month(month):
        return month_cosine_dict[month]

    def sine_for_month(month):
        return month_sinus_dict[month]

    cosine_vectorized = np.vectorize(cosine_for_month)
    sine_vectorized = np.vectorize(sine_for_month)

    times_dt = pd.to_datetime(times)
    months = times_dt.month

    cosine_months = cosine_vectorized(months)
    sine_months = sine_vectorized(months)
    
    return cosine_months, sine_months, cos_to_month

def get_lat_lon(labels):
    file_path = '../data/raw/wt/cell_coordinates_oficial.csv'
    lat_lon = pd.read_csv(file_path)
    lats=[]
    lons=[]
    for label in labels:
        num_cell = int(label.split('_')[-1])
        lat = lat_lon[lat_lon.Cell==num_cell].Latitude.values[0]
        lon = lat_lon[lat_lon.Cell==num_cell].Longitude.values[0]
        lats.append(lat)
        lons.append(lon)
    lats = np.array(lats)
    lons = np.array(lons)
    return lats, lons

def get_split_index(split, input_data, data_targets, labels, split_id=None,filt_alt=None):
    if split == 'time':
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
       
    elif split == 'stratified':
        print('yes !!!!!!!!!!')
        split_folder = f"../data/external/splits_low_alt/split_{split_id}_indices.json" if filt_alt else \
                        f"../data/external/splits/split_{split_id}_indices.json"
        with open(split_folder, 'r') as f:
            loaded_indices = json.load(f)
        train_index = loaded_indices['train']
        validation_index = loaded_indices['val']
        test_index = loaded_indices['test']
        #train_index, validation_index, test_index = split_data_stratified(input_data, data_targets, labels)
    else:
        train_index, validation_index, test_index = split_data(input_data, data_targets)
        
    return train_index, validation_index, test_index
            
def get_discharge(labels, times_ordered):
    discharge = pd.read_csv('../data/preprocessed/discharge/discharge.csv', index_col=0)
    disch = []
    for cell, date in zip(labels, times_ordered):
        value = discharge[(discharge["Cell"] == cell) & (discharge["Date"] == date)]["Discharge"]
        if not value.empty:
            disch.append(value.values[0])
    disch = np.array(disch)
    return disch

