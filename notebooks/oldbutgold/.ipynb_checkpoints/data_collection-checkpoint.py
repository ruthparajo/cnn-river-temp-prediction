from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest, SHConfig
import pandas as pd
import datetime
import os 
import numpy as np
import geopandas as gpd
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import LineString, box, Point

config = SHConfig()
config.instance_id = '186865be-334c-4a79-9035-75e69a871c01'
config.sh_client_id = '9d5fe07a-4d4d-4ceb-bef8-25411cd0349e'
config.sh_client_secret =  'YvYHw9VOktdPdRETOVMGJAcYg6USMjE1'


def first_day(month, year):
    return datetime.date(year, month, 1)

def last_day(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - datetime.timedelta(days=next_month.day)

def find_closest_point(points_gdf, quadrant_bbox):
    lat_min, lon_min, lat_max, lon_max = quadrant_bbox
    
    # Calcular el centro del cuadrante
    center_x = (lat_min + lat_max) / 2
    center_y = (lon_min + lon_max) / 2
    center_point = Point(center_y, center_x)
    # Calculate distances from the center of the quadrant to all points
    distances = points_gdf.geometry.apply(lambda p: p.distance(center_point))
    # Find the minimum distance
    min_distance = distances.min()
    # Filter points that match the minimum distance
    closest_points = points_gdf[distances == min_distance]
    return closest_points

def rasterize_linestrings(lines, transform, out_shape):
    # Convertir a geometría de raster
    shapes = ((geom, 1) for geom in lines)
    mask = rasterize(shapes, out_shape=out_shape, transform=transform)
    return mask

def get_data_request(time_interval, evalscript, bbox, size, data_type, folder):
    if data_type == 'lst':
        responses = [SentinelHubRequest.output_response('default', MimeType.TIFF)] 
    elif data_type == 'ndvi':
        responses = [
            SentinelHubRequest.output_response('default', MimeType.TIFF),
            SentinelHubRequest.output_response('ndvi_image', MimeType.PNG)]
        
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.LANDSAT_OT_L1,
                time_interval=time_interval,
                mosaicking_order='leastCC',
                maxcc=0.1
            )
        ],
        responses=responses,
        bbox=bbox,
        size=size,
        data_folder=folder,
        config=config
    )

def adjust_size(size, max_size=2500):
    """ Adjusts the size so that none of the axes exceed max_size, maintaining the aspect ratio """
    width, height = size
    max_dimension = max(width, height)

    if max_dimension > max_size:
        scale_factor = max_size / max_dimension
        width = int(width * scale_factor)
        height = int(height * scale_factor)

    return width, height

def get_data(shp, evalscript, time_intervals, data_type, folder):
    coords_wgs84 = list(shp.total_bounds)#[lon_min, lat_min, lon_max, lat_max]
    if data_type == 'lst':  # Assuming 'lst' uses thermal bands B10 or B11
        resolution = 100  # Set to 100 meters for thermal bands
    else:
        resolution = 30  # Default to 30 meters for other bands

    #resolution = 30
    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    # extract the size based on bbx and the resolution
    size = bbox_to_dimensions(bbox, resolution=resolution)

    size = adjust_size(size)

    # create a list of requests
    list_of_requests = [get_data_request(slot, evalscript, bbox, size, data_type, folder) for slot in time_intervals]
    list_of_requests = [request.download_list[0] for request in list_of_requests]

    # download data with multiple threads
    data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)

