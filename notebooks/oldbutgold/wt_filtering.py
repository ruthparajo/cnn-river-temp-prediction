import xarray as xr
import pandas as pd
import argparse
import geopandas as gpd
from shapely.geometry import Point
import pyproj

#archivo_nc = '../data/waterTemperature_monthly_1981-2014.nc'
archivo_nc = '../data/external/waterTemperature_monthlyAvg_1980_2019.nc'

ds = xr.open_dataset(archivo_nc)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process geospatial files.')
    
    # Add argument for the input file
    parser.add_argument(
        'input_file', 
        type=str, 
        help='Path to the input file (e.g., a shapefile or CSV)'
    )

    # Arguments for date or time interval
    parser.add_argument(
        '--start_date', 
        type=str, 
        help='Start date in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'
    )
    
    parser.add_argument(
        '--end_date', 
        type=str, 
        help='End date in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'
    )

    # Add argument for filtering by season
    parser.add_argument(
        '--season',
        type=str,
        choices=['autumn-winter', 'spring', 'summer'],
        help='Season to filter the data (autumn-winter, spring, or summer)'
    )

    # Add argument for filtering by range of years or year
    parser.add_argument(
        '--years', 
        nargs='+', 
        type=str,  # List of strings
        help='List of years or year range to filter the data (e.g., 2000 2005-2010)'
    )

    # Add argument for the output file
    parser.add_argument(
        'output_file', 
        type=str, 
        help='Path to the output file (e.g., to save the results)'
    )
    
    # Add optional argument for the output format
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['shp', 'geojson', 'csv'], 
        default='csv',
        help='File format for saving the output (default: csv)'
    )

    
    return parser.parse_args()

def parse_years(years):
    """Convierte una lista de años y rangos en una lista de años completos."""
    all_years = []
    for year in years:
        if '-' in year:
            start_year, end_year = map(int, year.split('-'))
            all_years.extend(list(range(start_year, end_year + 1)))
        else:
            all_years.append(int(year))
    return all_years

def filter_year(ds, years):
    # Convertir los años proporcionados a una lista completa de años si incluye rangos
    all_years = parse_years(years)
    
    # Filtrar los datos por los años seleccionados
    ds_filtered = ds.sel(time=ds['time.year'].isin(all_years))
    return ds_filtered

def filter_zone(ds,gdf):
    # Filtrar los puntos que están dentro de la región de interés
    gdf = gdf.to_crs("EPSG:4326")
    bounds = gdf.total_bounds
    lon_min, lat_min = (bounds[0], bounds[1])  # lat_min, lon_min
    lon_max, lat_max = (bounds[2], bounds[3])
    ds_filtered = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    return ds_filtered

def filter_season(ds, season):
    # Mapear las estaciones a los meses correspondientes
    if season == 'autumn-winter':
        months = [10, 11, 12, 1, 2, 3]  # Octubre-Marzo
    elif season == 'spring':
        months = [4, 5, 6]  # Abril-Junio
    elif season == 'summer':
        months = [7, 8, 9]  # Julio-Septiembre

    # Filtrar según los meses
    ds_filtered = ds.sel(time=ds['time.month'].isin(months))
    return ds_filtered


def filter_date(ds,start_date,end_date,season=None,years=None):
    if season:
        ds = filter_season(ds, season)
    if start_date or end_date:
        if start_date and end_date:
            ds = ds.sel(time=slice(start_date, end_date))
        elif end_date:
            ds = ds.sel(time=slice(None, end_date))
        elif start_date:
            ds = ds.sel(time=slice(start_date, None))
    if years:
        ds = filter_year(ds, years)
    return ds
    
def process(ds):
    df = ds.to_dataframe().reset_index()
    #df = df.rename(columns={'waterTemperature': 'waterTempK'})
    #df['waterTemperature'] = df['waterTempK'] - 273.15
    return df


def main():
    args = parse_arguments()
    
    # Load the input file into a GeoDataFrame
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    else:
        gdf = gpd.read_file(args.input_file)
    
    filtered_ds = filter_zone(ds,gdf)
    filtered_ds = filter_date(filtered_ds,args.start_date,args.end_date,args.season,args.years)
    df = process(filtered_ds)
    gdf_output = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
    
    # Save the file in the specified format
    if args.format == 'shp':
        gdf_output.to_file(f'../data/external/{args.output_file}', driver='ESRI Shapefile')
    elif args.format == 'geojson':
        gdf_output.to_file(f'../data/external/{args.output_file}', driver='GeoJSON')
    elif args.format == 'csv':
        #gdf_output['geometry_wkt'] = gdf['geometry'].apply(lambda x: x.wkt)
        gdf_output.to_csv(f'../data/{args.output_file}', index=False)

if __name__ == "__main__":
    main()