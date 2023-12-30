# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:46 2023

@author: louis, livacke
"""

# Task 2 

# first import libraries
import geopandas as gpd
import pandas as pd
import zipfile
# list of subfolders (1st column of the dataframe)
import os
import numpy as np
import rasterio
from rasterio.mask import mask as rmask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterstats

# make dataframe 
satellite = pd.DataFrame() #currently empty
currentdir = input('What is the path to your unzipped file (Clipped data) on your computer?\n')  # interacting with user

items_dir = []
items = os.listdir(currentdir) # code looks inside current directory/folder and searches all other directories
for item in items:
    path = os.path.join(currentdir, item)
    if item[-4:]=='.zip':      # code to unzip zipfiles if present
        with zipfile.ZipFile(path,"r") as zip_ref:
            zip_ref.extractall(currentdir)
        os.remove(path)
    if (os.path.isdir(os.path.join(currentdir, item))):
        
        if(item[:3]=='R08'): #only look for folders with filename that start with 'R08'
            items_R08=sorted(os.listdir(path))
        if(item[:2]=='S2'):
            items_dir.append(item)
            items_dir = sorted(items_dir)
    
# put list of filenames in dataframe 
satellite = pd.DataFrame({"filename": (items_dir)})

satellite["Band08"]=items_R08
# extract date from filename (2nd colomn dataframe)
satellite["date"]=satellite["filename"].str[11:19]

satellite = satellite.sort_values("date")

print(satellite)

# Path_Liv = /Users/livacke/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/3e Bachelor/Environmental Programming/Clipped_data
# Path_Louis = C:\Users\louis\Downloads\EP_Project\Data\Clipped_data
# Path_Alex = /Users/alexsamyn/Library/CloudStorage/OneDrive-Gedeeldebibliotheken-VrijeUniversiteitBrussel/Liv Acke - Environmental Programming/Clipped_data

## Task 4 and Task 5

All_Band08 = list(satellite["Band08"])

# Calculating TUR and SPM
def CalcRaster(A_p, C_p, Param, currentdir, Band, date, Display=False):
    path_shp = currentdir + '/reprojected_shapefile.shp'

    shapefile = gpd.read_file(path_shp)
    
    with rasterio.open(Band) as rho:
        
        # Define the target CRS you want to reproject to
        target_crs = 'EPSG:32631'  

        # Reproject the GeoDataFrame to the target CRS
        reprojected_shapefile = shapefile.to_crs(target_crs)
        output_path = currentdir + "/reprojected_shapefile.shp"
        reprojected_shapefile.to_file(output_path)
        
        # Extract the geometry of the reprojected shapefile
        geometry = [reprojected_shapefile.geometry.values[0]]
            
        # Clip GeoTIFF with shapefile
        clipped_data, out_transform = rmask(rho, geometry, crop=True)

        # Update metadata
        out_meta = rho.meta
        out_meta.update({"driver": "GTiff", "height": clipped_data.shape[1], "width": clipped_data.shape[2],
                 "transform": out_transform})

        # Read the clipped data
        rho_w_unfiltered = clipped_data[0, :, :]
    
    condition = ((rho_w_unfiltered != 0) & (rho_w_unfiltered != 32767) & (rho_w_unfiltered > 0))
    rho_w = np.ma.masked_array(rho_w_unfiltered, mask=~condition)*1/10000
    
    
    RasterData_unfiltered = np.zeros(np.shape(rho_w)) # Same shape as rho_w, first all zeros
    RasterData_unfiltered = A_p*rho_w/(1-rho_w/C_p) # Formula to calculate SPM and TUR
    condition = ((RasterData_unfiltered != 65535) & (RasterData_unfiltered > 0) & (RasterData_unfiltered < 100))
    RasterData = np.ma.masked_array(RasterData_unfiltered, mask=~condition)
    
    out_meta.update({'driver':'GTiff',         # adapt value of variables to desired
                     'width':rho.shape[1],
                     'height':rho.shape[0],
                     'count':1,
                     'dtype':'float64',
                     'crs':rho.crs, 
                     'transform':rho.transform,
                     'nodata':0})
    
    return RasterData, out_meta

# save data (task 5)
def SaveRaster(RasterData, out_meta, currentdir, Param, date):    
    path_out = currentdir + f'/{Param}/{Param}_{date}.tif' # path of where you want to save raster data
    with rasterio.open(fp=path_out, # outputpath_name
                  mode='w',**out_meta) as dst:
                  dst.write(RasterData, 1)
    
def PlotRaster(RasterData, Param, date, Display):
    if Display:
            title = Param + ' ' + date
            cmap = plt.get_cmap('rainbow')  # Colormap
            norm = mcolors.Normalize(vmin=RasterData.min(), vmax=100)
            plt.imshow(np.squeeze(RasterData), cmap=cmap, norm=norm)
            plt.title(title)
     
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Turbidity [FNU]')
     
            plt.show()      

# make the folders where RasterData is saved, if not created already (Task 5)
def CreateFolder(currentdir, Param):
    path_create = os.path.join(currentdir,Param)
    if not os.path.exists(path_create):
        os.makedirs(path_create) 
    
CreateFolder(currentdir, 'TUR')
CreateFolder(currentdir, 'SPM')


for Band in All_Band08:
    path_folder = currentdir + '/R08_Bands/'
    path = os.path.join(path_folder,Band)
    
    # Update expected dimensions based on clipped_data
    
    # calculate TUR:
    A_p = 1602.93
    C_p = 0.19130
    Param = 'TUR'
    date = list(satellite.loc[satellite.Band08 == Band, 'date']) # find date of folder
    TUR_data, out_meta = CalcRaster(A_p, C_p, Param, currentdir, path, date[0],True) # calculate and save TUR data
    SaveRaster(TUR_data, out_meta, currentdir, Param, date[0])
    PlotRaster(TUR_data, Param, date[0], True)
    
    # constant variables for both parameters (TUR and SPM)
    
    # calculate SPM:
    A_p = 1801.52
    C_p = 0.19130
    Param = 'SPM'
    SPM_data, out_meta = CalcRaster(A_p, C_p, Param, currentdir, path, date[0],True)
    SaveRaster(SPM_data, out_meta, currentdir, Param, date[0])
    PlotRaster(SPM_data, Param, date[0], True)
    
## Task 6
# Define the function to calculate zonal statistics using rasterstats.zonal_stats
def custom_zonal_stats(vector_path, tif_path, stats, prefix, nodata):
    # Calculate zonal statistics and convert the result to a GeoDataFrame
    result = rasterstats.zonal_stats(vectors=vector_path, raster=tif_path, stats=stats, prefix=prefix, nodata=nodata, geojson_out=True)
    geostats = gpd.GeoDataFrame.from_features(result)
    return geostats

# Define parameters
Params = ['TUR','SPM']
vector_path = currentdir + '/reprojected_shapefile.shp'
stats = ['min', 'max', 'mean', 'std', 'median']
nodata = 65535
geodataframes_list_TUR = []
geodataframes_list_SPM = []

# Loop through each parameter (TUR, SPM)
for Param in Params:
    tif_folder = currentdir + '/' + Param
    tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')])
    
    # Loop through each TIFF file for the current parameter
    for tif_file in tif_files:
        tif_path = os.path.join(tif_folder, tif_file)
        
        # Calculate zonal statistics for the current parameter and TIFF file
        geostats = custom_zonal_stats(vector_path, tif_path, stats, Param, nodata)
        
        # Select relevant columns from the result (excluding the first two columns)
        start_column_index = 2
        selected_geostats = geostats.iloc[:, start_column_index:]
        
        # Append the selected zonal statistics to the appropriate list based on the parameter
        if(Param == 'TUR'):
            geodataframes_list_TUR.append(selected_geostats)
        elif(Param == 'SPM'):
            geodataframes_list_SPM.append(selected_geostats)

# Concatenate the zonal statistics GeoDataFrames for TUR and SPM
concatenated_geodataframe_TUR = gpd.GeoDataFrame(pd.concat(geodataframes_list_TUR, ignore_index=True))
concatenated_geodataframe_SPM = gpd.GeoDataFrame(pd.concat(geodataframes_list_SPM, ignore_index=True))

# Concatenate the satellite GeoDataFrame with the zonal statistics for TUR and SPM
concatenated_geodataframes = gpd.GeoDataFrame(pd.concat([satellite, concatenated_geodataframe_TUR], axis=1))
satellite = gpd.GeoDataFrame(pd.concat([concatenated_geodataframes, concatenated_geodataframe_SPM], axis=1))

# Print the final GeoDataFrame
print(satellite)

# Task 8
# Getting the columns out of the DataFrame
years = satellite['date']
SPM_mean = satellite['SPMmean']
TUR_mean = satellite['TURmean']

satellite['date'] = pd.to_datetime(satellite['date'])
# Getting the year out of the date
satellite['year'] = satellite['date'].dt.year

# Function to plot both parameters over the years
def plot_mean(satellite, column, label, color):
    plt.plot(satellite['year'], satellite[column], label=label, color=color)
    plt.xlabel('Years')
    plt.ylabel(f'Mean for {column}')
    plt.legend()
    plt.title(f'Mean for {column} over the years') 
    plt.xticks(satellite['year'])
    plt.show()

# Plot Mean for TUR
plot_mean(satellite, 'TURmean', 'Mean for TUR', color='red')

# Plot Mean for SPM
plot_mean(satellite, 'SPMmean', 'Mean for SPM', color='blue')