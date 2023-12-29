# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:46 2023

@author: louis, livacke
"""

## Task 2 

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
        # print(item) #only print name not entire path
            items_R08=sorted(os.listdir(path))
        # items_dir.append(item)\=
        if(item[:2]=='S2'):
            items_dir.append(item)
            items_dir = sorted(items_dir)
    
# put list of filenames in dataframe 
satellite = pd.DataFrame({"filename": (items_dir)})

satellite["Band08"]=items_R08
# extract date from filename (2nd colomn dataframe)
satellite["date"]=satellite["filename"].str[11:19]

print(satellite)

# Path_Liv = /Users/livacke/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/3e Bachelor/Environmental Programming/Clipped_data
# Path_Louis = r'C:/Users/louis/Downloads/EP_Project/Data' 
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

        # Ensure the geometry is a valid GeoJSON-like geometry
        if not geometry[0].is_valid:
            geometry[0] = geometry[0].buffer(0)
        
        # Check and fix the validity again after buffering
        if not geometry[0].is_valid:
            geometry[0] = geometry[0].buffer(0)
            
        # Clip GeoTIFF with shapefile
        clipped_data, out_transform = rmask(rho, geometry, crop=True)

        # Update metadata
        out_meta = rho.meta
        out_meta.update({"driver": "GTiff", "height": clipped_data.shape[1], "width": clipped_data.shape[2],
                 "transform": out_transform})
        

        # Write the clipped data to a new GeoTIFF file
        # with rasterio.open(output_path, 'w', **meta) as dst:
        #     dst.write(clipped_R08)

        # Read the clipped data
        rho_w_unfiltered = clipped_data[0, :, :]
    
    condition = ((rho_w_unfiltered != 0) & (rho_w_unfiltered != 32767) & (rho_w_unfiltered > 0))
    # condition = rho_w_unfiltered > 0
    rho_w = np.ma.masked_array(rho_w_unfiltered, mask=~condition)*1/10000
    
    # Display a simple plot of the band if you put Display=True as input to this function
    # print('Rho_w' + date)
    # print(f"  Min value: {rho_w.min()}")
    # print(f"  Max value: {rho_w.max()}")
    # print(f"  Mean value: {rho_w.mean()}")
    # print(f"  Std value: {rho_w.std()}")
    
    RasterData_unfiltered = np.zeros(np.shape(rho_w)) # Same shape as rho_w, first all zeros
    RasterData_unfiltered = A_p*rho_w/(1-rho_w/C_p) # Formula to calculate SPM and TUR
    condition = ((RasterData_unfiltered != 65535) & (RasterData_unfiltered > 0) & (RasterData_unfiltered < 100))
    RasterData = np.ma.masked_array(RasterData_unfiltered, mask=~condition)
    
    # out_meta = out_meta.copy() # copy structure of meta_data 
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
    # now save data
    # band_meta = rho.meta  # Get metadata for the band
    
    path_out = currentdir + f'/{Param}/{Param}_{date}.tif' # path of where you want to save raster data
    with rasterio.open(fp=path_out, # outputpath_name
                  mode='w',**out_meta) as dst:
                  dst.write(RasterData, 1)
    
def PlotRaster(RasterData, Param, date, Display):
    if Display:
            title = Param + "_Band08_" + date[0]
            cmap = plt.get_cmap('rainbow')  # Colormap
            norm = mcolors.Normalize(vmin=RasterData.min(), vmax=100)
            plt.imshow(np.squeeze(RasterData), cmap=cmap, norm=norm)
            plt.title(title)
     
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Turbidity [FNU]')
     
            plt.show()
            
def ZonalStatistics(RasterData):
    Min_value = RasterData.min()
    Max_value = RasterData.max()
    Mean_value = RasterData.mean()
    Std_value = RasterData.std()
    return Mean_value


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
    
    # bereken TUR:
    A_p = 1602.93
    C_p = 0.19130
    Param = 'TUR'
    # tif = find_band08(folder) # get the right.tif file
    date = list(satellite.loc[satellite.Band08 == Band, 'date']) # find date of folder
    TUR_data, out_meta = CalcRaster(A_p, C_p, Param, currentdir, path, date[0],True) # calculate and save TUR data
    SaveRaster(TUR_data, out_meta, currentdir, Param, date[0])
    PlotRaster(TUR_data, Param, date[0], True)
    
    # bereken nu  SPM:
    A_p = 1801.52
    C_p = 0.19130
    Param = 'SPM'
    # tif = find_band08(folder)
    # date = list(satellite.loc[satellite.filename == folder, 'date'])
    SPM_data, out_meta = CalcRaster(A_p, C_p, Param, currentdir, path, date[0],True)
    SaveRaster(SPM_data, out_meta, currentdir, Param, date[0])
    PlotRaster(SPM_data, Param, date[0], True)
    
## Task 6

def custom_zonal_stats(vector_path, tif_path, stats, nodata):
    result = rasterstats.zonal_stats(vectors=vector_path, raster=tif_path, stats=stats, nodata=nodata)
    return result

vector_path = currentdir + '/reprojected_shapefile.shp'
stats = ['min', 'max', 'mean', 'std', 'median']
nodata = 65535

gdf = gpd.read_file(vector_path)

tif_folder = currentdir + '/TUR'
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

for tif_file in tif_files:
    tif_path = os.path.join(tif_folder, tif_file)

    prefix = 'TUR'
    result = custom_zonal_stats(gdf, tif_path, stats, nodata)
    print(f"Vector Path: {vector_path}")
    print(f"Raster Path: {tif_path}")
    print(f"Stats: {stats}")
    # print(f"Prefix: {prefix}")
    print(f"Nodata: {nodata}")
    print(f"Intermediate Result: {result}")


## Task 8

years = satellite['date']
mean_SPM = satellite['mean_SPM']
mean_TUR = satellite['mean_TUR']

satellite['date'] = pd.to_datetime(satellite['date'])
# Haal het jaartal uit de 'date' kolom
satellite['year'] = satellite['date'].dt.year

# functie om de beide parameters te plotten over de jaren heen
def plot_mean(satellite, column, label):
    plt.plot(satellite.index, satellite[column], label=label)
    plt.xlabel('Years')
    plt.ylabel(f'Mean for {column}')
    plt.legend()
    plt.title(f'Mean for {column} over the years')
    plt.show()

# Plot voor Mean for SPM
plot_mean(satellite, 'mean_SPM', 'Mean for SPM')

# Plot voor Mean for TUR
plot_mean(satellite, 'mean_TUR', 'Mean for TUR')