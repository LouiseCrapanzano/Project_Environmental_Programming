# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:46 2023

@author: louis, livacke
"""
# !pip install rasterio.mask

## Task 2 

# first import libraries
import pathlib
import geopandas as gpd
import pandas as pd
import zipfile
# list of subfolders (1ste column of the dataframe)
import os
import numpy as np
import rasterio
from rasterio.mask import mask as rmask
import matplotlib.pyplot as plt

# make dataframe 
satellite = pd.DataFrame() #currently empty
currentdir = input('What is the path of your data on your computer?\n')  # interacting with user

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

# Path_Liv = '/Users/livacke/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/3e Bachelor/Environmental Programming/Clipped_data'
# Path_Louis = r'C:\Users\louis\Downloads\EP_Project\Data' 

## Task 4 and Task 5

All_Band08 = list(satellite["Band08"])
print(All_Band08)

# Calculating TUR and SPM
def CalcRaster(A_p, C_p, Param, currentdir, Band, date, Display=False):
    path_shp = currentdir + '/scheldt_clip.shp'

    shapefile = gpd.read_file(path_shp)
    with rasterio.open(Band) as rho:
        
        # Define the target CRS you want to reproject to
        target_crs = 'EPSG:32631'  

        # Reproject the GeoDataFrame to the target CRS
        reprojected_shapefile = shapefile.to_crs(target_crs)
        
        # Extract the geometry of the reprojected shapefile
        geometry = [reprojected_shapefile.geometry.values[0]]

        # Ensure the geometry is a valid GeoJSON-like geometry
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
    print(Param + date)
    print(f"  Min value: {rho_w.min()}")
    print(f"  Max value: {rho_w.max()}")
    print(f"  Mean value: {rho_w.mean()}")
    print(f"  Std value: {rho_w.std()}")
    
    if Display:
        title = Param + "_Band08_" + date
        plt.imshow(np.squeeze(rho_w), cmap='gray')
        plt.title(title)
        plt.show()

    RasterData = np.zeros(np.shape(rho_w)) # Same shape as rho_w, first all zeros
    RasterData = A_p*rho_w/(1-rho_w/C_p) # Formula to calculate SPM and TUR
    
    # now save data
    
    band_meta = rho.meta  # Get metadata for the band
    out_meta = band_meta.copy() # copy structure of meta_data 
    out_meta.update({'driver':'GTiff',         # adapt value of variables to desired
                     'width':rho.shape[1],
                     'height':rho.shape[0],
                     'count':1,
                     'dtype':'float64',
                     'crs':rho.crs, 
                     'transform':rho.transform,
                     'nodata':0})
    
    path_out = currentdir + f'/{Param}/{Param}_{date}.tif' # path of where you want to save raster data
    with rasterio.open(fp=path_out, # outputpath_name
                  mode='w',**out_meta) as dst:
                  dst.write(RasterData, 1)

    return RasterData #output of function

# make the folders where RasterData is saved, if not created already (Task 5)
def CreateFolder(currentdir, Param):
    path_create = os.path.join(currentdir,Param)
    if not os.path.exists(path_create):
        os.makedirs(path_create) 
    
CreateFolder(currentdir, 'TUR')
CreateFolder(currentdir, 'SPM')
CreateFolder(currentdir, 'CHL')

for Band in All_Band08:
    path_folder = currentdir + '/R08_Bands/'
    path = os.path.join(path_folder,Band)
    # bereken TUR:
    A_p = 1602.93
    C_p = 0.19130
    Param = 'TUR'
    # tif = find_band08(folder) # get the right.tif file
    date = list(satellite.loc[satellite.Band08 == Band, 'date']) # find date of folder
    TUR_data = CalcRaster(A_p, C_p, Param, currentdir, path, date[0],True) # calculate and save TUR data

    # bereken nu  SPM:
    A_p = 1801.52
    C_p = 0.19130
    Param = 'SPM'
    # tif = find_band08(folder)
    # date = list(satellite.loc[satellite.filename == folder, 'date'])
    SPM_data = CalcRaster(A_p, C_p, Param, currentdir, path, date[0])