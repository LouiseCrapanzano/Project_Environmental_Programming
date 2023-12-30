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
# function to calculate zonal statistics using rasterstats.zonal_stats
def custom_zonal_stats(vector_path, tif_path, stats, prefix, nodata):
    result = rasterstats.zonal_stats(vectors=vector_path, raster=tif_path, stats=stats, prefix=prefix, nodata=nodata)
    return result

# constant values for both parameters (TUR and SPM)
vector_path = currentdir + '/reprojected_shapefile.shp'
stats = ['min', 'max', 'mean', 'std', 'median']
nodata = 65535

gdf = gpd.read_file(vector_path)

# calculating zonal statistics for every tif file in TUR folder using a for loop
tif_folder_TUR = currentdir + '/TUR'
tif_files_TUR = [f for f in os.listdir(tif_folder_TUR) if f.endswith('.tif')]

for tif_file_TUR in tif_files_TUR:
    tif_path = os.path.join(tif_folder_TUR, tif_file_TUR)
    prefix = 'TUR'
    result = custom_zonal_stats(gdf, tif_path, stats, prefix, nodata)
    print(f"Zonal statistics for {tif_file_TUR}: {result}")

# calculating zonal statistics for every tif file in SPM folder using a for loop
tif_folder_SPM = currentdir + '/SPM'
tif_files_SPM = [f for f in os.listdir(tif_folder_SPM) if f.endswith('.tif')]

for tif_file_SPM in tif_files_SPM:
    tif_path = os.path.join(tif_folder_SPM, tif_file_SPM)
    prefix = 'SPM'
    result = custom_zonal_stats(gdf, tif_path, stats, prefix, nodata)
    print(f"Zonal statistics for {tif_file_SPM}: {result}")


# Create 2 list of dictionaries from zonal stats (1 for TUR, 1 for SPM)
#TUR
results_list_TUR = [
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 10.525367290375188, 'TURstd': 15.007504252561962, 'TURmedian': 6.715951318910258},
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 7.008311438179179, 'TURstd': 12.968962695433968, 'TURmedian': 2.7494138465189875},
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 19.744980209635163, 'TURstd': 19.86631586763264, 'TURmedian': 19.794267692821368},
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 22.869519741495395, 'TURstd': 23.09241296062801, 'TURmedian': 26.846298115974992},
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 6.7305991093977156, 'TURstd': 9.860765562508538, 'TURmedian': 4.225041459459459},
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 19.61000536651641, 'TURstd': 19.545095548126554, 'TURmedian': 19.976024250000002},
    {'TURmin': 0.0, 'TURmax': 99.87598006237008, 'TURmean': 11.000622355221449, 'TURstd': 12.131615657528783, 'TURmedian': 9.928996513761469},
]

#SPM
results_list_SPM = [
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 11.41650600720615, 'SPMstd': 15.822073933103752, 'SPMmedian': 7.548003106837608},
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 7.377600785178969, 'SPMstd': 12.926593405598942, 'SPMmedian': 2.906743498154982},
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 20.924180634269295, 'SPMstd': 20.449331662834624, 'SPMmedian': 21.635154271111112},
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 23.783256306621794, 'SPMstd': 23.77214478032368, 'SPMmedian': 28.898218044192635},
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 7.373834421625951, 'SPMstd': 10.268240053375964, 'SPMmedian': 4.748489759406465},
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 20.919317537171043, 'SPMstd': 20.296808850522744, 'SPMmedian': 22.042569098998886},
    {'SPMmin': 0.0, 'SPMmax': 99.9266579096426, 'SPMmean': 12.173073488460528, 'SPMstd': 13.051696058022559, 'SPMmedian': 10.967214554476806},
    ]

# Extract values for multiple keys into a dictionary
#TUR
key_value_dict_TUR = {key: [entry[key] for entry in results_list_TUR] for key in ['TURmin', 'TURmax', 'TURmean', 'TURstd', 'TURmedian']}

#SPM
key_value_dict_SPM = {key: [entry[key] for entry in results_list_SPM] for key in ['SPMmin', 'SPMmax', 'SPMmean', 'SPMstd', 'SPMmedian']}

# Print the resulting dictionaries
print(key_value_dict_TUR)
print(key_value_dict_SPM)

# Create DataFrames from the dictionaries
df_to_add_TUR = pd.DataFrame(key_value_dict_TUR)
df_to_add_SPM = pd.DataFrame(key_value_dict_SPM)

# Concatenate the existing DataFrame with the new DataFrames
satellite = pd.concat([satellite, df_to_add_TUR, df_to_add_SPM], axis=1)

# Print the resulting DataFrame
print(satellite)

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