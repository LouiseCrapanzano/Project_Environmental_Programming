# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:46 2023

@author: louis, livacke
"""
## Task 2 

# first import libraries
import pathlib
import pandas as pd
import zipfile
# list of subfolders (1ste column of the dataframe)
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# make dataframe 
satellite = pd.DataFrame() #currently empty
currentdir = input('What is the path of your data on your computer?\n')  # interacting with user

items_dir = []
items = os.listdir(currentdir) # code looks inside current directory/folder and searches all other directories
for item in items:
    if item[-4:]=='.zip':      # code to unzip zipfiles if present
        
        path = os.path.join(currentdir, item)
        with zipfile.ZipFile(path,"r") as zip_ref:
            zip_ref.extractall(currentdir)
        os.remove(path)
    if (os.path.isdir(os.path.join(currentdir, item)) & (item[:2]=='S2')): #only look for folders with filename that start with 'S2'
        # print(item) #only print name not entire path 
        items_dir.append(item)
    
# put list of filenames in dataframe 
satellite = pd.DataFrame({"filename": (items_dir)})


# extract date from filename (2nd colomn dataframe)
satellite["date"]=satellite["filename"].str[11:19]

print(satellite)

# Path_Liv = '/Users/livacke/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/3e Bachelor/Environmental Programming/Clipped_data'
# Path_Louis = r'C:\Users\louis\Downloads\EP_Project\Data' 

## Task 4 and Task 5

folders = list(satellite["filename"])
print(folders)

# look for band08 in the dataframe
def find_band08(folder):
    path = os.path.join(currentdir,folder) # path of folder on computer
    folder_path = pathlib.Path(path)
    tif_files = (list(folder_path.rglob("*.tif"))) # regular expression: all files ending with .tif
    for tif in tif_files:
        if 'B08' in str(tif): # search files that contain 'B08' in their filename
            return tif

# Calculating TUR and SPM
def CalcRaster(A_p, C_p, Param, currentdir, folder, tif, date, Display=False):
    with rasterio.open(tif) as rho:

        rho_w = rho.read()[0]

        # Display a simple plot of the band if you put Display=True as input to this function
        if(Display):
            plt.imshow(rho_w)
            plt.title("Band08")
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
    
    path_out = currentdir + '/'+Param+'/'+Param+'_' + date +'.tif' # path of where you want to save raster data
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

for folder in folders:
    # bereken TUR:
    A_p = 1602.93
    C_p = 0.19130
    Param = 'TUR'
    tif = find_band08(folder) # get the right.tif file
    date = list(satellite.loc[satellite.filename == folder, 'date']) # find date of folder
    TUR_data = CalcRaster(A_p, C_p, Param, currentdir, folder, tif, date[0],True) # calculate and save TUR data

    # bereken nu  SPM:
    A_p = 1801.52
    C_p = 0.19130
    Param = 'SPM'
    tif = find_band08(folder)
    date = list(satellite.loc[satellite.filename == folder, 'date'])
    SPM_data = CalcRaster(A_p, C_p, Param, currentdir, folder, tif, date[0])
    