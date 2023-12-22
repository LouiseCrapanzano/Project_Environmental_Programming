# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:46 2023

@author: louis, livacke
"""
## Task 2 

#eerst importeren
import pathlib
import pandas as pd
import zipfile
#lijst van subfolders (1ste kolom van de dataframe)
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

#dataframe maken
satellite = pd.DataFrame() #momenteel leeg
# currentdir = input('What is the path of your data on your computer?\n')
currentdir = '/Users/livacke/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/3e Bachelor/Environmental Programming/Clipped_data'

items_dir = []
items = os.listdir(currentdir) #code gaat binnen current directory/folder op zoek naar alle andere directories
for item in items:
    if item[-4:]=='.zip':
        
        path = os.path.join(currentdir, item)
        with zipfile.ZipFile(path,"r") as zip_ref:
            zip_ref.extractall(currentdir)
        os.remove(path)
    if (os.path.isdir(os.path.join(currentdir, item)) & (item[:2]=='S2')): #je moet currentdir joinen met item naam want anders gaat code zeggen: dit is geen directory
        # print(item) #enkel naam van de folder printen en niet heel de path (dus met currentdir)
        items_dir.append(item)
    
#list van filenames in dataframe zetten
satellite = pd.DataFrame({"filename": (items_dir)})


#datum extracten uit filename (2de kolom dataframe)
satellite["date"]=satellite["filename"].str[11:19]

print(satellite)

# Path_Liv = '/Users/livacke/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/3e Bachelor/Environmental Programming/Clipped_data'
# Path_Louis = r'C:\Users\louis\Downloads\EP_Project\Data' 

## Task 4 en Task 5

folders = list(satellite["filename"])
print(folders)

#band08 zoeken in de dataframe
def find_band08(folder):
    path = os.path.join(currentdir,folder)
    folder_path = pathlib.Path(path)
    tif_files = (list(folder_path.rglob("*.tif"))) # regular expression: alles dat eindigt op .tif
    for tif in tif_files:
        if 'B08' in str(tif):
            return tif

#Calculating TUR en SPM
def CalcRaster(A_p, C_p, Param, currentdir, folder, tif, date, Display=False):
    with rasterio.open(tif) as rho:

        # Get metadata for the band
        band_meta = rho.meta
        band_data = rho.read()[0]

        # Display a simple plot of the band
        if(Display):
            plt.imshow(band_data, cmap='gray')
            plt.title("Band08")
            plt.show()

        rho_w = band_data # np.array(band_data)
    
    RasterData = np.zeros(np.shape(rho_w))
    RasterData = A_p*rho_w/(1-rho_w/C_p)
    
    #nu data opslaan
    out_meta = band_meta.copy()

    out_meta.update({'driver':'GTiff',
                     'width':rho.shape[1],
                     'height':rho.shape[0],
                     'count':1,
                     'dtype':'float64',
                     'crs':rho.crs, 
                     'transform':rho.transform,
                     'nodata':0})
    
    path = currentdir + '/'+Param+'/'+Param+'_' + date +'.tif'
    with rasterio.open(fp=path, #outputpath_name
                  mode='w',**out_meta) as dst:
                  dst.write(RasterData, 1)

    return RasterData

#maak folders voor raster lagen in op te slaan
def CreateFolder(currentdir, Param):
    path = os.path.join(currentdir,Param)
    if not os.path.exists(path):
        os.makedirs(path) 
    
CreateFolder(currentdir, 'TUR')
CreateFolder(currentdir, 'SPM')
CreateFolder(currentdir, 'CHL')

for folder in folders:
    #bereken TUR:
    A_p = 1602.93
    C_p = 0.19130
    Param = 'TUR'
    tif = find_band08(folder)
    date = list(satellite.loc[satellite.filename == folder, 'date'])
    TUR_data = CalcRaster(A_p, C_p, Param, currentdir, folder, tif, date[0])

    #bereken nu  SPM:
    A_p = 1801.52
    C_p = 0.19130
    Param = 'SPM'
    tif = find_band08(folder)
    date = list(satellite.loc[satellite.filename == folder, 'date'])
    SPM_data = CalcRaster(A_p, C_p, Param, currentdir, folder, tif, date[0])
    