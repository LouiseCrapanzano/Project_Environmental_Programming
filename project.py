# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:46 2023

@author: louis
"""

#task 2 project
#eerst pandas importeren
import pandas as pd

#dataframe maken
satellite = pd.DataFrame() #momenteel leeg

#lijst van subfolders (1ste kolom van de dataframe)
import os

currentdir = r'C:\Users\louis\Downloads\EP_Project\Data' #r niet vergeten!

items = os.listdir(currentdir) #code gaat binnen current directory/folder op zoek naar alle andere directories
for item in items:
    if os.path.isdir(os.path.join(currentdir, item)): #je moet currentdir joinen met item naam want anders gaat code zeggen: dit is geen directory
        print(item) #enkel naam van de folder printen en niet heel de path (dus met currentdir)

#list van filenames in dataframe zetten
satellite = pd.DataFrame({"filename": (items)})

#datum extracten uit filename (2de kolom dataframe)
satellite["date"]=satellite["filename"].str[11:19]





