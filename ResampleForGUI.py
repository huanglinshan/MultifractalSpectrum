# -*- coding: UTF-8 -*-  
# # Script to resample a raster to a smaller pixel size.


import os
from osgeo import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables as tb
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import platform

def resample_file(in_fn, out_fn, out_columns, out_rows):
    # Open the input raster.
    in_ds = gdal.Open(in_fn)
    in_band = in_ds.GetRasterBand(1)

    # Create the output raster using the computed dimensions.
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(out_fn, out_columns, out_rows)

    # Change the geotransform so it reflects the smaller cell size before
    # setting it onto the output.
    out_ds.SetProjection(in_ds.GetProjection())
    geotransform = list(in_ds.GetGeoTransform())
    height_multiplier = in_band.YSize/float(out_rows)
    width_multiplier = in_band.XSize/float(out_columns)
    geotransform[1] *= width_multiplier
    geotransform[5] *= height_multiplier
    out_ds.SetGeoTransform(geotransform)

    # Read in the data, but have gdal resample it so that it has the specified
    # number of rows and columns instead of the numbers that the input has.
    # This effectively resizes the pixels.
    # data = in_band.ReadAsArray(buf_xsize=out_columns, buf_ysize=out_rows)
    data = in_band.ReadAsArray()
    sum_data = np.zeros((out_rows,out_columns),np.float64)
    win_size = (int(width_multiplier),int(height_multiplier))

    frame = np.zeros((win_size[1],win_size[0]),np.float64)

    for i in range(out_columns):
        for j in range(out_rows):
            frame = data[j*win_size[1]:(j+1)*win_size[1],i*win_size[0]:(i+1)*win_size[0]]
            sum_data[j,i] = np.sum(frame)        

    # if in the form of (shape = 1, blank = 0)
    # data

    # Write the data to the output raster.
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(sum_data)

    # Compute statistics and build overviews.
    out_band.FlushCache()
    out_band.ComputeStatistics(False)
    # out_ds.BuildOverviews('average', [2, 4, 8, 16, 32, 64])

    del out_ds
    del in_ds
    return sum_data

def make_resample_slices(data, win_size):
    """Return a list of resampled slices given a window size.

    data     - two-dimensional array to get slices from
    win_size - tuple of (rows, columns) for the input window
    """
    row = int(data.shape[0] / win_size[0]) * win_size[0]
    col = int(data.shape[1] / win_size[1]) * win_size[1]
    slices = []

    for i in range(win_size[0]):
        for j in range(win_size[1]):
            slices.append(data[i:row:win_size[0], j:col:win_size[1]])
    return slices

# Get all the .tif files in the folder
def file_name(file_dir, type):
    """Returns the lists of a specific type of files in the folder.

    file_dir  -  folder of dataset
    type - the file type (str)
    """
    sysstr = platform.system()
    if(sysstr =="Windows"):
        file_dir = file_dir.replace("/","\\")
        print ("The platform is Windows. Please use '\\' to split the path.")
    elif(sysstr == "Linux"):
        print ("The platform is Linux. Please use '/' to split the path.")
    else: # Other platform might get wrong!
        print ("Other System tasks, might be wrong!")

    L=[] 
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == type:  
                    L.append(os.path.join(root, file))  
    return L
            
def resample_folder(in_folder, out_folder, out_columns, out_rows):
    # Open the input folder.
    if in_folder is None:
        raise RuntimeError('Could not open datasource')
    in_ds_list = file_name(in_folder,'.tif')

    # Loop through the tif files in the directory.
    for i in range(len(in_ds_list)):
        in_fn = in_ds_list[i]
        in_ds = gdal.Open(in_fn)
        print('Resampling ' + in_fn + '...')

        sysstr = platform.system()
        if(sysstr =="Windows"):
            in_fn_without_dir = in_fn.split('\\')[-1]
            print ("The platform is Windows. Please use '\\' to split the path.")
        elif(sysstr == "Linux"):
            in_fn_without_dir = in_fn.split('/')[-1]
            print ("The platform is Linus. Please use '/' to split the path.")
        else: # Other platform might get wrong!
            print ("Other System tasks, might be wrong!")

        out_fn_ori = in_fn_without_dir[0:15]
        out_fn = out_fn_ori+'Resample'+str(out_columns) +'x'+str(out_rows)+'.tif'
        out_fn = os.path.join(out_folder,out_fn)
        resample_file(in_fn,out_fn,out_columns,out_rows)
        del in_ds
