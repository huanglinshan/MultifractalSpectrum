#-*-coding:utf-8-*-
# -*- coding: UTF-8 -*-
# # Script to resample a raster to a specific size.
import os

from osgeo import gdal
from scipy.misc import imresize,imsave

class Resampler():
    '''
    Resample image
    '''
    def __init__(self, export_file_folder, file_name, out_column, out_row):
        self.export_file_folder = export_file_folder
        self.file_name = file_name
        # column number of resampled image.
        self.out_column = out_column
        # row number of resampled image.
        self.out_row = out_row

    def _read_file_with_resample(self):
        """
        Read file and resample
        """
        in_ds = gdal.Open(self.file_name)
        in_band = in_ds.GetRasterBand(1)
        data = in_band.ReadAsArray(buf_xsize=self.out_column, buf_ysize=self.out_row)

        del in_ds
        return data

    def resample_file(self):
        """
        ################################ MAIN FUNCTION #############################
        """
        # Open the input raster.
        resize_data = self._read_file_with_resample()
        self.resize_data = resize_data
        clip_name = os.path.join(self.export_file_folder, self.file_name.split('\\')[-1].split('.')[0] + '_clip.jpg')
        imsave(clip_name, self.resize_data*255)
