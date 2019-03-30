#-*-coding:utf-8-*-

"""
    Author:       ls-huang
    Edition:      1.0
    Data:         2018/10/18
    File Name:    main.py
    Function：    main part

    Function：    Use .tif image file and calculate multifractal spectrum and visualize the spatial spectrum.
"""

import numpy as np
import winsound
import time
import os
import gc

# import relevant class defined in the project
from fileList import FileList
from clipRaster import Clipper
from resizeRaster import Resampler
from calculation import Calculator
from saveDataAndPlot import SaveDataAndPic

class Main():
    def __init__(self):
        """
            setting of the main parameters
        """
        # 1. the .tif file folder
        # use "\\" when in Windows platform and "/" in Linux
        self.filefolder = u'C:\\Tiff_File_Path'
        self.filetype = '.tif'
        filelist = FileList(self.filefolder, self.filetype)
        filelist.get_file_name()
        self.file_list = filelist.file_list

        # 2. the extent of the outermost box
        # upper-left coordinates and lower-right coordinates
        # [x: horizontal coordinate; y: vertical coordinate]
        self.box_ulx = 12950524
        self.box_uly = 4847567
        self.box_lrx = 12952284
        self.box_lry = 4846532
        # whether to be clipped
        # 0: not to be clipped
        # 1: would be clipped as coordinates set above
        self.is_to_be_clipped = 0
        # save path of clipped file
        self.clip_filefolder = u"C:\\Tiff_File_Path\\Clip"

        # 3. resample parameters
        # scale of resample size
        # CAUTION: must be the power of 2, e.g. 128, 256, 512, 1024, ... 4096)
        self.resample_size = 4096
        self.resample_filefolder = u"C:\\Tiff_File_Path\\resample"

        # 4. OLS regression parameters
        # from left to right: q_min, q_max, q_interval
        self.q_list = np.arange(-20, 21, 1)
        # 0 - Fix the intercept to zero during OLS
        # 1 - Do NOT Fix the intercept to zero during OLS
        self.intercept = 1

        # 5. save data tp .hdf5 and save pic to .png.
        # 1 - Export the data to .hdf5 file
        # 0 - Do NOT Export the data to .hdf5 file
        # self.is_export_data = 1
        # 1 - Export the result to .png file
        # 0 - Do NOT Export the result to .png file
        # self.is_export_png = 1
        # whether to save data and pic
        self.is_save_data = True
        self.export_file_folder = u'C:\\Tiff_File_Path\\Result'

        # 6. set the scaling range for different parameter
        ## scaling range for D_q and tau(q) during OLS estimation
        # minimum: 1
        self.dq_scale_min = 1
        # maximum
        self.dq_scale_max = 512
        dq_scale_ul = np.log2(self.dq_scale_max)
        dq_scale_dl = np.log2(self.dq_scale_min)

        ## scaling range for alpha(q) during OLS estimation
        # minimus: 1
        self.alpha_scale_min = 1
        # maximum
        self.alpha_scale_max = 512
        alpha_scale_ul = np.log2(self.alpha_scale_max)
        alpha_scale_dl = np.log2(self.alpha_scale_min)

        ## scaling range for alpha(q) during OLS estimation
        # minimus: 1
        self.f_scale_min = 1
        # maximum
        self.f_scale_max = 512
        f_scale_ul = np.log2(self.f_scale_max)
        f_scale_dl = np.log2(self.f_scale_min)

        # combined scaling range (would be distinguished by "label")
        self.scale_limit = [(dq_scale_dl, dq_scale_ul), (alpha_scale_dl, alpha_scale_ul), (f_scale_dl, f_scale_ul)]

    def start_to_calculate(self):
        print('Begin to calculate...')
        time.sleep(1)

        for i in range(len(self.file_list)):
            file_name = self.file_list[i]
            if self.is_to_be_clipped == 1:
                self.clip_box = [(self.box_ulx, self.box_uly), (self.box_lrx, self.box_lry)]
                print("Cliping..." + file_name)
                clipper = Clipper(file_name, self.clip_box, out_folder=os.getcwd())
                clipper.extract_subset_from_file()
                file_name = clipper.out_file_name
                print(file_name.split('\\')[-1] + "has been clipped.")

            print("Resampling..." + file_name.split('\\')[-1])
            resampler = Resampler(self.export_file_folder,file_name,self.resample_size,self.resample_size)
            resampler.resample_file()
            resize_data = resampler.resize_data

            print("Calculating..." + file_name.split('\\')[-1])
            calculator = Calculator(resize_data, self.q_list, self.scale_limit, self.intercept,
                                    self.resample_size, self.resample_size)
            calculator.calculation_and_save()
            prob_data_list = calculator.prob_data_list
            parameters = calculator.parameters
            parameters_calculation = calculator.parameters_calculation
            miu_q_list_list = calculator.miu_q_list_list
            data = calculator.data

            if self.is_save_data == True:
                print("Saving data and pic...")
                file_name_without_dir = file_name.split('\\')[-1].split('.')[0]
                fig_name_prefix = os.path.join(self.export_file_folder, file_name_without_dir)
                save_data_and_pic = SaveDataAndPic(prob_data_list, parameters, parameters_calculation, miu_q_list_list, data, self.q_list,
                                                   os.path.join(self.export_file_folder, file_name_without_dir + '.h5'),
                                                                self.scale_limit, self.intercept, fig_name_prefix)
                save_data_and_pic.save_to_hdf5()
                save_data_and_pic.plot_spectrum()
                print("Data and pic have been saved.")

        print("The calculation is done!\n")
        # Play music when finished.
        winsound.PlaySound("Bongos.wav", winsound.SND_ASYNC)

if __name__ == "__main__":
    gc.collect()
    main = Main()
    main.start_to_calculate()
