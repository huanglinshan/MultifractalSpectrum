# -*- coding: UTF-8 -*-  
# # Script to resample a raster to a smaller pixel size.

import os
from osgeo import gdal
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from SaveDataAndPNGForGUI import *
from ResampleForGUI import *
import platform

def outputTifFromArray(pro_data, resample_fn, fn, out_columns, out_rows):
    # Create the output raster using the computed dimensions.
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(fn,out_columns, out_rows)
    in_ds = gdal.Open(resample_fn)
    in_band = in_ds.GetRasterBand(1)
    out_ds.SetProjection(in_ds.GetProjection())

    gt = list(in_ds.GetGeoTransform())
    height_multiplier = in_band.YSize/float(out_rows)
    width_multiplier = in_band.XSize/float(out_columns)
    gt[1] *= width_multiplier
    gt[5] *= height_multiplier
    out_ds.SetGeoTransform(gt)
    
    data = np.copy(pro_data)
    # Multiplier of the data, so that the hightest is 255
    multiplier = float(255)/np.amax(data)
    data *= multiplier
    # Write the data to the output raster.
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data)

    # Compute statistics and build overviews.
    out_band.FlushCache()
    out_band.ComputeStatistics(True)
    # This might get wrong if the number of pixals in one row is less than 64
    # out_ds.BuildOverviews('average', [2, 4, 8, 16, 32, 64])

    del out_ds
    del in_ds

def calculate_the_probability(data, resample_fn, probability_fn, win_size,save_data):
    slices = make_resample_slices(data,win_size)
    stacked = np.stack(slices)
    prob_data = np.zeros((stacked.shape[1],stacked.shape[2]),np.float64)
    ln_prob_data = np.zeros((stacked.shape[1],stacked.shape[2]),np.float64)

    sum_data = np.sum(stacked,0)
    all_sum = np.sum(stacked)
    prob_data = sum_data / float(all_sum)
    ln_prob_data = np.where(prob_data==0,0,np.log(prob_data))

    # export the average_data to tif
    if save_data == 1:
        outputTifFromArray(prob_data, resample_fn, probability_fn, stacked.shape[1], stacked.shape[2])
        # sum = np.sum(outdata)
    # print(sum)
    return (prob_data,ln_prob_data)

def calculate_from_old_prob_data(old_prob_data, resample_fn, probability_fn, win_size,save_data):
    slices = make_resample_slices(old_prob_data,win_size)
    stacked = np.stack(slices)
    new_prob_data = np.zeros((stacked.shape[1],stacked.shape[2]),np.float64)
    new_ln_prob_data = np.zeros((stacked.shape[1],stacked.shape[2]),np.float64)
    new_prob_data = np.sum(stacked,0)
    new_ln_prob_data = np.where(new_prob_data==0,0,np.log(new_prob_data))

    # export the average_data to tif
    if save_data == 1:
        outputTifFromArray(new_prob_data, resample_fn, probability_fn, stacked.shape[1], stacked.shape[2])
    # sum = np.sum(outdata)
    # print(sum)
    return (new_prob_data,new_ln_prob_data)

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

def calculate_miu_lnmiu(prob_data,q):
    prob_data_q = np.zeros((prob_data.shape[0],prob_data.shape[1]),np.float64)
    prob_data_q = np.where(prob_data==0,0,np.power(prob_data,q))
    sum_prob_q = np.sum(prob_data_q)
    if sum_prob_q == 0:
        raise RuntimeError('Calculation of miu_q failed.')
    miu_q = prob_data_q/float(sum_prob_q)
    ln_miu_q = np.where(miu_q == 0,0,np.log(miu_q))
    return (prob_data_q,miu_q,ln_miu_q)

def ols_regression(x,y,scale_limit,fix_intercept):
    x = np.reshape(x,(len(x),1))
    y = np.reshape(y,(len(y),1))

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    if fix_intercept == 0:
        linreg = LinearRegression(fit_intercept=1)
    elif fix_intercept == 1:
        linreg = LinearRegression(fit_intercept=0)
    
    model = linreg.fit(x,y)
    y_predict = model.predict(x)
    
    r2 = model.score(x,y)
    # r2_score = metrics.r2_score(y,y_predict)

    #explained_variance_score = metrics.explained_variance_score(y,y_predict)
    #mean_absolute_error = metrics.mean_absolute_error(y,y_predict)
    mean_squared_error = metrics.mean_squared_error(y,y_predict)
    #median_absolute_error = metrics.median_absolute_error(y,y_predict)
    root_mean_squared_error = np.sqrt(mean_squared_error/len(x))

    # print(model)
    # print(linreg.intercept_)
    # print(linreg.coef_)
    if fix_intercept == 0:
        return [y_predict,linreg.intercept_[0],linreg.coef_[0][0],r2,root_mean_squared_error]
    elif fix_intercept == 1:
        return [y_predict,linreg.intercept_,linreg.coef_[0][0],r2,root_mean_squared_error]         


def calculate_the_spectrum(prob_data_list,q_list,scale_limit,fix_intercept,out_columns):

    order_q = [] 
    tau_q = []
    D_q = []
    alpha_q = []
    f_q = []

    ln_epsilon_list = []
    ln_sum_prob_q_list = []
    ln_sum_prob_q_divide_q_list = []
    sum_miu_q_ln_prob_list = []
    sum_miu_q_ln_miu_q_list = []

    miu_q_list_list = []
    for q in q_list:
        miu_q_list = []
        
        ln_epsilon = []
        ln_sum_prob_q = []
        ln_sum_prob_q_divide_q = []
        sum_miu_q_ln_prob = []
        sum_miu_q_ln_miu_q = []

        for i in range(int(np.log(out_columns)/np.log(2))+1):
            (prob_data_q,miu_q,ln_miu_q) = calculate_miu_lnmiu(prob_data_list[i][0],q)
            miu_q_list.append([prob_data_q,miu_q,ln_miu_q])

            ln_epsilon.append(np.log(1/float(out_columns/2**(i))))
            element = np.log(np.sum(prob_data_q))
            ln_sum_prob_q.append(element)
            if q == 1:
                element = np.sum(np.multiply(prob_data_list[i][0],prob_data_list[i][1]))
                ln_sum_prob_q_divide_q.append(element)
            else:
                element = ln_sum_prob_q[i]/float(q-1)
                ln_sum_prob_q_divide_q.append(element)
            element = np.sum(np.multiply(miu_q,prob_data_list[i][1]))
            sum_miu_q_ln_prob.append(element)
            element = np.sum(np.multiply(miu_q,ln_miu_q))
            sum_miu_q_ln_miu_q.append(element)
        #print('calculating q = {} ...'.format(q))


        order_q.append(q)
        tau_q.append(ols_regression(ln_epsilon,ln_sum_prob_q,scale_limit,fix_intercept))
        D_q.append(ols_regression(ln_epsilon,ln_sum_prob_q_divide_q,scale_limit,fix_intercept))
        alpha_q.append(ols_regression(ln_epsilon,sum_miu_q_ln_prob,scale_limit,fix_intercept))
        f_q.append(ols_regression(ln_epsilon,sum_miu_q_ln_miu_q,scale_limit,fix_intercept))
        
        ln_epsilon_list.append(ln_epsilon)
        ln_sum_prob_q_list.append(ln_sum_prob_q)
        ln_sum_prob_q_divide_q_list.append(ln_sum_prob_q_divide_q)      
        sum_miu_q_ln_prob_list.append(sum_miu_q_ln_prob)
        sum_miu_q_ln_miu_q_list.append(sum_miu_q_ln_miu_q)

        miu_q_list_list.append(miu_q_list)

    # print('q is {}\n'.format(order_q))
    # print('Tau(q) is {}\n'.format(tau_q))
    # print('D(q) is {}\n'.format(D_q))
    # print('alpha(q) is {}\n'.format(alpha_q))
    # print('f(q) is {}\n'.format(f_q))

    parameters = [order_q,tau_q,D_q,alpha_q,f_q]
    parameters_calculation = [ln_epsilon_list,ln_sum_prob_q_list,\
        ln_sum_prob_q_divide_q_list,sum_miu_q_ln_prob_list,sum_miu_q_ln_miu_q_list]
    return parameters,parameters_calculation,miu_q_list_list


def calculation_and_save(in_folder,out_folder,out_columns,out_rows,q_range=[-20,10,20],
            effective_scale_min=0,effective_scale_max=1024,
            fix_intercept=0,save_data=False,save_pic=True):
    # Open the input folder.
    if in_folder is None:
        raise RuntimeError('Could not open datasource')
    in_ds_list = file_name(in_folder,'.tif')

    name_list = []
    parameters_list = []

    prob_data = np.zeros((out_columns,out_rows),np.float64)
    ln_prob_data = np.zeros((out_columns,out_rows),np.float64)
    miu_q = np.zeros((out_columns,out_rows),np.float64)
    ln_miu_q = np.zeros((out_columns,out_rows),np.float64)
    # Loop through the tif files in the directory.
    for i in range(len(in_ds_list)):
        in_fn = in_ds_list[i]
        in_ds = gdal.Open(in_fn)
        in_band = in_ds.GetRasterBand(1)
        print('Calculating ' + in_fn + '...')

        sysstr = platform.system()
        if(sysstr =="Windows"):
            in_fn_without_dir = in_fn.split('\\')[-1]
            print ("The platform is Windows. Please use '\\' to split the path.")
        elif(sysstr == "Linux"):
            in_fn_without_dir = in_fn.split('/')[-1]
            print ("The platform is Linus. Please use '/' to split the path.")
        else: # Other platform might get wrong!
            print ("Other System tasks, might be wrong!")
        
        name_list.append(in_fn_without_dir)
        out_fn_ori = in_fn_without_dir[0:15]

        data = in_band.ReadAsArray()
        # data = np.where(data > 1, 0, 1)

        prob_data_list = []

        q_list = np.arange(q_range[0],q_range[2]+q_range[1],q_range[1])
        print(q_list)

        scale_ul = np.log2(effective_scale_max)
        scale_dl = np.log2(effective_scale_min)
        scale_limit = [scale_dl,scale_ul]
        

        for j in range(int(np.log(out_columns)/np.log(2))+1):
            column_num = out_columns/2**(j)
            row_num = out_rows/2**(j)
            probability_fn = out_fn_ori+'Prob_'+str(row_num)+'x'+str(column_num)+'.tif'
            probability_fn = os.path.join(out_folder,probability_fn)
            # change it to another 
            if j == 0:
                (prob_data,ln_prob_data) = calculate_the_probability(data,in_fn,
                        probability_fn,(1,1),save_data)
            elif j > 0:
                (prob_data,ln_prob_data) = calculate_from_old_prob_data(prob_data_list[j-1][0],
                        in_fn,probability_fn,(2,2),save_data)
            prob_data_list.append([prob_data,ln_prob_data])

        parameters,parameters_calculation,miu_q_list_list = calculate_the_spectrum(prob_data_list,
                q_list,scale_limit,fix_intercept,out_columns)
        parameters_list.append(parameters)

        if save_data == 1:
            save_to_hdf5(parameters,parameters_calculation,miu_q_list_list,prob_data_list,q_list,
                os.path.join(out_folder,out_fn_ori +'multispectrum'+'.h5'))
        
    
        fig_name_prefix = os.path.join(out_folder,out_fn_ori)
        plot_spectrum(parameters,parameters_calculation,q_list,scale_limit,fig_name_prefix,save_pic)
        
        del in_ds
    
    return (name_list,parameters_list)
# Script to run a bilinear interpolation
