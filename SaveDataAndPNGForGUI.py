# -*- coding: UTF-8 -*-  
# # Script to resample a raster to a smaller pixel size.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables as tb

def save_to_hdf5(parameters,parameters_calculation,miu_q_list_list,prob_data_list,q_list,fn):
    h5 = tb.open_file(fn,'w')

    table1_name = 'global and local parameters'
    table1_title = 'Global and Local Parameters'
    dty1 = np.dtype([('order_q','<f8'),('tau','<f8'),('D','<f8'),('alpha','<f8'),('f','<f8')])
    sarray = np.zeros(len(parameters[0]),dtype=dty1)
    sarray['order_q'] = parameters[0]
    sarray['tau'] = zip(*parameters[1])[2]
    sarray['D'] = zip(*parameters[2])[2]
    sarray['alpha'] = zip(*parameters[3])[2]
    sarray['f'] = zip(*parameters[4])[2]
    filters = tb.Filters(complevel=0)
    table1 = h5.create_table('/',table1_name,sarray,
                                title = table1_title,
                                expectedrows=len(parameters[0]),filters=filters)
    table1.flush()

    group2_name = 'parameters_calculation'
    group2 = h5.create_group(h5.root,group2_name)
    table2_name_list = ['ln_sum_prob_q','ln_sum_prob_q_divide_q',
            'sum_miu_q_ln_prob','sum_miu_q_ln_miu_q']
    table2_title_list = ['Raw Data in estimation of tau','Raw Data in estimation of D',
            'Raw Data in estimation of alpha','Raw Data in estimation of f']
    
    type2_list = []
    type2_list.append(('lnepsilon','<f8'))
    for q in q_list:
        type2_list.append(('q='+str(q),'<f8'))
    dty2 = np.dtype(type2_list)
    
    filters = tb.Filters(complevel=0)

    for i in range(len(parameters_calculation)-1):
        sarray = np.zeros(len(parameters_calculation[0][0]),dtype=dty2)
        sarray[type2_list[0][0]] = parameters_calculation[0][0]
        for j in range(len(type2_list)-1):
            sarray[type2_list[j+1][0]] = parameters_calculation[i+1][j]
        table2 = h5.create_table("/parameters_calculation",table2_name_list[i],sarray,
                                title = table2_title_list[i],
                                expectedrows=len(parameters_calculation[0][0]),
                                filters=filters)
        table2.flush()

    # group2._f_close()

    group3_name = 'Pq_miu_lnmiu'
    group3 = h5.create_group(h5.root,group3_name)
    group3_q_s_r_name_list = ['Pq','miu','lnmiu']
    for q in q_list:
        index = 0
        group3_q_name = 'q='+str(q)
        group3_q = h5.create_group(group3,group3_q_name)

        for scale in range(len(parameters_calculation[0][0])):
            scale_max = 2**(len(parameters_calculation[0][0])-1)
            scale_now = str(scale_max/2**scale)
            group3_q_s_name = 'scale='+scale_now
            group3_q_s = h5.create_group(group3_q,group3_q_s_name)
            for r in range(len(group3_q_s_r_name_list)):
                array3_q_s_r_name = group3_q_s_r_name_list[r]
                array3_q_s_r_title = 'The 2-D array of '+group3_q_s_r_name_list[r]
                array3 = h5.create_array(group3_q_s,array3_q_s_r_name,
                    miu_q_list_list[index][scale][r],array3_q_s_r_title)
                array3.flush()
        index += 1

    group4_name = 'P_and_lnP'
    group4 = h5.create_group(h5.root,group4_name)

    for scale in range(len(parameters_calculation[0][0])):
        scale_max = 2**(len(parameters_calculation[0][0])-1)
        scale_now = str(scale_max/2**scale)
        group4_s_name = 'scale='+scale_now
        group4_s = h5.create_group(group4,group4_s_name)
        group4_s_r_name_list = ['P','lnP']
    
        for r in range(len(group4_s_r_name_list)):
            array4_s_r_name = group4_s_r_name_list[r]
            array4_s_r_title = 'The '+group4_s_r_name_list[r]
            array4 = h5.create_array(group4_s,array4_s_r_name,
                    prob_data_list[scale][r],array4_s_r_title)
            array4.flush()

    h5.flush()
    h5.close()         

    # read from hdf5
    # a = pd.read_hdf(fn)

def plot_spectrum(parameters,parameters_calculation,
            q_list,scale_limit,fig_name_prefix,save_pic):
    from matplotlib.ticker import NullFormatter  # useful for `logit` scale

    # plot with various axes scales
    plt.figure(1,figsize=(12,8))

    # Find where q == 0 & 1 & 2
    where_q_is_zero = np.where(q_list==0)[0][0]
    where_q_is_one = np.where(q_list==1)[0][0]
    where_q_is_two = np.where(q_list==2)[0][0]
    # D_0
    plt.subplot(241)
    x = parameters_calculation[0][1]

    y = parameters_calculation[2][where_q_is_zero]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[2][where_q_is_zero][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,'$D_0 = {:.6f}$'.format(parameters[2][where_q_is_zero][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[2][where_q_is_zero][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[2][where_q_is_zero][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\ln N(\epsilon)$')
    plt.title('The capacity dimension $D_0$')
    plt.grid(True)


    # D(1)
    plt.subplot(242)
    x = parameters_calculation[0][1]
    
    y = parameters_calculation[2][where_q_is_one]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[2][where_q_is_one][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,'$D_1 = {:.6f}$'.format(parameters[2][where_q_is_one][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[2][where_q_is_one][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[2][where_q_is_one][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-I(\epsilon)$')
    plt.title('The information dimension $D_1$')
    plt.grid(True)

    # D(2)
    plt.subplot(243)
    x = parameters_calculation[0][1]

    y = parameters_calculation[2][where_q_is_two]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[2][where_q_is_two][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,'$D_2 = {:.6f}$'.format(parameters[2][where_q_is_two][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[2][where_q_is_two][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[2][where_q_is_two][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\ln\Sigma P^2(\epsilon)$')
    plt.title('The correlation dimension $D_2$')
    plt.grid(True)

    # alpha(q) v.s. f(q)
    plt.subplot(244)
    x = zip(*parameters[3])[2]
    y = zip(*parameters[4])[2]
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black")
    plt.plot(x,y,"r-")
    plt.xlabel(r'$\alpha(q)$')
    plt.ylabel(r'$f(q)$')
    plt.title(r'$\alpha(q)$ v.s. $f(q)$')
    plt.grid(True)

    # q v.s. tau(q)
    plt.subplot(245)
    x = parameters[0]
    y = zip(*parameters[1])[2]
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black")
    plt.plot(x,y,"r-")
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\tau(q)$')
    plt.title(r'$q$ v.s. $\tau(q)$')
    plt.grid(True)

    # q v.s. D(q)
    plt.subplot(246)
    x = parameters[0]
    y = zip(*parameters[2])[2]
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black")
    plt.plot(x,y,"r-")
    plt.xlabel(r'$q$')
    plt.ylabel('$D(q)$')
    plt.title(r'$q$ v.s. $D(q)$')
    plt.grid(True)

    # q v.s. alpha(q)
    plt.subplot(247)
    x = parameters[0]
    y = zip(*parameters[3])[2]
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black")
    plt.plot(x,y,"r-")
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\alpha(q)$')
    plt.title(r'$q$ v.s. $\alpha(q)$')
    plt.grid(True)

    # q v.s. f(q)
    plt.subplot(248)
    x = parameters[0]
    y = zip(*parameters[4])[2]
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black")
    plt.plot(x,y,"r-")
    plt.xlabel('$q$')
    plt.ylabel('$f(q)$')
    plt.title('$q$ v.s. $f(q)$')
    plt.grid(True)

    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.40)
    if save_pic == 1:
        plt.savefig(fig_name_prefix+'Parameters.png')
    plt.show()
    plt.close()

    # plot with various axes scales
    plt.figure(2,figsize=(12,8))

    # Find where q == -20 & -10 & 10 & 20
    where_q_is_minus_20 = np.where(q_list==-20)[0][0]
    where_q_is_minus_10 = np.where(q_list==-10)[0][0]
    where_q_is_positive_10 = np.where(q_list==10)[0][0]
    where_q_is_positive_20 = np.where(q_list==20)[0][0]

    # alpha(q=minus_20)
    plt.subplot(241)
    x = parameters_calculation[0][1]

    y = parameters_calculation[3][where_q_is_minus_20]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[3][where_q_is_minus_20][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$\alpha (q=-20) = {:.6f}$'.format(parameters[3][where_q_is_minus_20][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[3][where_q_is_minus_20][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[3][where_q_is_minus_20][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln P$')
    plt.title(r'$\alpha (q=-20)$')
    plt.grid(True)


    # alpha(q=-10)
    plt.subplot(242)
    x = parameters_calculation[0][1]

    y = parameters_calculation[3][where_q_is_minus_10]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[3][where_q_is_minus_10][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$\alpha (q=-10) = {:.6f}$'.format(parameters[3][where_q_is_minus_10][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[3][where_q_is_minus_10][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[3][where_q_is_minus_10][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln P$')
    plt.title(r'$\alpha (q=-10)$')
    plt.grid(True)

    # alpha(q=10)
    plt.subplot(243)
    x = parameters_calculation[0][1]

    y = parameters_calculation[3][where_q_is_positive_10]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[3][where_q_is_positive_10][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$\alpha (q=10) = {:.6f}$'.format(parameters[3][where_q_is_positive_10][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[3][where_q_is_positive_10][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[3][where_q_is_positive_10][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln P$')
    plt.title(r'$\alpha (q=10)$')
    plt.grid(True)

    # alpha(q=20)
    plt.subplot(244)
    x = parameters_calculation[0][1]
    
    y = parameters_calculation[3][where_q_is_positive_20]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[3][where_q_is_positive_20][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$\alpha (q=20) = {:.6f}$'.format(parameters[3][where_q_is_positive_20][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[3][where_q_is_positive_20][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[3][where_q_is_positive_20][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln P$')
    plt.title(r'$\alpha (q=20)$')
    plt.grid(True)

    # f(q=minus_20)
    plt.subplot(245)
    x = parameters_calculation[0][1]

    y = parameters_calculation[4][where_q_is_minus_20]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[4][where_q_is_minus_20][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$f(q=-20) = {:.6f}$'.format(parameters[4][where_q_is_minus_20][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[4][where_q_is_minus_20][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[4][where_q_is_minus_20][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln\mu$')
    plt.title(r'$f(q=-20)$')
    plt.grid(True)


    # f(q=-10)
    plt.subplot(246)
    x = parameters_calculation[0][1]

    y = parameters_calculation[4][where_q_is_minus_10]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[4][where_q_is_minus_10][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$f(q=-10) = {:.6f}$'.format(parameters[4][where_q_is_minus_10][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[4][where_q_is_minus_10][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[4][where_q_is_minus_10][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln\mu$')
    plt.title(r'$f(q=-10)$')
    plt.grid(True)

    # f(q=10)
    plt.subplot(247)
    x = parameters_calculation[0][1]

    y = parameters_calculation[4][where_q_is_positive_10]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[4][where_q_is_positive_10][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$f(q=10) = {:.6f}$'.format(parameters[4][where_q_is_positive_10][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[4][where_q_is_positive_10][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[4][where_q_is_positive_10][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln\mu$')
    plt.title(r'$f(q=10)$')
    plt.grid(True)

    # f(q=20)
    plt.subplot(248)
    x = parameters_calculation[0][1]

    y = parameters_calculation[4][where_q_is_positive_20]

    x = x[(len(x)-int(scale_limit[1])-1):(len(x)-int(scale_limit[0]))]
    y = y[(len(y)-int(scale_limit[1])-1):(len(y)-int(scale_limit[0]))]

    x_array = np.array(x)
    y_predict = parameters[4][where_q_is_positive_20][0]
    y_linreg = list(y_predict)
    plt.plot(x,y,"o",ms=4,alpha=0.7,mfc="black",label='value')
    plt.plot(x,y_linreg,"r-",label='regression line')
    plt.text(-7,-1,r'$f(q=20) = {:.6f}$'.format(parameters[4][where_q_is_positive_20][2]))
    plt.text(-7,-2.5,'$R^2 = {:.6f}$'.format(parameters[4][where_q_is_positive_20][3]))
    plt.text(-7,-4,'$se = {:.6f}$'.format(parameters[4][where_q_is_positive_20][4]))
    #plt.legend(loc=0)
    plt.xlabel(r'$\ln\epsilon$')
    plt.ylabel(r'$-\Sigma\mu\ln\mu$')
    plt.title(r'$f(q=20)$')
    plt.grid(True)

    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.40)
    if save_pic == 1:
        plt.savefig(fig_name_prefix+'Alpha_f.png')
    plt.show()
    plt.close()

