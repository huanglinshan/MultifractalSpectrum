# -*- coding: UTF-8 -*-
# # Script to calculate multifractal spectra.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import gc


class Calculator():
    def __init__(self, data, q_list, scale_limit, intercept, out_column, out_row):
        """
        initialize.

        :param data        - data (zero for no data)
        :param q_list      - q range
        :param scale_limit - scaling range during OLS
        :param intercept   - whether to fix the intercept | 0: intercept = 0 & 1: intercept = 1
        :param out_column  - num of columns
        :param out_row     - num of rows
        """
        self.data = data

        # q range
        self.q_list = q_list
        # scaling range during OLS
        self.scale_limit = scale_limit
        # whether to fix the intercept | 0: intercept = 0 & 1: intercept = 1
        self.intercept = intercept

        # num of columns
        self.out_column = out_column
        # num of rows
        self.out_row = out_row
        # error when comparing whether two numbers are equal
        self.DELTA = 1e-9

        # init prob_matrix list
        self.prob_data_list = []
        # init parameter list
        self.parameters = []
        # ?
        self.parameters_calculation = []
        # init miu list | it can be set whether to save miu list
        self.miu_q_list_list = []

    @staticmethod
    def _make_resample_slices(data, win_size):
        """
        Return a list of resampled slices given a window size.

        :param  data     - two-dimensional array to get slices from
        :param  win_size - tuple of (rows, columns) for the input window
        :return slices   - a list of resampled slices given a window size
        """
        row = int(data.shape[0] / win_size[0]) * win_size[0]
        col = int(data.shape[1] / win_size[1]) * win_size[1]
        slices = []

        for i in range(win_size[0]):
            for j in range(win_size[1]):
                slices.append(data[i:row:win_size[0], j:col:win_size[1]])
        return slices

    def _calculate_the_probability(self, data, win_size):
        """
        Calculate probability arrays from the original slices.

        :return prob_data    - probability of data arrays
                ln_prob_data - log of prob_data (0 when prob_data == 0)
        """
        slices = self._make_resample_slices(data, win_size)
        stacked = np.stack(slices)

        sum_data = np.sum(stacked, 0)
        all_sum = np.sum(sum_data)
        prob_data = sum_data / float(all_sum)
        ln_prob_data = np.where(prob_data == 0, 0, np.log(prob_data))
        return prob_data, ln_prob_data

    def _calculate_from_old_prob_data(self, old_prob_data, win_size):
        """
        Calculate probability arrays from slices based on the previous step.

        :param  old_prob_data - slices calculated in the previous step
        :param  win_size      - tuple of (rows, columns) for the input window
        :return prob_data     - probability of data arrays
                ln_prob_data  - log of prob_data (0 when prob_data == 0)
        """
        slices = self._make_resample_slices(old_prob_data, win_size)
        stacked = np.stack(slices)
        new_prob_data = np.sum(stacked, 0)
        new_ln_prob_data = np.where(new_prob_data == 0, 0, np.log(new_prob_data))
        return new_prob_data, new_ln_prob_data

    @staticmethod
    def _calculate_miu_lnmiu(prob_data, q):
        """
        Calculate miu(q) and ln(miu(q)).

        :param  prob_data   - Pi(probability of data arrays)
        :param  q           - order of moment q
        :return prob_data_q - Pi^q
                miu_q       - miu(q)
                ln_miu_q    - ln(miu(q))
        """
        prob_data_q = np.where(prob_data == 0, 0, np.power(prob_data, q))
        sum_prob_q = np.sum(prob_data_q)
        if sum_prob_q == 0:
            raise RuntimeError('Calculation of miu_q failed.')
        miu_q = prob_data_q / float(sum_prob_q)
        ln_miu_q = np.where(miu_q == 0, 0, np.log(miu_q))
        return prob_data_q, miu_q, ln_miu_q

    def _ols_regression(self, x, y, label=0):
        """
        OLS regression in calculating multifractal spectrum.

        :param  x                       - independent variable
        :param  y                       - dependent variable
        :param  label                   - used to distinguish scale_limit for different parameters.
        :return y_predict               - predicted dependent variable
                linreg.intercept_[0]    - intercept of regression model (OR linreg.intercept_)
                linreg.coef_[0][0]      - coefficient of regression model
                r2                      - the goodness of fit (R2)
                root_mean_squared_error - root of mean squared error
        """
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))

        # define scaling range during OLS regression
        x = x[(len(x) - int(self.scale_limit[label][1]) - 1):(len(x) - int(self.scale_limit[label][0]))]
        y = y[(len(y) - int(self.scale_limit[label][1]) - 1):(len(y) - int(self.scale_limit[label][0]))]

        #### choose to fix the intercept during OLS regression or NOT.
        # fix the intercept to zero
        if self.intercept == 0:
            linreg = LinearRegression(fit_intercept=0)
        # not to fix the intercept
        elif self.intercept == 1:
            linreg = LinearRegression(fit_intercept=1)

        lin_fit = linreg.fit(x, y)
        y_predict = lin_fit.predict(x)
        r2 = lin_fit.score(x, y)
        mean_squared_error = metrics.mean_squared_error(y, y_predict)
        root_mean_squared_error = np.sqrt(mean_squared_error / len(x))

        if self.intercept == 1:
            return [y_predict, linreg.intercept_[0], linreg.coef_[0][0], r2, root_mean_squared_error]
        elif self.intercept == 0:
            return [y_predict, linreg.intercept_, linreg.coef_[0][0], r2, root_mean_squared_error]

    def _alculate_the_spectrum(self):
        # initiate spectra list
        order_q = []
        tau_q = []
        D_q = []
        alpha_q = []
        f_q = []

        # initiate data list for calculating four parameters and ln(epsilon) list
        ln_epsilon_list = []
        ln_sum_prob_q_list = []
        ln_sum_prob_q_divide_q_list = []
        sum_miu_q_ln_prob_list = []
        sum_miu_q_ln_miu_q_list = []

        miu_q_list_list = []
        # save parameter to list for different q.
        for q in self.q_list:
            miu_q_list = []

            ln_epsilon = []
            ln_sum_prob_q = []
            ln_sum_prob_q_divide_q = []
            sum_miu_q_ln_prob = []
            sum_miu_q_ln_miu_q = []

            print('q = {} ...'.format(q))
            for i in range(int(np.log(self.out_column) / np.log(2)) + 1):
                (prob_data_q, miu_q, ln_miu_q) = self._calculate_miu_lnmiu(self.prob_data_list[i][0], q)
                miu_q_list.append([prob_data_q, miu_q, ln_miu_q])

                ln_epsilon.append(np.log(1 / float(self.out_column / 2 ** (i))))
                element = np.log(np.sum(prob_data_q))
                ln_sum_prob_q.append(element)
                if q == 1:
                    element = np.sum(np.multiply(self.prob_data_list[i][0], self.prob_data_list[i][1]))
                    ln_sum_prob_q_divide_q.append(element)
                else:
                    element = ln_sum_prob_q[i] / float(q - 1)
                    ln_sum_prob_q_divide_q.append(element)
                element = np.sum(np.multiply(miu_q, self.prob_data_list[i][1]))
                sum_miu_q_ln_prob.append(element)
                element = np.sum(np.multiply(miu_q, ln_miu_q))
                sum_miu_q_ln_miu_q.append(element)

            # append calculated parameters to corresponding list
            order_q.append(q)
            tau_q.append(self._ols_regression(ln_epsilon, ln_sum_prob_q, label=0))
            D_q.append(self._ols_regression(ln_epsilon, ln_sum_prob_q_divide_q, label=0))
            alpha_q.append(self._ols_regression(ln_epsilon, sum_miu_q_ln_prob, label=1))
            f_q.append(self._ols_regression(ln_epsilon, sum_miu_q_ln_miu_q, label=2))

            ln_epsilon_list.append(ln_epsilon)
            ln_sum_prob_q_list.append(ln_sum_prob_q)
            ln_sum_prob_q_divide_q_list.append(ln_sum_prob_q_divide_q)
            sum_miu_q_ln_prob_list.append(sum_miu_q_ln_prob)
            sum_miu_q_ln_miu_q_list.append(sum_miu_q_ln_miu_q)

            #### save miu(q) | not to save by default since it might consume much memory.
            # miu_q_list_list.append(miu_q_list)

            # deallocate memory.
            del ln_epsilon,ln_sum_prob_q,ln_sum_prob_q_divide_q,sum_miu_q_ln_prob,sum_miu_q_ln_miu_q

        #### save parameters and data used to calculater parameters using OLS regression
        parameters = [order_q, tau_q, D_q, alpha_q, f_q]
        parameters_calculation = [ln_epsilon_list, ln_sum_prob_q_list,
                                  ln_sum_prob_q_divide_q_list, sum_miu_q_ln_prob_list, sum_miu_q_ln_miu_q_list]
        return parameters, parameters_calculation, miu_q_list_list

    def calculation_and_save(self):
        """
        ############################# MAIN FUNCTION #############################
                    Calcute the parameters and save relevant data to disk.
        #########################################################################
        """
        # deallocate memory.
        gc.collect()

        prob_data = np.zeros((self.out_column, self.out_row), np.float64)
        ln_prob_data = np.zeros((self.out_column, self.out_row), np.float64)

        for j in range(int(np.log(self.out_column) / np.log(2)) + 1):
            # calculate the first probability matrix.
            if j == 0:
                (prob_data, ln_prob_data) = self._calculate_the_probability(self.data, (1, 1))
            # calculate other probability based on the previous matrix.
            elif j > 0:
                (prob_data, ln_prob_data) = self._calculate_from_old_prob_data(self.prob_data_list[j - 1][0],(2, 2))
            self.prob_data_list.append([prob_data, ln_prob_data])

        #### calculate multifractal parameters
        self.parameters, self.parameters_calculation, self.miu_q_list_list = self._calculate_the_spectrum()


