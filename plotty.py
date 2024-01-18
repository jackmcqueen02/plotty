#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:41:08 2024

@author: jackmcqueen
"""

# Standard libary imports
# Numba imported to speed up walk simulations
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rayleigh
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import kstest  # may require install
# sklearn.metrics used to calculate r squared values
from sklearn.metrics import r2_score  # may require install
from scipy.stats import chi2_contingency



# Plotting Functions


# This function describes a simple power function for use in later regressions.
# is used with scipy.optimise's curve_fit to fit regression lines and to
# calculate R^2 values for power function fits. Used in the variance vs number
# of steps to fit the best fit. Additionally, it is used in bonus projects
# to perfom regressions.


def PowerLaw(x_array, a_fit, n_fit, c_fit):
    """
     Power function for use in fitting.

    Parameters
    ----------
    x_array : numpy.ndarray
        array of steps taken on a random walk.
    a_fit : float
        Scale factor.
    n _fit: float
        Power relation.
    c_fit : float
        Intercept of the fit.

    Returns
    -------
    numpy.ndarrray
        1d array containing a*(x)**n + c where a,n and c are the best
        fit parameters.

    """
    # returns power function for use in regressions. a_fit,n_fit and c_fit
    # are the best fit parameters for the regression fit.
    return a_fit*(x_array)**n_fit + c_fit


def linear(x_fit, a_fit, c_fit):
    """
    Straight line function for use in fitting.

    Parameters
    ----------
    x_fit : numpy.ndarray
        array of steps taken on a random walk.
    a_fit : float
        Scale factor.
    c_fit : float
        Intercept of the fit.

    Returns
    -------
    numpy.ndarrray
        1d array containing a_fit*(x_fit) + c_fit where a and c are the best
        fit parameters.

    """
    # returns straight line function with fit parameters
    return a_fit*(x_fit) + c_fit

def exponential(x_fit, a_fit, c_fit):
    """
    exponential line function for use in fitting.

    Parameters
    ----------
    x_fit : numpy.ndarray
        array of steps taken on a random walk.
    a_fit : float
        Scale factor.
    c_fit : float
        Intercept of the fit.

    Returns
    -------
    numpy.ndarrray
        1d array containing a_fit*(x_fit) + c_fit where a and c are the best
        fit parameters.

    """
    # returns straight line function with fit parameters
    return a_fit**(x_fit) + c_fit

def polynomial(x_fit,a_fit,b_fit,c_fit):
    return a_fit*(x_fit)**2 + b_fit*(x_fit) + c_fit
    
def r_squared(data, fit_function):
    """
    Determine the R Squared value for the regression fit.

    Parameters
    ----------
    data : numpy.ndarray
        Results from 'process_r_t_o_data'.
    fit_function : callable
        function to be fitted to the data.

    Returns
    -------
    r_squared : float
        Value for the r_squared of the best fit.

    """
    # creates an array from the inputted data. This is important
    # as it allows for calculations using numpy which is convinient.
    data = np.array(data)

    # extracts the best fit parameters for the inputted (x,y) data for use
    # in calculating the best fit lines parameters
    popt, _ = curve_fit(fit_function, data[0], data[1])

    # calculates the y values from the regression line. These are the
    # predicted y values and are compared with the datas y values to
    # access the r squared values. Makes use of the defined fit functions
    # to calculate the relevant y valuese based on the regression parameters.
    y_pred = fit_function(data[0], *popt)

    # calculates the r squared value for the best fit line. This is done
    # using the r2_score package which provides a convinient way of
    # calculating r squared values. the datas y are inputted alongside
    # the y values predicted by the regression line function.
    r_squared_value = r2_score(data[1], y_pred)

    # returns the calculated r squared value for the specific regression fit
    return r_squared_value

def r_squared_error(data, fit_function):
    """
    Determine the R Squared value for the regression fit.

    Parameters
    ----------
    data : numpy.ndarray
        Results from 'process_r_t_o_data'.
    fit_function : callable
        function to be fitted to the data.

    Returns
    -------
    r_squared : float
        Value for the r_squared of the best fit.

    """
    # creates an array from the inputted data. This is important
    # as it allows for calculations using numpy which is convinient.
    data = np.array(data)

    # extracts the best fit parameters for the inputted (x,y) data for use
    # in calculating the best fit lines parameters
    popt, _ = curve_fit(fit_function, data[0], data[1],yerr =data[2])

    # calculates the y values from the regression line. These are the
    # predicted y values and are compared with the datas y values to
    # access the r squared values. Makes use of the defined fit functions
    # to calculate the relevant y valuese based on the regression parameters.
    y_pred = fit_function(data[0], *popt)

    # calculates the r squared value for the best fit line. This is done
    # using the r2_score package which provides a convinient way of
    # calculating r squared values. the datas y are inputted alongside
    # the y values predicted by the regression line function.
    r_squared_value = r2_score(data[1], y_pred)

    # returns the calculated r squared value for the specific regression fit
    return r_squared_value


def chi2_test(data_arr):
    observed = np.array([data_arr[0],
                         data_arr[1]])
    chi2, p, dof, expected = chi2_contingency(observed)
    return chi2,dof

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



def LSF_gaussian(data, fit_function, x_label, y_label, plot_title,label):
    """
    Perform regression on the variance vs number of steps on a SAW walk.

    Can be used to fit an exponent of 1.5 or a best fitting exponent and
    to calculate the errors on these best fit values.



    Parameters
    ----------
    data : numpy.ndarray
        'process_s_a_w_data' results.
    fit_function : callable
        Function to be fitted to the data using regression.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    plot_title : str
        Title of the plot.

    Returns
    -------
    none
    """
    # turns the inputted data (variance and number of steps) into an array.
    # This step is crucial to enable analysis using numpy functions
    data = np.array(data)

    # Calculates the KS best fitting indicators (D and p), it fits evaluates
    # the difference in the fit points and the data points. Makes use of
    # the defined k_s_statistic function to extract these parameters. This
    # function is defined in the document and is provides a convinient way
    # to calculate the KS parameters

    # extracts the D and p values for the fit from the k_s_analysis variabl

    # formats the D and p values to be presentable on the plots as they have
    # a high number of floating points.
    
    rsquared = r_squared(data,fit_function)
    rsquared_formatted = f'{rsquared:.3f}'
    # defines the x array for the plot by extracting it from the inputted data,
    # for the specific case being analysed this will be the number of steps
    # array
    x_data = data[0]

    # defines the y array for the plot by extracting it from the inputted data
    # for the specific case being evaluated this will be the array of the
    # variances.
    y_data = data[1]

    # calculates the best fit parameters (popt) and the covariance matrix
    # (pcov) for the fitting function. This is done using the curve_fit
    # package provided by scipy
    popt, pcov = curve_fit(fit_function, data[0], data[1])
    
    shape = np.shape(pcov)[0]
    errors = np.zeros(shape)
    for i in range(shape):
        errors[i] = np.sqrt(pcov[i,i])
    
    # takes advantage of the covariance matrix to calculate estimates for
    # the errors on the best fit parameters. Calculates the standard errors
    # from the diagonals of the covariance matrix. Square rooting the
    # covariance matrix returns the standard error on the scale and
    # intercepts of the fit.
    
    plot_errors = [f'{error:.3f}' for error in errors]
    formatted_strings = []
    for i in range(len(popt)):
        formatted_str = f"$p_{i}= ({popt[i]:.3f} \\pm {plot_errors[i]})$"
        formatted_strings.append(formatted_str)
    # formats the errors to be more readable on the plot legend as they
    # have a large number of floating points.
    legend_entry = '\n'.join(formatted_strings)
    # creates a scatter plot of the x,y data to enable the regression
    # using the 1.5 exponent.
    plt.rcParams['figure.dpi'] = 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9))

    ax1.scatter(x_data, y_data, label='', marker = 's', s = 25)
    
    # adds the regression line to the plot, alongside formatted labels
    ax1.plot(x_data, fit_function(x_data, *popt), 'r', label=(
        f'$R^{2} = {rsquared_formatted}$\n'))

    ax1.set_xlabel(x_label, fontsize=12)  # sets the x axis label for all plots
    ax1.set_ylabel(y_label, fontsize=12)  # sets the y axis label for all plots
    ax1.set_title(plot_title, fontsize=14)   # sets the plot title for all plots
    ax1.tick_params(axis='x', direction='in', length=10, width=1)  # Adjust length and width as needed
    ax1.tick_params(axis='y', direction='in', length=10, width=1)
    ax1.tick_params(axis='x', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    ax1.tick_params(axis='y', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    

    # displays the legend in a convinient and readable position using the bbox
    # anchor arguent and the location argument.
    
    ax1.legend(framealpha = 0.5, frameon = True, markerscale = 0, handleheight = 0, handlelength = 0, fontsize = 12)
    
    residuals = data[1] - fit_function(data[0],*popt)
    
    ax2.scatter(data[0], residuals, alpha=1,marker = 's', s = 25)
    ax2.axhline(y=0, linestyle='-', color ='red')
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel('Residuals', fontsize = 14)
    
    plt.subplots_adjust(hspace=0.2)
    ax2.tick_params(axis='x', direction='in', length=10, width=1)  # Adjust length and width as needed
    ax2.tick_params(axis='y', direction='in', length=10, width=1)
    ax2.tick_params(axis='x', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    ax2.tick_params(axis='y', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    plt.show()  # displays the relevant plot
     
    print_strings = []
    plot_errors_print = [f'{error}' for error in errors]
    
    for i in range(len(popt)):
        formatted_str = f"p_{i}= ({popt[i]} \\pm {plot_errors_print[i]})"
        print_strings.append(formatted_str)
    
    # formats the errors to be more readable on the plot legend as they
    # have a large number of floating points.
    legend_entry_print = '\n'.join(print_strings)

    print(
        f'------------------------------------\n'
        f'{label}\n'
        f'R^2 = {rsquared}\n'
        f"{legend_entry_print}\n"
        f'------------------------------------'
    )

















def LSF(data, fit_function, x_label, y_label, plot_title,label):
    """
    Perform regression on the variance vs number of steps on a SAW walk.

    Can be used to fit an exponent of 1.5 or a best fitting exponent and
    to calculate the errors on these best fit values.



    Parameters
    ----------
    data : numpy.ndarray
        'process_s_a_w_data' results.
    fit_function : callable
        Function to be fitted to the data using regression.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    plot_title : str
        Title of the plot.

    Returns
    -------
    none
    """
    # turns the inputted data (variance and number of steps) into an array.
    # This step is crucial to enable analysis using numpy functions
    data = np.array(data)

    # Calculates the KS best fitting indicators (D and p), it fits evaluates
    # the difference in the fit points and the data points. Makes use of
    # the defined k_s_statistic function to extract these parameters. This
    # function is defined in the document and is provides a convinient way
    # to calculate the KS parameters

    # extracts the D and p values for the fit from the k_s_analysis variabl

    # formats the D and p values to be presentable on the plots as they have
    # a high number of floating points.
    
    rsquared = r_squared(data,fit_function)
    rsquared_formatted = f'{rsquared:.3f}'
    chi2 = chi2_test(data)[0]
    dof = chi2_test(data)[1]
    chi2ndf = chi2/dof
    chi2ndf_formatted = f'{chi2ndf:.3f}'
    chi2_formatted = f'{chi2:.3f}'
    # defines the x array for the plot by extracting it from the inputted data,
    # for the specific case being analysed this will be the number of steps
    # array
    x_data = data[0]

    # defines the y array for the plot by extracting it from the inputted data
    # for the specific case being evaluated this will be the array of the
    # variances.
    y_data = data[1]

    # calculates the best fit parameters (popt) and the covariance matrix
    # (pcov) for the fitting function. This is done using the curve_fit
    # package provided by scipy
    popt, pcov = curve_fit(fit_function, data[0], data[1])
    
    shape = np.shape(pcov)[0]
    errors = np.zeros(shape)
    for i in range(shape):
        errors[i] = np.sqrt(pcov[i,i])
    
    # takes advantage of the covariance matrix to calculate estimates for
    # the errors on the best fit parameters. Calculates the standard errors
    # from the diagonals of the covariance matrix. Square rooting the
    # covariance matrix returns the standard error on the scale and
    # intercepts of the fit.
    
    plot_errors = [f'{error:.3f}' for error in errors]
    formatted_strings = []
    for i in range(len(popt)):
        formatted_str = f"$p_{i}= ({popt[i]:.3f} \\pm {plot_errors[i]})$"
        formatted_strings.append(formatted_str)
    # formats the errors to be more readable on the plot legend as they
    # have a large number of floating points.
    legend_entry = '\n'.join(formatted_strings)
    # creates a scatter plot of the x,y data to enable the regression
    # using the 1.5 exponent.
    plt.rcParams['figure.dpi'] = 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9))

    ax1.scatter(x_data, y_data, label='', marker = 's', s = 25)
    
    # adds the regression line to the plot, alongside formatted labels
    ax1.plot(x_data, fit_function(x_data, *popt), 'r', label=(
        f"$\\frac{{\\chi^2}}{{ndf}} = {chi2ndf_formatted}$\n"
        f'$R^{2} = {rsquared_formatted}$\n'
        f"{legend_entry}"))

    ax1.set_xlabel(x_label, fontsize=12)  # sets the x axis label for all plots
    ax1.set_ylabel(y_label, fontsize=12)  # sets the y axis label for all plots
    ax1.set_title(plot_title, fontsize=14)   # sets the plot title for all plots
    ax1.tick_params(axis='x', direction='in', length=10, width=1)  # Adjust length and width as needed
    ax1.tick_params(axis='y', direction='in', length=10, width=1)
    ax1.tick_params(axis='x', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    ax1.tick_params(axis='y', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    

    # displays the legend in a convinient and readable position using the bbox
    # anchor arguent and the location argument.
    
    ax1.legend(framealpha = 0.5, frameon = True, markerscale = 0, handleheight = 0, handlelength = 0, fontsize = 12)
    
    residuals = data[1] - fit_function(data[0],*popt)
    
    ax2.scatter(data[0], residuals, alpha=1,marker = 's', s = 25)
    ax2.axhline(y=0, linestyle='-', color ='red')
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel('Residuals', fontsize = 14)
    
    plt.subplots_adjust(hspace=0.2)
    ax2.tick_params(axis='x', direction='in', length=10, width=1)  # Adjust length and width as needed
    ax2.tick_params(axis='y', direction='in', length=10, width=1)
    ax2.tick_params(axis='x', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    ax2.tick_params(axis='y', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    plt.show()  # displays the relevant plot
     
    print_strings = []
    plot_errors_print = [f'{error}' for error in errors]
    
    for i in range(len(popt)):
        formatted_str = f"p_{i}= ({popt[i]} \\pm {plot_errors_print[i]})"
        print_strings.append(formatted_str)
    
    # formats the errors to be more readable on the plot legend as they
    # have a large number of floating points.
    legend_entry_print = '\n'.join(print_strings)

    print(
        f'------------------------------------\n'
        f'{label}\n'
        f'chi^2 per ndf = {chi2ndf}\n'
        f'R^2 = {rsquared}\n'
        f"{legend_entry_print}\n"
        f'------------------------------------'
    )

def error_LSF(data, fit_function, x_label, y_label, plot_title,label):
    """
    Perform regression on the variance vs number of steps on a SAW walk.

    Can be used to fit an exponent of 1.5 or a best fitting exponent and
    to calculate the errors on these best fit values.



    Parameters
    ----------
    data : numpy.ndarray
        'process_s_a_w_data' results.
    fit_function : callable
        Function to be fitted to the data using regression.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    plot_title : str
        Title of the plot.

    Returns
    -------
    none
    """
    # turns the inputted data (variance and number of steps) into an array.
    # This step is crucial to enable analysis using numpy functions
    data = np.array(data)

    # Calculates the KS best fitting indicators (D and p), it fits evaluates
    # the difference in the fit points and the data points. Makes use of
    # the defined k_s_statistic function to extract these parameters. This
    # function is defined in the document and is provides a convinient way
    # to calculate the KS parameters

    # extracts the D and p values for the fit from the k_s_analysis variabl

    # formats the D and p values to be presentable on the plots as they have
    # a high number of floating points.
    if np.shape(data)[0] == 3:
        chi2_cheese = np.array([np.abs(data[0]),data[1],data[2]])
    elif np.shape(data)[1] == 2:
        chi2_cheese = np.array([np.abs(data[0]),data[1]])
    rsquared = r_squared(data,fit_function)
    rsquared_formatted = f'{rsquared:.3f}'
    chi2 = chi2_test(chi2_cheese)[0]
    dof = chi2_test(chi2_cheese)[1]
    chi2ndf = chi2/dof
    chi2ndf_formatted = f'{chi2ndf:.3f}'
    chi2_formatted = f'{chi2:.3f}'
    # defines the x array for the plot by extracting it from the inputted data,
    # for the specific case being analysed this will be the number of steps
    # array
    x_data = data[0]

    # defines the y array for the plot by extracting it from the inputted data
    # for the specific case being evaluated this will be the array of the
    # variances.
    y_data = data[1]

    # calculates the best fit parameters (popt) and the covariance matrix
    # (pcov) for the fitting function. This is done using the curve_fit
    # package provided by scipy
    popt, pcov = curve_fit(fit_function, data[0], data[1],sigma = data[2])
    
    shape = np.shape(pcov)[0]
    errors = np.zeros(shape)
    for i in range(shape):
        errors[i] = np.sqrt(pcov[i,i])
    
    # takes advantage of the covariance matrix to calculate estimates for
    # the errors on the best fit parameters. Calculates the standard errors
    # from the diagonals of the covariance matrix. Square rooting the
    # covariance matrix returns the standard error on the scale and
    # intercepts of the fit.
    
    plot_errors = [f'{error:.3f}' for error in errors]
    formatted_strings = []
    for i in range(len(popt)):
        formatted_str = f"$p_{i}= ({popt[i]:.3f} \\pm {plot_errors[i]})$"
        formatted_strings.append(formatted_str)
    # formats the errors to be more readable on the plot legend as they
    # have a large number of floating points.
    legend_entry = '\n'.join(formatted_strings)
    # creates a scatter plot of the x,y data to enable the regression
    # using the 1.5 exponent.
    plt.rcParams['figure.dpi'] = 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9))
    
    ax1.scatter(x_data, y_data, label='', marker = 's', s = 25)
    ax1.errorbar(x_data, y_data, yerr=data[2], fmt='o', markersize=5, linestyle='None', capsize=3, elinewidth=1)

    # adds the regression line to the plot, alongside formatted labels
    ax1.plot(x_data, fit_function(x_data, *popt), 'r', label=(
        f"$\\frac{{\\chi^2}}{{ndf}} = {chi2ndf_formatted}$\n"
        f'$R^{2} = {rsquared_formatted}$\n'
        f"{legend_entry}"))

    ax1.set_xlabel(x_label, fontsize=12)  # sets the x axis label for all plots
    ax1.set_ylabel(y_label, fontsize=12)  # sets the y axis label for all plots
    ax1.set_title(plot_title, fontsize=14)   # sets the plot title for all plots
    ax1.tick_params(axis='x', direction='in', length=10, width=1)  # Adjust length and width as needed
    ax1.tick_params(axis='y', direction='in', length=10, width=1)
    ax1.tick_params(axis='x', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    ax1.tick_params(axis='y', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    

    # displays the legend in a convinient and readable position using the bbox
    # anchor arguent and the location argument.
    
    ax1.legend(framealpha = 0.5, frameon = True, markerscale = 0, handleheight = 0, handlelength = 0, fontsize = 12)
    
    residuals = data[1] - fit_function(data[0],*popt)
    
    ax2.scatter(data[0], residuals, alpha=1,marker = 's', s = 25)
    ax2.errorbar(data[0], residuals, yerr=data[2], fmt='o', markersize=5, linestyle='None', capsize=3, elinewidth=1)
    ax2.axhline(y=0, linestyle='-', color ='red')
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel('Residuals', fontsize = 14)
    
    plt.subplots_adjust(hspace=0.2)
    ax2.tick_params(axis='x', direction='in', length=10, width=1)  # Adjust length and width as needed
    ax2.tick_params(axis='y', direction='in', length=10, width=1)
    ax2.tick_params(axis='x', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    ax2.tick_params(axis='y', which='minor', direction='in', length=5, width=1)  # Adjust length and width as needed
    plt.show()  # displays the relevant plot
    
    print_strings = []
    plot_errors_print = [f'{error}' for error in errors]
    
    for i in range(len(popt)):
        formatted_str = f"p_{i}= ({popt[i]} \\pm {plot_errors_print[i]})"
        print_strings.append(formatted_str)
    
    # formats the errors to be more readable on the plot legend as they
    # have a large number of floating points.
    legend_entry_print = '\n'.join(print_strings)

    print(
        f'------------------------------------\n'
        f'{label}\n'
        f'chi^2 per ndf = {chi2ndf}\n'
        f'R^2 = {rsquared}\n'
        f"{legend_entry_print}\n"
        f'------------------------------------'
    )



def error_plot(data, x_label, y_label, plot_title):
    plt.rcParams['figure.dpi'] = 1000
    plt.figure(figsize=(4,4))

    # Scatter plot for data points
    plt.scatter(data[0], data[1], label='Data', marker='s', s=25)
    
    # Plot regression line or curve (assuming data[0] represents x-axis)
    plt.plot(data[0], data[1], 'r', label='Regression')

    # Set labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(plot_title, fontsize=14)

    # Adjust ticks as needed
    plt.tick_params(axis='x', direction='in', length=10, width=1)
    plt.tick_params(axis='y', direction='in', length=10, width=1)
    plt.tick_params(axis='x', which='minor', direction='in', length=5, width=1)
    plt.tick_params(axis='y', which='minor', direction='in', length=5, width=1)

    # Display legend with appropriate settings
    plt.legend(framealpha=0.5, frameon=True, markerscale=1, fontsize=12)

    plt.show()  # Display the plot

    
    
    
def error_plot_gaussian(data, x_label, y_label, plot_title):
    plt.rcParams['figure.dpi'] = 1000
    plt.figure(figsize=(4,4))

    
    # Plot regression line or curve (assuming data[0] represents x-axis)
    plt.plot(data[0], data[1], 'r', label='Data')

    # Set labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(plot_title, fontsize=14)

    # Adjust ticks as needed
    plt.tick_params(axis='x', direction='in', length=10, width=1)
    plt.tick_params(axis='y', direction='in', length=10, width=1)
    plt.tick_params(axis='x', which='minor', direction='in', length=5, width=1)
    plt.tick_params(axis='y', which='minor', direction='in', length=5, width=1)

    # Display legend with appropriate settings
    plt.legend(framealpha=0.5, frameon=True, markerscale=1, fontsize=12)

    plt.show()  # Display the plot
    
def plotty_mcplotface(data):
    print('------------------------------------')
    print('-------- Plotty_McPlotFace v1 ------')
    print('------ THE ALL-IN-ONE PLOTTER! -----')
    print('------------------------------------')
    functions = [PowerLaw,linear,exponential,polynomial]
    r_squared_values = []
    cleaned_data = [[],[]]
    cleaned_data_err = [[],[],[]] 
    positive_data = [[],[]]
    positive_data_err = [[],[],[]]
    error_bar = input('Does the dataset contain errors on values? (y/n): ')


    
    clean = input("Would you like to clean the dataset by removing infinity values? [Recommended] (y/n): ")

    if clean == 'y' and error_bar == 'n':
        for i in range(len(data[0])):
            x,y = data[0][i], data[1][i]
            #if np.isinf(x) or np.isinf(y):
            if np.any(np.isinf(x)) or np.any(np.isinf(y)):
                print(f"--Inf At Index {i}--")
            elif not (np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(np.isnan(y)) or np.any(np.isinf(y))):
                cleaned_data[0].append(x)
                cleaned_data[1].append(y)
        data_arr = np.array(cleaned_data)
        pos_arr = data_arr
    found_infinity = False
    if clean == 'y' and error_bar == 'y':
        for i in range(len(data[0])):
            x,y,err = data[0][i], data[1][i],data[2][i]
            if np.any(np.isinf(x)) or np.any(np.isinf(y)):
                print(f"Inf At Index {i}")
                found_infinity = True 
            elif not np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
                cleaned_data_err[0].append(x)
                cleaned_data_err[1].append(y)
                cleaned_data_err[2].append(err)
        data_arr = np.array(cleaned_data_err)
        pos_err = data_arr
        if not found_infinity:
            print('--No Singularities Were Found--')
    if clean == 'n':
        pos_arr = np.array(data)
        cleaned_data = np.array(data)
        
    if error_bar =='n':
        for i in range(len(cleaned_data[0])):
            x,y = cleaned_data[0][i],cleaned_data[1][i]
            if y > 0:
                positive_data[0].append(x)
                positive_data[1].append(y)
        pos_arr = np.array(positive_data)
    if error_bar == 'y':
        for i in range(len(cleaned_data_err[0])):
            x,y,err = cleaned_data_err[0][i],cleaned_data_err[1][i],cleaned_data_err[2][i]
            if y > 0:
                positive_data_err[0].append(x)
                positive_data_err[1].append(y)
                positive_data_err[2].append(err)
        pos_arr = np.array(positive_data_err)
   
    x_label = input('X Label For Plot:')
    y_label = input('Y Label For Plot:')
    plot_title = input('Plot Title:')
    if error_bar == 'n':
        try:
            for function in functions:
                    try:
                        if function== exponential or PowerLaw:
                            rsq = r_squared(pos_arr, function)
                            r_squared_values.append(rsq)
                        else:
                            rsq = r_squared(data_arr, function)
                            r_squared_values.append(rsq)
                            
                    except Exception as e:
                        r_squared_values.append(0)
                        print(f'The {function.__name__} regression failed')
                        error_code= input("-Would You Like The Error Code? (y)/(n):")
                        if error_code == 'y':
                            print(e)
                        elif error_code == 'n':
                            pass
                        continue
            fit_function = functions[np.argmax(r_squared_values)]
            if fit_function == exponential or PowerLaw: 
                LSF(pos_arr, fit_function, x_label, y_label, plot_title,function.__name__)
                print('---------AUTO FIT COMPLETE -------')
            else:
                LSF(data_arr, fit_function, x_label,y_label, plot_title,function.__name__)
                print('---------AUTO FIT COMPLETE -------')
                
        except Exception as error:
            error_plot(data_arr,x_label,y_label,plot_title)
            print(f'-----The Data Did Not Match Any Of The Predefined Functions------')
            fit_error = input("-Would You Like The Error Code? (y)/(n):")
            if fit_error == 'y':
                print(error)
                print('---------AUTO FIT COMPLETE -------')
            elif fit_error == 'n':
                print('---------AUTO FIT COMPLETE -------')
        
        replot = input('-Would You Like to Refit: (lin),(exp),(pow),(pol),(n):')
        try:
            if replot == 'lin':
                LSF(pos_arr, linear, x_label, y_label, plot_title,'linear')
                print('---------Linear FIT COMPLETE -------')
            if replot == 'pow':
                LSF(pos_arr, PowerLaw, x_label, y_label, plot_title,'power law')
                print('---------Power Law FIT COMPLETE -------')
            if replot == 'exp':
                LSF(pos_arr, exponential, x_label, y_label, plot_title,'exponential')
                print('---------Exponential FIT COMPLETE -------')
            if replot == 'pol':
                LSF(pos_arr,polynomial, x_label, y_label, plot_title,'polynomial')
                print('---------Exponential FIT COMPLETE -------')
        except Exception as fail:
            print('---- Fit Failed -----')
    elif error_bar == 'y':
        try:
            for function in functions:
                    try:
                        if function== exponential or PowerLaw:
                            rsq = r_squared(pos_arr, function)
                            r_squared_values.append(rsq)
                        else:
                            rsq = r_squared(data_arr, function)
                            r_squared_values.append(rsq)
                            
                    except Exception as e:
                        r_squared_values.append(0)
                        print(f'The {function.__name__} regression failed')
                        error_code= input("-Would You Like The Error Code? (y)/(n):")
                        if error_code == 'y':
                            print(e)
                        elif error_code == 'n':
                            pass
                        continue
            fit_function = functions[np.argmax(r_squared_values)]
            if fit_function == exponential or PowerLaw: 
                error_LSF(pos_arr, fit_function, x_label, y_label, plot_title,function.__name__)
                print('---------AUTO FIT COMPLETE -------')
            else:
                error_LSF(data_arr, fit_function, x_label, y_label, plot_title,function.__name__)
                print('---------AUTO FIT COMPLETE -------')
                
        except Exception as error:
            error_plot(data_arr,x_label,y_label,plot_title)
            print(f'-----Unable to fit Due To Singularity------')
            fit_error = input("-Would You Like The Error Code? (y)/(n):")
            if fit_error == 'y':
                print(error)
                print('---------AUTO FIT COMPLETE -------')
            elif fit_error == 'n':
                print('---------AUTO FIT COMPLETE -------')
        
        replot = input('-Would You Like to Refit: (lin),(exp),(pow),(pol),(n):')
        try:
            if replot == 'lin':
                error_LSF(pos_arr, linear, x_label, y_label, plot_title,'linear')
                print('---------Linear FIT COMPLETE -------')
            if replot == 'pow':
                error_LSF(pos_arr, PowerLaw, x_label, y_label, plot_title,'power law')
                print('---------Power Law FIT COMPLETE -------')
            if replot == 'exp':
                error_LSF(pos_arr, exponential, x_label, y_label, plot_title,'exponential')
                print('---------Exponential FIT COMPLETE -------')
            if replot == 'pol':
                error_LSF(pos_arr, polynomial, x_label, y_label, plot_title,'polynomial')
                print('---------Exponential FIT COMPLETE -------')
        except Exception as fail:
            print('---- Fit Failed-----')

    print('---------------------------------------')
    print('------------ FIT COMPLETE ------------')
    print('-------- plotty_mcplotface v1 --------')
    print('-------- Created by Jack McQueen ------')
    print('---------------------------------------')




    
    
    
