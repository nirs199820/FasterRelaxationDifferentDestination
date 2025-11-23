import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate as integrate

def histogram_potential_fitter(x_range, histogram, normalized_pdf, args=[], initial_guess=0, bounds_array=()):
    normalized_pdf_for_fit = lambda x, fitting_param: normalized_pdf(x, fitting_param,*args)
    popt, pcov = curve_fit(normalized_pdf_for_fit, x_range, histogram, p0=[initial_guess], bounds=bounds_array)
    parameter_from_fit = popt[0]
    return parameter_from_fit

def normalized_pdf_from_potential(x_range, potential, kT, fitting_arg ,potential_args=[]):
    un_normalized_pdf = lambda x, fitting_argument, args=[]: np.exp(-potential(x,fitting_argument,*args)/kT)
    normalization_factor = integrate.quad(un_normalized_pdf,x_range[0], x_range[-1], args=(fitting_arg, potential_args))[0]
    normalized_pdf = lambda x, fitting_argument, args=[]: un_normalized_pdf(x, fitting_argument, args)/normalization_factor
    if x_range[0] > 0:
        left_bound = -x_range[0]
    else: 
        left_bound = 3*x_range[0]
    if x_range[-1] > 0:
        right_bound = 3*x_range[-1]
    else:
        right_bound = -x_range[-1]
    normalization_score = 100-100*np.abs((integrate.quad(normalized_pdf, left_bound, right_bound, args=(fitting_arg, potential_args))[0] - 1))
    return normalized_pdf, normalization_score

def generate_theoretical_pdf_for_fit(function, args):
  f = lambda x, fitting_argument: function(x, fitting_argument, *args)
  return f

def shift_potential(potential, shift=0, args=[]):
    shifted = lambda x, *args: potential(x+shift, *args)
    return shifted