import mpmath as mp
import warnings
import numpy as np

def pdf_resetting_in_harmonic_potential(x_range, k_trap, delta_fraction, shift, dt=0.01, kT=1, gamma=1, resetting_position=0):
  '''
  Reference:
  Goerlich, R., Li, M., Pires, L. B., Hervieux, P.-A., Manfredi, G., & Genet, C. (2023). 
  Experimental test of Landauer's principle for stochastic resetting. 
  arXiv. https://arxiv.org/abs/2306.09503
  (Equation D9)
  '''
  mp.dps = 15; mp.pretty = True
  delta = dt
  k = k_trap  # in range of 10^-5         |kg/s^2
  D = kT/gamma #                         |m^2/s
  w_0 = k/gamma #
  resetting_param = delta_fraction*(1/delta)
  alpha = np.sqrt(resetting_param/D)  #    |1/m
  a = w_0/(2*D) #                          |1/m^2
  b = 2*w_0 #                              |1/s
  p = b/resetting_param #                  |unitless
  consts = (1/p)*(np.sqrt(a/np.pi))*mp.gamma(1/p)
  kappa = 0.25-(1/p)
  mu = 0.25
  epsilon = 1e-12

  whittaker = np.vectorize(mp.whitw)
  res = []
  for x in x_range:
    with warnings.catch_warnings():
      try:
          res_i = consts*np.exp(-0.5*a*((x-shift)**2))*np.power(a*((x-shift)**2),-0.25)*whittaker(kappa,mu,a*((x-shift)**2))
          res.append(res_i)
      except Warning as e:
          print('error found:', e)
          res.append(epsilon)
  return np.array(res, dtype=np.float64)

def pdf_resetting_with_constant_drift(x_range, drift_force, delta_fraction, shift, dt=0.01, kT=1, gamma=1, resetting_position=0):
  '''
  Reference:
  Evans, M. R., Majumdar, S. N., & Schehr, G. (2020). 
  Stochastic resetting and applications. 
  Journal of Physics A: Mathematical and Theoretical, 53(19), 193001. https://doi.org/10.1088/1751-8121/ab7cfe
  (Equation 2.22)
  * There is a mistake in the equation. The denominator in the exponent should be 2D and not D *
  '''
  mp.dps = 15; mp.pretty = True
  delta = dt
  mu = drift_force/gamma
  D = kT/gamma #                         |m^2/s
  resetting_param = delta_fraction*(1/delta)

  result = lambda x: (resetting_param/np.sqrt(4*resetting_param*D+mu**2))*np.exp((mu*x/(2*D))-np.abs(x)*np.sqrt(4*resetting_param*D+mu**2)/(2*D))

  x_range = x_range - shift
  if isinstance(x_range, int):
    x = x_range
    return result(x_range)
  elif isinstance(x_range, float):
    x = x_range
    return result(x_range)

  res = []
  for x in x_range:
    res_i = result(x)
    res.append(res_i)
  return np.array(res, dtype=np.float64)

def pdf_harmonic_boltzmann(k, kT) -> callable:
  pdf = lambda x: np.exp(-0.5*k*x**2/kT)/(np.sqrt(2*np.pi*kT/k))
  return pdf