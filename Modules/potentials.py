import numpy as np

slope_potential = lambda x, drift_force: drift_force*x # Drift in the positive x direction
harmonic_potential = lambda x, k: 0.5*k*x**2
double_well_potential = lambda x, k, a, b: a*((0.5*k*(x**2)-b)**2)
gaussian = lambda x, A, B, C, D: A*np.exp(-B*(D*x-C)**2)
double_well_gaussians = lambda x, A, B, C, D, offset: offset-(gaussian(x,A,B,C,D)+gaussian(x,A,B,-C,D))
double_well_umbrella_demo = lambda x, H, W: (H/W**4)*(x**2-W**2)**2
asymmetric_dw_potential = lambda x, a, b, c, d: a*x**4-b*x**2+c*x**3+d
sinusoidal_noise = lambda x, a, b: a*np.sin(b*x)
