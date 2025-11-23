import numpy as np
from Modules.potentials import gaussian

slope_force = lambda x, drift_force: -drift_force # This force does not depend on location, also known as constant drift
harmonic_force = lambda x, k: -k*x
double_well_force = lambda x, k, a, b: -a*(k**2)*(x**3) + 2*a*b*k*x
double_well_guassians_force = lambda x, A, B, C, D: -(2*B*D*(D*x-C)*gaussian(x,A,B,C,D) + 2*B*D*(D*x+C)*gaussian(x,A,B,-C,D))
double_well_umbrella_demo_force = lambda x, H, W: -(4*H/W**4)*x*(x**2-W**2)
asymmetric_dw_force = lambda x, a, b, c, d: -4*a*x**3 + 2*b*x - 3*c*x**2
sinusoidal_noise_force = lambda x, a, b: -a*b*np.cos(b*x)