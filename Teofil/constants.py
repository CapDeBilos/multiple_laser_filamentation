# ============================================================
# Physical constants
# ============================================================
import numpy as np

elem_charge = 1.602176634e-19
electron_mass = 9.109383713928e-31
eps0 = 8.854187818814e-12
lightspeed = 299792458.0
planck = 6.62607015e-34
reduced_planck = planck / (2.0 * np.pi)
kB = 1.380649e-23

# Laser constants
lam = 775e-9
wavenumber = 2.0 * np.pi / lam
omega = wavenumber * lightspeed
waist = 0.7e-3
tp = 85e-15

# Material constants
nb = 1.0
n2 = 5.57e-23
tau = 3.5e-13
sigma = wavenumber * elem_charge**2 * tau / (omega * electron_mass * eps0) / (1.0 + omega**2 * tau**2)
Pcr = lam**2 / (2.0 * np.pi * nb * n2)
beta_k = 6.5e-104
K = 7

pressure = 101_325.0
temperature = 25.0 + 273.15
rho_neutral = 2.0 * pressure / (kB * temperature)

# Gaussian-in-time effective integration factor used in the artificial plasma model.
plasma_time_factor = 0.5 * np.sqrt(np.pi) * tp / np.sqrt(2.0 * K)