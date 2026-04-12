# ==============================================================
# Section 1: import libraries
# ==============================================================

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
mpl.rcParams['animation.embed_limit'] = 100; # MB

# =============================================================
# Section 2: physical constants 
# =============================================================

c = 299792458; # light speed, in m s^{-1}
planck = 6.62607015e-34; # Planck's constant in J s
reduced_planck = planck / (2 * np.pi); # reduced Planck's constant in J s
eps0 = 8.854187818814e-12; # electrical permitivity of vacuum in F m^{-1}

# =============================================================
# Section 3: beam parameters
# =============================================================

lam = 775e-9; # central wavelength of the pulse, in m
wavenumber = 2 * np.pi / lam; # central wavenumber of the pulse, in m^{-1}
omega = 2 * np.pi * c / lam; # central pulsation of the puls, in rad s^{-1}
w = 0.7e-3; # beam waist of the pulse, in m
zR = wavenumber * w**2 / 2; # Rayleigh range of the pulse, in m
tp = 85e-15; # temporal length of the pulse, in s
ddot_k = 2 * (1e-15)**2 * (1e-2)**(-1); # group velocity dispersion in m^{-1} s^2

# =============================================================
# Section 4: material parameters
# =============================================================

# air at STP
n2 = 5.57e-23; # optical Kerr coefficient, in m^2 W^{-1}
Pcr = lam**2 / (2 * np.pi * n2); # critical power associated to the optical Kerr effect
Eg = 11 * 1.6e-19; # ionization energy of air in J
coeff_keldysh = np.ceil(Eg / (reduced_planck * omega)); # Keldysh coefficient associated to the ioniztion energy of air, no dimensions
beta_keldysh = 6.5e-104; # MPA coefficient from the Keldysh theory, in m^{11} W^{-6} 
Pin = 1 * Pcr; # initial power of the electric field, expressed in units of Pcr

# =============================================================
# Section 5: define the space mesh
# =============================================================

rmin = 0; rmax = 5 * w; # rmin is the origin of the radial coordinate system, rmax is the maximum extent of the radial coordinate system
Nr = 100; DeltaR = (rmax - rmin) / (Nr + 1); # Nr number of points between rmin and rmax (excluding rmin and rmax), DeltaR radial resolution
r_vector = np.linspace(rmin, rmax, Nr + 2);

tmin = 0; tmax = 5 * tp; 
Nt = 100; DeltaT = (tmax - tmin) / (Nt + 1);
t_vector = np.linspace(tmin, tmax, Nt + 2);

zmin = 0; zmax = 10; # in m
Nz = 100_000; DeltaZ = (zmax - zmin) / (Nz + 1);
z_vector = np.linspace(zmin, zmax, Nz + 2);

radial_matrix, time_matrix = np.meshgrid(r_vector, t_vector, indexing='ij');

# =============================================================
# Section 6: initialize the matrices
# =============================================================

# define the delta and d parameters
delta = DeltaZ / (4 * wavenumber * DeltaR**2); d = -ddot_k * DeltaZ / (4 * DeltaT**2);

main_plus_delta = np.array([1 - 4j*delta] + [1 - 2j*delta for k in range(1, Nr + 1)] + [0]);
upper_plus_delta = np.array([4j*delta] + [1j*delta*(1 + 0.5 / k) for k in range(1, Nr + 1)]);
lower_plus_delta = np.array([1j*delta*(1 - 0.5 / k) for k in range(1, Nr + 1)] + [0]);
diag1 = [lower_plus_delta, main_plus_delta, upper_plus_delta];
L_plus_delta = diags(diag1, [-1, 0, 1], format='csc', dtype=np.complex128);

main_minus_delta = np.array([1 + 4j*delta] + [1 + 2j*delta for k in range(1, Nr + 1)] + [1]);
upper_minus_delta = np.array([-4j*delta] + [-1j*delta*(1 + 0.5 / k) for k in range(1, Nr + 1)]);
lower_minus_delta = np.array([-1j*delta*(1 - 0.5 / k) for k in range(1, Nr + 1)] + [0]);
diag2 = [lower_minus_delta, main_minus_delta, upper_minus_delta];
L_minus_delta = diags(diag2, [-1, 0, 1], format='csc', dtype=np.complex128);

main_plus_d = np.array([1 - 2j*d for k in range(Nt + 1)] + [0]);
upper_plus_d = np.array([1j*d for k in range(Nt)] + [0]);
lower_plus_d = np.array([2j*d] + [1j*d for k in range(2, Nt + 2)]);
diag3 = [lower_plus_d, main_plus_d, upper_plus_d];
L_plus_d = diags(diag3, [-1, 0, 1], format='csc', dtype=np.complex128)

main_minus_d = np.array([1 + 2j*d for k in range(Nt + 1)] + [1]);
upper_minus_d = np.array([-1j*d for k in range(Nt)] + [0]);
lower_minus_d = np.array([-2j*d] + [-1j*d for k in range(2, Nt + 2)]);
diag4 = [lower_minus_d, main_minus_d, upper_minus_d];
L_minus_d = diags(diag4, [-1, 0, 1], format='csc', dtype=np.complex128);

# =============================================================
# Section 7: initialize the envelope of the electric field at z = 0, along r and t
# =============================================================

envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128);
envelope_electric_field[0] = np.sqrt(2 * Pin / (np.pi * w**2)) * np.exp(-radial_matrix**2 / w**2 - time_matrix**2 / tp**2);

# =============================================================
# Section 8: propagate the envelope of the electric field along z and find the intensity of the electric field
# =============================================================

M1 = splu(L_minus_delta); M2 = splu(L_minus_d.T); M3 = L_plus_delta.T;

for k in range(Nz + 1):
    if k == 0:
        E = envelope_electric_field[k]; 
        nonlinear = -0.5*beta_keldysh*np.abs(E)**(2*coeff_keldysh-2) * E + 1j*wavenumber*n2*np.abs(E)**2 * E
        RHS = E @ L_plus_d + DeltaZ * nonlinear
        intermediary_step = M1.solve(RHS).T
        envelope_electric_field[k + 1] = M2.solve(intermediary_step @ M3).T
    else:
        E = envelope_electric_field[k]; back_E = envelope_electric_field[k - 1]
        nonlinear = (1.5 * (-0.5*beta_keldysh*np.abs(E)**(2*coeff_keldysh-2) * E + 1j*wavenumber*n2*np.abs(E)**2 * E) 
        - 0.5 * (-0.5*beta_keldysh*np.abs(back_E)**(2*coeff_keldysh-2) * back_E + 1j*wavenumber*n2*np.abs(back_E)**2 * back_E))
        RHS = E @ L_plus_d + DeltaZ * nonlinear
        intermediary_step = M1.solve(RHS).T
        envelope_electric_field[k + 1] = M2.solve(intermediary_step @ M3).T


# define the intensity of the electric field
intensity = np.abs(envelope_electric_field) ** 2;


# ================================================================
# Section 9: make the animation for the intensity of the electric field
# ================================================================

fig, ax = plt.subplots()

im = ax.imshow(
    intensity[0],
    origin='lower',
    aspect='auto',
    extent=[t_vector[0], t_vector[-1], r_vector[0], r_vector[-1]],
    vmin=np.nanmin(intensity),
    vmax=np.nanmax(intensity)
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Intensity in V^2 / m^2')

ax.set_xlabel('time coordinate (s)')
ax.set_ylabel('radial coordinate (m)')
title = ax.set_title(f'z = {z_vector[0]:.6g} m')

def update(k):
    im.set_data(intensity[k])
    title.set_text(f'z = {z_vector[k]:.6g} m')
    return im, title

ani = FuncAnimation(fig, update, frames=Nz + 2, interval=50, blit=False)

from IPython.display import HTML
HTML(ani.to_jshtml())

# ================================================================
# Section 10: plot normalized max intensity 
# ================================================================

amp = np.sqrt(2 * Pin / (np.pi * w **2));

M = np.zeros(Nz + 2)
for k in range(Nz + 2):
    X = intensity[k]
    M[k] = np.nanmax(X) / amp**2

plt.xlabel('z coordinate (m)'); plt.ylabel('Normalized intensity $E^2/E_0^2$'); plt.title('Normalized intensity as a function of the z coordinate');
plt.plot(z_vector, M); plt.legend();