# ==============================================================================
# Section 1: import libraries
# ==============================================================================

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
mpl.rcParams['animation.embed_limit'] = 100 # MB

# ==============================================================================
# Section 2: beam parameters
# ==============================================================================

# Gaussian profile
w = 0.7e-3; # beam waist in m
lam = 775e-9; # central wavelength of the beam in m
wavenumber = 2*np.pi / lam; # central wavenumber of the beam in m^{-1}
zr = wavenumber * w**2 / 2; # Rayleigh range of the beam in m
tp = 85e-15; # time length of the beam in vacuum in m

# ==============================================================================
# Section 3: material parameters
# ==============================================================================

# air at STP
Pcr = 1.7e9; # critical power for the beam and environment in W
ddot_k = 2 * (1e-15)**2 / (1e-2); # group velocity dispersion in s^2 m ^{-1}
n2 = 5.57e-23; # optical kerr effect coefficient in m^2 W^{-1}
eps0 = 8.85e-12; # electrical permitivity of vacuum in F m^{-1}
c = 3e8; # light speed in m s^{-1}
Pin = 6.5*Pcr; # beam power in vacuum, in units of Pcr

# ==============================================================================
# Section 4: space mesh
# ==============================================================================

rmin = 0; # origin of the radial coordinate in m
rmax = 5 * w; # maximum extent of the radial coordinate in m
Nr = 100; # number of points between rmin and rmax, excluding rmin and rmax
DeltaR = (rmax - rmin) / (Nr + 1); # radial resolution in m

tmin = 0; # origin of the time coordinate in s
tmax = 5 * tp; # maximum extent of the time coordinate in s
Nt = 100; # number of points between tmin and tmax, excluding tmin and tmax
DeltaT = (tmax - tmin) / (Nt + 1); # time resolution in s

zmin = 0; # origin of the longitudinal coordinate in m
zmax = 10; # maximum extent of the longitudinal coordinate in m
Nz = 1000; # number of points between zmin and zmax, excluding zmin and zmax
DeltaZ = (zmax - zmin) / (Nz + 1); # longitudinal resolution in m

radial_vector = np.linspace(rmin, rmax, Nr + 2); # this vector stores all the radial coordinates
time_vector = np.linspace(tmin, tmax, Nt + 2); # this vector stores all the time coordinates
z_vector = np.linspace(zmin, zmax, Nz + 2); # this vector stores all the longitudinal coordinates

radial_matrix, time_matrix = np.meshgrid(radial_vector, time_vector, indexing='ij');

# ==============================================================================
# Section 5: construct matrices
# ==============================================================================

delta = DeltaZ / (4 * wavenumber * DeltaR**2); # parameter appearing in the expression of the laplacian matrix acting on the radial component
d = -ddot_k * DeltaZ / (4 * DeltaT**2); # parameter appearing in the expression of the double time partial derivative matrix

main_plus_delta = [1 - 2j*delta for k in range(Nr + 2)];
main_plus_delta[0] = 1 - 4j*delta; main_plus_delta[Nr + 1] = 0;
upper_plus_delta = [4j*delta] + [1j*delta*(1 + 0.5/k) for k in range(1, Nr + 1)];
lower_plus_delta = [1j*delta*(1 - 0.5/k) for k in range(1, Nr + 1)] + [0];
diag1 = [lower_plus_delta, main_plus_delta, upper_plus_delta];
L_plus_delta = diags(diag1, [-1, 0, 1], format='csc');

main_minus_delta = [1 + 2j*delta for k in range(Nr + 2)];
main_minus_delta[0] = 1 + 4j*delta; main_minus_delta[Nr + 1] = 1;
upper_minus_delta = [-4j*delta] + [-1j*delta*(1 + 0.5/k) for k in range(1, Nr + 1)];
lower_minus_delta = [-1j*delta*(1 - 0.5/k) for k in range(1, Nr + 1)] + [0];
diag2 = [lower_minus_delta, main_minus_delta, upper_minus_delta];
L_minus_delta = diags(diag2, [-1, 0, 1], format='csc');

main_plus_d = [1 - 2j*d for k in range(Nt + 2)];
main_plus_d[Nt + 1] = 0;
upper_plus_d = [1j*d for k in range(1, Nt + 1)] + [0];
lower_plus_d = [1j*d for k in range(Nt + 1)];
lower_plus_d[0] = 2j*d;
diag3 = [lower_plus_d, main_plus_d, upper_plus_d];
L_plus_d = diags(diag3, [-1, 0, 1], format='csc');

main_minus_d = [1 + 2j*d for k in range(Nt + 2)];
main_minus_d[Nt + 1] = 1;
upper_minus_d = [-1j*d for k in range(1, Nt + 1)] + [0];
lower_minus_d = [-1j*d for k in range(Nt + 1)];
lower_minus_d[0] = -2j*d;
diag4 = [lower_minus_d, main_minus_d, upper_minus_d];
L_minus_d = diags(diag4, [-1, 0, 1], format='csc');

# ==============================================================================
# Section 6: initialize the electric field with initial conditions
# ==============================================================================

envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128);
amp = np.sqrt(2*Pin / np.pi / w**2);
envelope_electric_field[0] = amp * np.exp(- radial_matrix**2 / w**2 - time_matrix**2 / tp**2);

# ==============================================================================
# Section 7: propagate the beam along the longitudinal coordinate
# ==============================================================================

M1_lu = splu(L_minus_delta);
M2_lu_transpose = splu(L_minus_d.T);
L_plus_delta_transpose = L_plus_delta.T;

for k in range(Nz + 1):
    if k == 0:
        a = envelope_electric_field[k];
        A = a @ L_plus_d + 1j * wavenumber * n2 * DeltaZ * (1 * (np.abs(a))**2 * a);
        dummy = M1_lu.solve(A).T;
        envelope_electric_field[k + 1] = M2_lu_transpose.solve(dummy @ L_plus_delta_transpose).T;
    else:
        a = envelope_electric_field[k]; b = envelope_electric_field[k - 1];
        A = a @ L_plus_d + 1j * wavenumber * n2 * DeltaZ * (1.5 * (np.abs(a))**2 * a - 0.5 * (np.abs(b))**2 * b);
        dummy = M1_lu.solve(A).T;
        envelope_electric_field[k + 1] = M2_lu_transpose.solve(dummy @ L_plus_delta_transpose).T;

# ==============================================================================
# Section 8: animate the results for intensity
# ==============================================================================

intensity = 0.5*eps0*c * np.abs(envelope_electric_field) ** 2; 

fig, ax = plt.subplots()

im = ax.imshow(
    intensity[0],
    origin='lower',
    aspect='auto',
    extent=[time_vector[0], time_vector[-1], radial_vector[0], radial_vector[-1]],
    vmin=np.nanmin(intensity),
    vmax=np.nanmax(intensity)
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Intensity in W / m^2')

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

# ==============================================================================
# Section 9: animate the results for the envelope of the electric field
# ==============================================================================

M = np.nanmax(np.abs(envelope_electric_field)[:, 0, :], axis=1);
normalized = M ** 2 / amp ** 2;

plt.plot(z_vector, normalized);
plt.xlabel('z coordinate in m'); plt.ylabel('$\\max_{t}|E(z, r=0, t)/\\text{amp}|^2$');
plt.title('Maximum value of the normalized intensity at $r=0$ as a function of the longitudinal coordinate z');