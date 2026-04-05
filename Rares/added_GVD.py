# ==============================================================================
# Section 1: import libraries
# ==============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
mpl.rcParams['animation.embed_limit'] = 50  # MB
import scipy as scp
from scipy.sparse.linalg import splu
from scipy.sparse import diags

# ==============================================================================
# Section 2: beam parameters
# ==============================================================================

# define beam parameters
# profile of the beam is Gaussian
w = 0.7e-3; # waist of the beam in m
lam = 775e-9; # central wavelength of the beam in m 
k = 2 * np.pi / lam; # central wavenumber in m^{-1}
zr = k*w**2/2; # Rayleigh length in m
tp = 85e-15; # length of the beam in vacuum in s

# ==============================================================================
# Section 3: material parameters
# ==============================================================================

# define material parameters
# here, the material is taken to be air at STP
Pcr = 1.7e9; # critical power of the material according to Mlejnek98
ddot_k = 2 * (1e-15) ** 2 * (1e2); # GVD for air 
Pin = 1*Pcr; # power of the beam expressed as units of the critical power

# ==============================================================================
# Section 4: space mesh
# ==============================================================================

# define the space mesh
# radial coordinate

rmin = 0; # origin of the radial coordinate in m
rmax = 5 * w; # maximum extent of the radial coordinate in m
Nr = 100; # number of radial points excluding rmin and rmax
DeltaR = (rmax - rmin) / (Nr + 1); # radial resolution or radial step
radial_vector = np.linspace(rmin, rmax, Nr + 2);

# time coordinate

tmin = 0; # origin of the time coordinate in s
tmax = 5 * tp; # maximum extent of the time coordinate in s
Nt = 100; # number of time points excluding tmin and tmax
DeltaT = (tmax - tmin) / (Nt + 1); # time resolution or time step
time_vector = np.linspace(tmin, tmax, Nt + 2);

# longitudinal coordinate (along propagation direction of the beam)

zmin = 0; # origin of the longitudinal coordinate in m
zmax = 10; # maximum extent of the longitudinal coordinate in m
Nz = 1000; # number of longitudinal points excluding zmin and zmax
DeltaZ = (zmax - zmin) / (Nz + 1); # longitudinal resolution or longitudinal step
z_vector = np.linspace(zmin, zmax, Nz + 2);

# radial and time matrices

time_matrix, radial_matrix = np.meshgrid(time_vector, radial_vector);

# ==============================================================================
# Section 5: define the matrices used to propagate the beam in the longitudinal direction
# ==============================================================================

delta = DeltaZ / (4 * k * DeltaR**2); # first useful parameter, associated with the radial laplacian
d = -ddot_k * DeltaZ / (4 * DeltaT**2); # second useful parameter, associated with the time double partial derivative

main_diag_plus_delta = [1 - 2j*delta for k in range(Nr + 2)];
main_diag_plus_delta[0] = 1 - 4j*delta;
main_diag_plus_delta[Nr + 1] = 0;
upper_diag_plus_delta = [4j * delta] + [1j*delta*(1 + 0.5/k) for k in range(1, Nr + 1)];
lower_diag_plus_delta = [1j*delta*(1 - 0.5/k) for k in range(1, Nr + 1)] + [0];
diag1 = [lower_diag_plus_delta, main_diag_plus_delta, upper_diag_plus_delta];
L_plus_delta = diags(diag1, [-1, 0, 1], format='csc'); # matrix of dimension Nr+2, complex coefficients, associated with the radial laplacian


main_diag_minus_delta = [1 + 2j*delta for k in range(Nr + 2)];
main_diag_minus_delta[0] = 1 + 4j*delta;
main_diag_minus_delta[Nr + 1] = 1;
upper_diag_minus_delta = [-4j * delta] + [-1j*delta*(1 + 0.5/k) for k in range(1, Nr + 1)];
lower_diag_minus_delta = [-1j*delta*(1 - 0.5/k) for k in range(1, Nr + 1)] + [0];
diag2 = [lower_diag_minus_delta, main_diag_minus_delta, upper_diag_minus_delta];
L_minus_delta = diags(diag2, [-1, 0, 1], format='csc'); # matrix of dimension Nr+2, complex coefficients, associated with the radial laplacian


main_diag_plus_d = [1 - 2j*d for k in range(Nt + 2)];
main_diag_plus_d[Nt + 1] = 0;
upper_diag_plus_d = [1j*d for k in range(Nt + 1)];
upper_diag_plus_d[Nt] = 0;
lower_diag_plus_d = [1j*d for k in range(Nt + 1)];
lower_diag_plus_d[0] = 2j*d;
diag3 = [lower_diag_plus_d, main_diag_plus_d, upper_diag_plus_d];
L_plus_d = diags(diag3, [-1, 0, 1], format='csc'); # matrix of dimension Nt+2, complex coefficients, associated to the time double partial derivative


main_diag_minus_d = [1 + 2j*d for k in range(Nt + 2)];
main_diag_minus_d[Nt + 1] = 1;
upper_diag_minus_d = [-1j*d for k in range(Nt + 1)];
upper_diag_minus_d[Nt] = 0;
lower_diag_minus_d = [-1j*d for k in range(Nt + 1)];
lower_diag_minus_d[0] = -2j*d;
diag4 = [lower_diag_minus_d, main_diag_minus_d, upper_diag_minus_d]; 
L_minus_d = diags(diag4, [-1, 0, 1], format='csc'); # matrix of dimension Nt+2, complex coefficients, associated to the time double partial derivative

# ==============================================================================
# Section 6: initialize the electric field envelope at zmin
# ==============================================================================

envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128);
amp = np.sqrt(2 * Pin / np.pi / w**2); 
envelope_electric_field[0] = amp * np.exp(-1/w**2 * radial_matrix**2 - 1/tp**2 * time_matrix**2); # based on initial conditions
X = envelope_electric_field[0].copy();

# ==============================================================================
# Section 7: propagate the electric field envelope recursively
# ==============================================================================

L_plus_delta_transpose = L_plus_delta.T;
M1_lu = splu(L_minus_delta);
M2_lu_transpose = splu(L_minus_d.T);
for k in range(Nz + 1):
    # solve using LU decomposition 
    B = M1_lu.solve(X @ L_plus_d).T;
    X = M2_lu_transpose.solve(B @ L_plus_delta_transpose).T;
    envelope_electric_field[k + 1] = X;


# ==============================================================================
# Section 8: compute then plot relative error wrt the known analytic solution 
# ==============================================================================

# define the exact analytic solution and compute the relative error

exact_envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128);
err = np.zeros((Nz + 2, Nr + 2, Nt + 2));
for k in range(Nz + 2):
    z = z_vector[k];
    e_t_z_t = np.exp( -time_matrix**2 / (tp**2*(1 - 2j*ddot_k*z/tp**2)) );
    term1 : complex = amp / (1 + 1j*z/zr); term2 : complex = 1 / np.sqrt(1 - 2j*ddot_k*z/tp**2); 
    term3  = np.exp( -radial_matrix**2 / (w**2*(1 + 1j*z/zr)) );
    e_r_z_r = term1 * term2 * term3;
    exact_envelope_electric_field[k] = e_r_z_r * e_t_z_t;
    den = np.abs(exact_envelope_electric_field[k]);
    err[k] = 100 * np.divide(np.abs(envelope_electric_field[k] - exact_envelope_electric_field[k]), den,
                       out = np.zeros_like(den), where=den != 0);

fig, ax = plt.subplots()

im = ax.imshow(
    err[0],
    origin='lower',
    aspect='auto',
    extent=[time_vector[0], time_vector[-1], radial_vector[0], radial_vector[-1]],
    vmin=np.nanmin(err),
    vmax=np.nanmax(err)
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Relative error in percent')

ax.set_xlabel('time coordinate (s)')
ax.set_ylabel('radial coordinate (m)')
title = ax.set_title(f'z = {z_vector[0]:.6g} m')

def update(k):
    im.set_data(err[k])
    title.set_text(f'z = {z_vector[k]:.6g} m')
    return im, title

ani = FuncAnimation(fig, update, frames=Nz + 2, interval=50, blit=False)

from IPython.display import HTML
HTML(ani.to_jshtml())