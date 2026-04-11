# ==============================================================================
# Section 1: import libraries
# ==============================================================================

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mpl.rcParams['animation.embed_limit'] = 100  # MB

# ==============================================================================
# Section 2: beam parameters
# ==============================================================================

# Gaussian profile
w = 0.7e-3                    # beam waist [m]
lam = 775e-9                  # central wavelength [m]
c = 3e8                       # speed of light [m/s]
wavenumber = 2 * np.pi / lam  # central wavenumber [m^-1]
omega = 2 * np.pi * c / lam   # optical angular frequency [rad/s]
zr = wavenumber * w**2 / 2    # Rayleigh range [m]
tp = 85e-15                   # pulse duration [s]

# ==============================================================================
# Section 3: material parameters
# ==============================================================================

# air at STP
Pcr = 1.7e9                   # critical power [W]
ddot_k = 2 * (1e-15)**2 / (1e-2)   # GVD [s^2 m^-1]
n2 = 5.57e-23                 # Kerr coefficient [m^2 W^-1]
Pin = 2.5 * Pcr               # input power [W] (start moderate for testing)

# ---------- plasma / ionization parameters ----------
nb = 1.0                      # refractive index of air
sigma = 5.1e-24               # inverse bremsstrahlung cross section [m^2]
tau_c = 3.5e-13               # collision time [s]
Eg = 11.0 * 1.602176634e-19   # ionization energy [J]
hbar = 1.054571817e-34        # reduced Planck constant [J s]
alpha_rec = 5.0e-13           # recombination coefficient [m^3 s^-1]
K = 7                         # multiphoton order
betaK = 1e-96                 # MPI coefficient [tunable]
rho_cap = 1e25                # cap for numerical stability [m^-3]

# ==============================================================================
# Section 4: space mesh
# ==============================================================================

rmin = 0
rmax = 5 * w
Nr = 100
DeltaR = (rmax - rmin) / (Nr + 1)

# center time window around zero
tmin = -2.5 * tp
tmax =  2.5 * tp
Nt = 100
DeltaT = (tmax - tmin) / (Nt + 1)

zmin = 0
zmax = 2.0
Nz = 8000
DeltaZ = (zmax - zmin) / (Nz + 1)

radial_vector = np.linspace(rmin, rmax, Nr + 2)
time_vector = np.linspace(tmin, tmax, Nt + 2)
z_vector = np.linspace(zmin, zmax, Nz + 2)

radial_matrix, time_matrix = np.meshgrid(radial_vector, time_vector, indexing='ij')

# ==============================================================================
# Section 5: construct matrices
# ==============================================================================

delta = DeltaZ / (4 * wavenumber * DeltaR**2)
d = -ddot_k * DeltaZ / (4 * DeltaT**2)

main_plus_delta = [1 - 2j * delta for _ in range(Nr + 2)]
main_plus_delta[0] = 1 - 4j * delta
main_plus_delta[Nr + 1] = 0
upper_plus_delta = [4j * delta] + [1j * delta * (1 + 0.5 / k) for k in range(1, Nr + 1)]
lower_plus_delta = [1j * delta * (1 - 0.5 / k) for k in range(1, Nr + 1)] + [0]
diag1 = [lower_plus_delta, main_plus_delta, upper_plus_delta]
L_plus_delta = diags(diag1, [-1, 0, 1], format='csc')

main_minus_delta = [1 + 2j * delta for _ in range(Nr + 2)]
main_minus_delta[0] = 1 + 4j * delta
main_minus_delta[Nr + 1] = 1
upper_minus_delta = [-4j * delta] + [-1j * delta * (1 + 0.5 / k) for k in range(1, Nr + 1)]
lower_minus_delta = [-1j * delta * (1 - 0.5 / k) for k in range(1, Nr + 1)] + [0]
diag2 = [lower_minus_delta, main_minus_delta, upper_minus_delta]
L_minus_delta = diags(diag2, [-1, 0, 1], format='csc')

main_plus_d = [1 - 2j * d for _ in range(Nt + 2)]
main_plus_d[Nt + 1] = 0
upper_plus_d = [1j * d for _ in range(Nt + 1)]
upper_plus_d[Nt] = 0
lower_plus_d = [1j * d for _ in range(Nt + 1)]
lower_plus_d[0] = 2j * d
diag3 = [lower_plus_d, main_plus_d, upper_plus_d]
L_plus_d = diags(diag3, [-1, 0, 1], format='csc')

main_minus_d = [1 + 2j * d for _ in range(Nt + 2)]
main_minus_d[Nt + 1] = 1
upper_minus_d = [-1j * d for _ in range(Nt + 1)]
upper_minus_d[Nt] = 0
lower_minus_d = [-1j * d for _ in range(Nt + 1)]
lower_minus_d[0] = -2j * d
diag4 = [lower_minus_d, main_minus_d, upper_minus_d]
L_minus_d = diags(diag4, [-1, 0, 1], format='csc')

# ==============================================================================
# Section 6: initialize the electric field
# ==============================================================================

# Convention used here:
# |E|^2 is intensity [W/m^2]
amp = np.sqrt(2 * Pin / np.pi / w**2)

envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128)
envelope_electric_field[0] = amp * np.exp(
    - radial_matrix**2 / w**2
    - time_matrix**2 / tp**2
)

# ---------- plasma density storage ----------
rho_saved = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.float64)

# ==============================================================================
# Section 7: function to compute rho(r,t) at one z slice
# ==============================================================================

def compute_rho(E_field):
    """
    Computes electron density rho(r,t) from:
        d rho / d t =
            (sigma / (nb^2 * Eg)) * rho * |E|^2
            + betaK * |E|^(2K) / (K * hbar * omega)
            - alpha_rec * rho^2
    """
    rho_local = np.zeros_like(E_field.real, dtype=np.float64)
    intensity = np.abs(E_field)**2

    for j in range(Nt + 1):
        I = intensity[:, j]
        rho_j = rho_local[:, j]

        avalanche = (sigma / (nb**2 * Eg)) * rho_j * I
        mpi = betaK * (I**K) / (K * hbar * omega)
        recombination = alpha_rec * rho_j**2

        drho_dt = avalanche + mpi - recombination
        rho_next = rho_j + DeltaT * drho_dt

        # numerical protections
        rho_next = np.maximum(rho_next, 0.0)
        rho_next = np.minimum(rho_next, rho_cap)

        rho_local[:, j + 1] = rho_next

    return rho_local

# ==============================================================================
# Section 8: propagate the beam along z
# ==============================================================================

M1_lu = splu(L_minus_delta)
M2_lu_transpose = splu(L_minus_d.T)
L_plus_delta_transpose = L_plus_delta.T

# store rho at z = 0
rho_saved[0] = compute_rho(envelope_electric_field[0])

for k in range(Nz + 1):
    a = envelope_electric_field[k]
    rho_current = compute_rho(a)
    rho_saved[k] = rho_current

    # Kerr term
    kerr_term = 1j * wavenumber * n2 * (np.abs(a)**2) * a

    # ---------- plasma term ----------
    plasma_term = -0.5 * sigma * (1 + 1j * omega * tau_c) * rho_current * a

    if k == 0:
        A = a @ L_plus_d + DeltaZ * (kerr_term + plasma_term)
    else:
        b = envelope_electric_field[k - 1]
        rho_prev = rho_saved[k - 1]

        kerr_prev = 1j * wavenumber * n2 * (np.abs(b)**2) * b
        plasma_prev = -0.5 * sigma * (1 + 1j * omega * tau_c) * rho_prev * b

        A = a @ L_plus_d + DeltaZ * (
            1.5 * (kerr_term + plasma_term)
            - 0.5 * (kerr_prev + plasma_prev)
        )

    dummy = M1_lu.solve(A).T
    envelope_electric_field[k + 1] = M2_lu_transpose.solve(dummy @ L_plus_delta_transpose).T

# save final rho
rho_saved[Nz + 1] = compute_rho(envelope_electric_field[Nz + 1])

# ==============================================================================
# Section 9: diagnostics
# ==============================================================================

intensity = np.abs(envelope_electric_field)**2

M = np.nanmax(intensity[:, 0, :], axis=1)
normalized = M / amp**2

plt.figure()
plt.plot(z_vector, normalized)
plt.xlabel('z coordinate in m')
plt.ylabel(r'$\max_t |E(z,r=0,t)|^2 / \mathrm{amp}^2$')
plt.title('Maximum normalized on-axis intensity versus z')
plt.grid(True)
plt.show()

# ==============================================================================
# Section 10: show final intensity and final plasma density
# ==============================================================================

final_intensity = intensity[-1]
final_rho = rho_saved[-1]

plt.figure()
plt.imshow(
    final_intensity,
    origin='lower',
    aspect='auto',
    extent=[time_vector[0], time_vector[-1], radial_vector[0], radial_vector[-1]],
    vmin=0,
    vmax=np.percentile(final_intensity[np.isfinite(final_intensity)], 99)
)
plt.colorbar(label='Intensity [W/m^2]')
plt.xlabel('time coordinate [s]')
plt.ylabel('radial coordinate [m]')
plt.title(f'Final intensity at z = {z_vector[-1]:.3f} m')
plt.show()

plt.figure()
plt.imshow(
    final_rho,
    origin='lower',
    aspect='auto',
    extent=[time_vector[0], time_vector[-1], radial_vector[0], radial_vector[-1]],
    vmin=0,
    vmax=np.percentile(final_rho[np.isfinite(final_rho)], 99)
)
plt.colorbar(label=r'Electron density $\rho$ [m$^{-3}$]')
plt.xlabel('time coordinate [s]')
plt.ylabel('radial coordinate [m]')
plt.title(f'Final electron density at z = {z_vector[-1]:.3f} m')
plt.show()

# ==============================================================================
# Section 11: optional animation of intensity evolution
# ==============================================================================

# To make animation lighter, subsample in z
step_anim = max(1, Nz // 200)
intensity_anim = intensity[::step_anim]
z_anim = z_vector[::step_anim]

fig, ax = plt.subplots()

frame0 = intensity_anim[0]
im = ax.imshow(
    frame0,
    origin='lower',
    aspect='auto',
    extent=[time_vector[0], time_vector[-1], radial_vector[0], radial_vector[-1]],
    vmin=0,
    vmax=np.percentile(frame0[np.isfinite(frame0)], 99)
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Intensity [W/m^2]')

ax.set_xlabel('time coordinate [s]')
ax.set_ylabel('radial coordinate [m]')
title = ax.set_title(f'z = {z_anim[0]:.6g} m')

def update(frame):
    data = intensity_anim[frame]
    vmax = np.percentile(data[np.isfinite(data)], 99)
    if vmax <= 0:
        vmax = 1.0
    im.set_data(data)
    im.set_clim(0, vmax)
    title.set_text(f'z = {z_anim[frame]:.6g} m')
    return im, title

ani = FuncAnimation(fig, update, frames=len(z_anim), interval=60, blit=False)

from IPython.display import HTML
HTML(ani.to_jshtml())