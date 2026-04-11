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

w      = 0.7e-3           # beam waist [m]
lam    = 775e-9           # central wavelength [m]
k0     = 2 * np.pi / lam  # central wavenumber [m^-1]
zr     = k0 * w**2 / 2   # Rayleigh range [m]
tp     = 85e-15           # pulse half-duration (1/e in intensity) [s]

# ==============================================================================
# Section 3: material parameters (air at STP, Mlejnek OL 1998)
# ==============================================================================

Pcr      = 1.7e9                          # critical power for SF [W]
ddot_k   = 2e-15**2 / 1e-2              # GVD k'' [s^2 m^-1]  (2 fs^2/cm)
n2       = 5.57e-23                       # nonlinear Kerr index [m^2 W^-1]
eps0     = 8.85e-12                       # permittivity of vacuum [F m^-1]
c        = 3e8                            # speed of light [m s^-1]
Pin      = 6.5 * Pcr                      # input peak power [W]

# --- MPA / multiphoton ionization parameters ---
# From Mlejnek OL 1998: K = 7 (air, E_gap ~ 11 eV, lambda = 775 nm)
# beta^(7) = 6.5e-104 m^11 W^-6   [units: m^(2K-3) W^(1-K)]
K_mpa    = 7                              # number of photons absorbed per ionization event
beta_K   = 6.5e-104                       # K-photon absorption coefficient [m^(2K-3) W^(1-K)]

# ==============================================================================
# Section 4: space mesh
# ==============================================================================

rmin   = 0.0;   rmax = 5 * w
Nr     = 100
DeltaR = (rmax - rmin) / (Nr + 1)
radial_vector = np.linspace(rmin, rmax, Nr + 2)

tmin   = 0.0;   tmax = 5 * tp
Nt     = 100
DeltaT = (tmax - tmin) / (Nt + 1)
time_vector = np.linspace(tmin, tmax, Nt + 2)

zmin   = 0.0;   zmax = 10.0
# Nz = 10000 is required here (not 1000 as in added_GVD_Kerr.py).
# Reason: with Pin = 6.5 Pcr the Kerr self-focusing drives the on-axis
# intensity up by ~4–5x before MPA arrests it.  The nonlinear phase per
# z-step scales as k0*n2*I*DeltaZ; at 5x peak intensity this reaches
# ~0.3 rad with Nz=1000, which is too large for the explicit AB-2 step
# and causes numerical blowup.  Nz=10000 keeps the phase per step < 0.07 rad
# throughout the propagation.
Nz     = 10000
DeltaZ = (zmax - zmin) / (Nz + 1)
z_vector = np.linspace(zmin, zmax, Nz + 2)

radial_matrix, time_matrix = np.meshgrid(radial_vector, time_vector, indexing='ij')

# ==============================================================================
# Section 5: construct Crank-Nicolson matrices
# ==============================================================================

# --- radial diffraction ---
# delta = DeltaZ / (4 k0 DeltaR^2)  appears in the discretised transverse Laplacian
delta = DeltaZ / (4 * k0 * DeltaR**2)

main_plus_delta  = [1 - 2j*delta] * (Nr + 2)
main_plus_delta[0]    = 1 - 4j*delta
main_plus_delta[Nr+1] = 0
upper_plus_delta = [4j*delta] + [1j*delta*(1 + 0.5/k) for k in range(1, Nr+1)]
lower_plus_delta = [1j*delta*(1 - 0.5/k) for k in range(1, Nr+1)] + [0]
L_plus_delta  = diags([lower_plus_delta, main_plus_delta, upper_plus_delta], [-1,0,1], format='csc')

main_minus_delta  = [1 + 2j*delta] * (Nr + 2)
main_minus_delta[0]    = 1 + 4j*delta
main_minus_delta[Nr+1] = 1
upper_minus_delta = [-4j*delta] + [-1j*delta*(1 + 0.5/k) for k in range(1, Nr+1)]
lower_minus_delta = [-1j*delta*(1 - 0.5/k) for k in range(1, Nr+1)] + [0]
L_minus_delta = diags([lower_minus_delta, main_minus_delta, upper_minus_delta], [-1,0,1], format='csc')

# --- GVD in time ---
# d = -k'' DeltaZ / (4 DeltaT^2)  appears in the discretised d^2/dt^2
d = -ddot_k * DeltaZ / (4 * DeltaT**2)

main_plus_d  = [1 - 2j*d] * (Nt + 2)
main_plus_d[Nt+1] = 0
upper_plus_d = [1j*d] * (Nt + 1) + [0]
lower_plus_d = [1j*d] * (Nt + 1)
lower_plus_d[0] = 2j*d
L_plus_d  = diags([lower_plus_d, main_plus_d, upper_plus_d], [-1,0,1], format='csc')

main_minus_d  = [1 + 2j*d] * (Nt + 2)
main_minus_d[Nt+1] = 1
upper_minus_d = [-1j*d] * (Nt + 1) + [0]
lower_minus_d = [-1j*d] * (Nt + 1)
lower_minus_d[0] = -2j*d
L_minus_d = diags([lower_minus_d, main_minus_d, upper_minus_d], [-1,0,1], format='csc')

# ==============================================================================
# Section 6: initial conditions
# ==============================================================================

envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128)
amp = np.sqrt(2 * Pin / (np.pi * w**2))
envelope_electric_field[0] = amp * np.exp(-radial_matrix**2 / w**2
                                          - time_matrix**2  / tp**2)

# ==============================================================================
# Section 7: propagation
#
# The GNLSE (Mlejnek OL 1998, eq. 1) in the moving frame reads
#
#   dE/dz = (i/2k0) nabla_perp^2 E
#           - (i k''_0 / 2) d^2E/dt^2
#           + i k0 n2 |E|^2 E                      [Kerr]
#           - (beta_K / 2) |E|^(2K-2) E            [MPA]
#
# The first two (linear) terms are handled by the Crank-Nicolson split
# already in place.  The last two (nonlinear) terms are grouped as a
# single local operator N(E) and advanced with an Adams-Bashforth
# predictor:
#
#   N(E) = i k0 n2 |E|^2 E  -  (beta_K/2) |E|^(2K-2) E
#
# Step k=0 (no previous step available): forward Euler,
#   A = L+_delta . E . L+_d  +  DeltaZ * N(E)
# Step k>0: AB-2 extrapolation (avoids phase lag),
#   A = L+_delta . E . L+_d  +  DeltaZ * (3/2 N(E_k) - 1/2 N(E_{k-1}))
#
# The nonlinear term is treated explicitly because it does not couple
# neighbouring grid points.  The Crank-Nicolson matrices are unchanged
# by the addition of MPA.
# ==============================================================================

def nonlinear_term(E):
    """
    Evaluate the nonlinear source at a single z-slice.

    Kerr:  i k0 n2 |E|^2 E
    MPA:   -(beta_K/2) |E|^(2K-2) E

    Note on units:  |E|^2 has units of W m^-2 (intensity in the plane-wave
    convention used here, I = eps0 c |E|^2 / 2, but the code absorbs that
    1/2 into amp at initialisation).  beta_K carries the compensating units
    m^(2K-3) W^(1-K) so the product beta_K |E|^(2K-2) is dimensionless per
    metre, consistent with a z-derivative of the field envelope.

    I_cap: float64 overflows at I^6 when I > ~2.6e51 W/m^2, which is
    unphysical (optical breakdown in air is ~1e17 W/m^2 = 1e13 W/cm^2).
    We cap I before raising to (K-1) to prevent NaN propagation during
    any transient numerical overshoot.  The cap is set well above physical
    filament intensities so it only activates if the field has already
    diverged numerically.
    """
    I       = np.abs(E)**2                          # |E|^2  [W m^-2]
    I_safe  = np.minimum(I, 1e22)                   # overflow guard for I^(K-1)
    kerr_term = 1j * k0 * n2 * I * E
    mpa_term  = -(beta_K / 2.0) * I_safe**(K_mpa - 1) * E
    return kerr_term + mpa_term

# LU factorisations (computed once, reused every step)
M1_lu            = splu(L_minus_delta)
M2_lu_transpose  = splu(L_minus_d.T)
L_plus_delta_T   = L_plus_delta.T

for k in range(Nz + 1):
    E_now = envelope_electric_field[k]
    N_now = nonlinear_term(E_now)

    if k == 0:
        # Forward-Euler predictor for the first step
        NL = DeltaZ * N_now
    else:
        # Adams-Bashforth 2nd-order predictor
        E_prev = envelope_electric_field[k - 1]
        N_prev = nonlinear_term(E_prev)
        NL = DeltaZ * (1.5 * N_now - 0.5 * N_prev)

    # Apply linear operators (Crank-Nicolson split: diffraction first, then GVD)
    A     = E_now @ L_plus_d + NL
    dummy = M1_lu.solve(A).T
    envelope_electric_field[k + 1] = M2_lu_transpose.solve(dummy @ L_plus_delta_T).T

# ==============================================================================
# Section 8: diagnostics
# ==============================================================================

intensity = 0.5 * eps0 * c * np.abs(envelope_electric_field)**2

# --- Plot 1: animation of intensity in (r, t) plane as z advances ---
fig1, ax1 = plt.subplots()
im = ax1.imshow(
    intensity[0],
    origin='lower',
    aspect='auto',
    extent=[time_vector[0], time_vector[-1], radial_vector[0], radial_vector[-1]],
    vmin=np.nanmin(intensity),
    vmax=np.nanmax(intensity)
)
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Intensity [W m$^{-2}$]')
ax1.set_xlabel('t  [s]')
ax1.set_ylabel('r  [m]')
title1 = ax1.set_title(f'z = {z_vector[0]:.4g} m')

def update(k):
    im.set_data(intensity[k])
    title1.set_text(f'z = {z_vector[k]:.4g} m')
    return im, title1

ani = FuncAnimation(fig1, update, frames=Nz + 2, interval=50, blit=False)

# --- Plot 2: on-axis peak intensity vs z (key filamentation diagnostic) ---
peak_onaxis = np.max(np.abs(envelope_electric_field)[:, 0, :]**2, axis=1)
I_peak_onaxis = 0.5 * eps0 * c * peak_onaxis

fig2, ax2 = plt.subplots()
ax2.plot(z_vector, I_peak_onaxis / (0.5 * eps0 * c * amp**2))
ax2.set_xlabel('z  [m]')
ax2.set_ylabel(r'$\max_t \; I(z, r{=}0, t) \;/\; I_0$')
ax2.set_title('On-axis peak intensity (normalised) — MPA + Kerr + GVD + diffraction')
ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', label='input level')
ax2.legend()
plt.tight_layout()

try:
    from IPython.display import HTML
    HTML(ani.to_jshtml())
except ImportError:
    plt.show()
