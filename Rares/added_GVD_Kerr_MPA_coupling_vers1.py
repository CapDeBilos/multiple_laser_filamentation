# ==============================================================
# Section 1: import libraries
# ==============================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.sparse import diags
from scipy.linalg import lu_factor, lu_solve
from numba import njit

mpl.rcParams['animation.embed_limit'] = 100  # MB

# =============================================================
# Section 2: physical constants
# =============================================================

c = 299792458.0                      # m s^-1
planck = 6.62607015e-34             # J s
reduced_planck = planck / (2.0 * np.pi)
eps0 = 8.854187818814e-12           # F m^-1
mass = 9.109383713928e-31           # kg
elem_charge = 1.602176634e-19       # C
kB = 1.380649e-23
p0 = 10**5; T0 = 273.15

# =============================================================
# Section 3: beam parameters
# =============================================================

lam = 775e-9
wavenumber = 2.0 * np.pi / lam
omega = 2.0 * np.pi * c / lam
w = 0.7e-3
zR = wavenumber * w**2 / 2.0
tp = 85e-15
ddot_k = 2.0 * (1e-15)**2 * (1e-2)**(-1)

# =============================================================
# Section 4: material parameters
# =============================================================

n2 = 5.57e-23
Pcr = lam**2 / (2.0 * np.pi * n2)
Eg = 11.0 * 1.6e-19
coeff_keldysh = int(np.ceil(Eg / (reduced_planck * omega)))
beta_keldysh = 6.5e-104
tau = 3.5e-13
a = 5e-13
sigma = (
    wavenumber * elem_charge**2 * tau
    / (omega * mass * eps0)
    / (1.0 + omega**2 * tau**2)
)
Pin = 6.5 * Pcr
rho0 = p0 / (kB * T0)

# =============================================================
# Section 5: define the space mesh
# =============================================================

rmin = 0.0
rmax = 5.0 * w
Nr = 100
DeltaR = (rmax - rmin) / (Nr + 1)
r_vector = np.linspace(rmin, rmax, Nr + 2)

tmin = 0.0
tmax = 5.0 * tp
Nt = 100
DeltaT = (tmax - tmin) / (Nt + 1)
t_vector = np.linspace(tmin, tmax, Nt + 2)

zmin = 0.0
zmax = 4
Nz = 100_000
DeltaZ = (zmax - zmin) / (Nz + 1)
z_vector = np.linspace(zmin, zmax, Nz + 2)

radial_matrix, time_matrix = np.meshgrid(r_vector, t_vector, indexing='ij')

# =============================================================
# Section 6: initialize the matrices
# =============================================================

delta = DeltaZ / (4.0 * wavenumber * DeltaR**2)
d = -ddot_k * DeltaZ / (4.0 * DeltaT**2)

main_plus_delta = np.array(
    [1.0 - 4j * delta] + [1.0 - 2j * delta for _ in range(1, Nr + 1)] + [0.0],
    dtype=np.complex128
)
upper_plus_delta = np.array(
    [4j * delta] + [1j * delta * (1.0 + 0.5 / k) for k in range(1, Nr + 1)],
    dtype=np.complex128
)
lower_plus_delta = np.array(
    [1j * delta * (1.0 - 0.5 / k) for k in range(1, Nr + 1)] + [0.0],
    dtype=np.complex128
)
L_plus_delta = diags(
    [lower_plus_delta, main_plus_delta, upper_plus_delta],
    offsets=[-1, 0, 1],
    format='csc',
    dtype=np.complex128
)

main_minus_delta = np.array(
    [1.0 + 4j * delta] + [1.0 + 2j * delta for _ in range(1, Nr + 1)] + [1.0],
    dtype=np.complex128
)
upper_minus_delta = np.array(
    [-4j * delta] + [-1j * delta * (1.0 + 0.5 / k) for k in range(1, Nr + 1)],
    dtype=np.complex128
)
lower_minus_delta = np.array(
    [-1j * delta * (1.0 - 0.5 / k) for k in range(1, Nr + 1)] + [0.0],
    dtype=np.complex128
)
L_minus_delta = diags(
    [lower_minus_delta, main_minus_delta, upper_minus_delta],
    offsets=[-1, 0, 1],
    format='csc',
    dtype=np.complex128
)

main_plus_d = np.array(
    [1.0 - 2j * d for _ in range(Nt + 1)] + [0.0],
    dtype=np.complex128
)
upper_plus_d = np.array(
    [1j * d for _ in range(Nt)] + [0.0],
    dtype=np.complex128
)
lower_plus_d = np.array(
    [2j * d] + [1j * d for _ in range(2, Nt + 2)],
    dtype=np.complex128
)
L_plus_d = diags(
    [lower_plus_d, main_plus_d, upper_plus_d],
    offsets=[-1, 0, 1],
    format='csc',
    dtype=np.complex128
)

main_minus_d = np.array(
    [1.0 + 2j * d for _ in range(Nt + 1)] + [1.0],
    dtype=np.complex128
)
upper_minus_d = np.array(
    [-1j * d for _ in range(Nt)] + [0.0],
    dtype=np.complex128
)
lower_minus_d = np.array(
    [-2j * d] + [-1j * d for _ in range(2, Nt + 2)],
    dtype=np.complex128
)
L_minus_d = diags(
    [lower_minus_d, main_minus_d, upper_minus_d],
    offsets=[-1, 0, 1],
    format='csc',
    dtype=np.complex128
)

# Small matrices: dense LU is usually faster than repeated sparse solves here.
A_r = np.ascontiguousarray(L_minus_delta.toarray())
B_r_T = np.ascontiguousarray(L_plus_delta.T.toarray())
B_t = np.ascontiguousarray(L_plus_d.toarray())
A_t = np.ascontiguousarray(L_minus_d.T.toarray())

lu_r, piv_r = lu_factor(A_r)
lu_t, piv_t = lu_factor(A_t)

# =============================================================
# Section 7: initialize the envelope, rho, and intensity
# =============================================================

envelope_electric_field = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.complex128)
rho = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.float64)

intensity = np.zeros((Nz + 2, Nr + 2, Nt + 2), dtype=np.float64)

amp0 = np.sqrt(2.0 * Pin / (np.pi * w**2))
envelope_electric_field[0] = amp0 * np.exp(
    -radial_matrix**2 / w**2 - time_matrix**2 / tp**2
)
intensity[0] = np.abs(envelope_electric_field[0])**2

# =============================================================
# Section 8: fast rho update
# =============================================================

@njit(cache=True, fastmath=True)
def update_rho_slice(
    rho_k,
    absE_k,
    Nr,
    Nt,
    DeltaT,
    sigma,
    Eg,
    beta_keldysh,
    coeff_keldysh,
    reduced_planck,
    omega,
    a,
):
    """
    rho_k is the electron (population) density and is forced to remain >= 0.
    """
    avalanche_pref = sigma * DeltaT / Eg
    mpa_pref = beta_keldysh * DeltaT / (coeff_keldysh * reduced_planck * omega)
    recomb_pref = a * DeltaT

    # boundary conditions
    for l in range(Nt + 2):
        rho_k[Nr + 1, l] = 0
    for j in range(Nr + 2):
        rho_k[j, Nt + 1] = 0

    # keep the initial time condition at the background
    for j in range(Nr + 2):
        rho_k[j, 0] = 0

    for j in range(Nr + 1):
        # first time step
        balance = rho_k[j, 0]
        Eabs = absE_k[j, 0]
        I = Eabs * Eabs
        Ipow = I**coeff_keldysh

        val = (
            balance
            + avalanche_pref * balance * I
            + mpa_pref * Ipow
            - recomb_pref * balance * balance
        )

        # rho must stay >= 0
        rho_k[j, 1] = val if val >= 0.0 else 0.0

        # Adams-Bashforth-like update
        for l in range(1, Nt):
            balance = rho_k[j, l]
            back_balance = rho_k[j, l - 1]

            Eabs = absE_k[j, l]
            back_Eabs = absE_k[j, l - 1]

            I = Eabs * Eabs
            back_I = back_Eabs * back_Eabs

            Ipow = I**coeff_keldysh
            back_Ipow = back_I**coeff_keldysh

            val = (
                balance
                + avalanche_pref * (1.5 * balance * I - 0.5 * back_balance * back_I)
                + mpa_pref * Ipow 
                - recomb_pref * (1.5 * balance * balance - 0.5 * back_balance * back_balance)
            )

            rho_k[j, l + 1] = val if val >= 0.0 else 0.0

# =============================================================
# Section 9: propagate the envelope of the electric field
# =============================================================

gamma = -0.5 * sigma * (1.0 + 1j * omega * tau)
mpa_field_pref = -0.5 * beta_keldysh
kerr_pref = 1j * wavenumber * n2

for k in range(Nz + 1):
    E = envelope_electric_field[k]

    # Update rho[k] from current field slice
    absE = np.abs(E)
    update_rho_slice(
        rho[k],
        absE,
        Nr,
        Nt,
        DeltaT,
        sigma,
        Eg,
        beta_keldysh,
        coeff_keldysh,
        reduced_planck,
        omega,
        a,
    )

    absE2 = absE * absE
    current_nonlinear = (
        mpa_field_pref * (absE2**(coeff_keldysh - 1)) * E
        + kerr_pref * absE2 * E
        + gamma * rho[k] * E
    )

    if k == 0:
        nonlinear = current_nonlinear
    else:
        back_E = envelope_electric_field[k - 1]
        back_absE = np.abs(back_E)
        back_absE2 = back_absE * back_absE

        previous_nonlinear = (
            mpa_field_pref * (back_absE2**(coeff_keldysh - 1)) * back_E
            + kerr_pref * back_absE2 * back_E
            + gamma * rho[k - 1] * back_E
        )

        nonlinear = 1.5 * current_nonlinear - 0.5 * previous_nonlinear

    RHS = E @ B_t + DeltaZ * nonlinear
    intermediary_step = lu_solve((lu_r, piv_r), RHS).T
    envelope_electric_field[k + 1] = lu_solve((lu_t, piv_t), intermediary_step @ B_r_T).T

    intensity[k + 1] = np.abs(envelope_electric_field[k + 1])**2

    if (k % 200) == 0:
        if not np.isfinite(intensity[k + 1]).all():
            print(f"Non-finite values detected at step {k+1}")
            break

# ================================================================
# Section 10: make the animation for the intensity
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

def update(frame_idx):
    im.set_data(intensity[frame_idx])
    title.set_text(f'z = {z_vector[frame_idx]:.6g} m')
    return im, title

ani = FuncAnimation(fig, update, frames=Nz + 2, interval=50, blit=False)

from IPython.display import HTML, display
display(HTML(ani.to_jshtml()))

# ================================================================
# Section 11: plot ratio of max intensity to I0
# ================================================================

M = np.max(intensity[:, 0, :], axis=1) / (amp0**2)

fig2, ax2 = plt.subplots()
ax2.plot(z_vector[z_vector <= 4], M[z_vector <= 4], label='max intensity / I0')
ax2.set_xlabel('z coordinate (m)')
ax2.set_ylabel(r'Relative intensity $E^2/E_0^2$')
ax2.set_title('Relative intensity as a function of the z coordinate')
ax2.legend()
plt.show()



# ===================================================================
# Section 12: smoothen out M using a Gaussian filter
# ===================================================================

from scipy.ndimage import gaussian_filter1d
M_smooth_ver2 = gaussian_filter1d(M, sigma = 1000)
plt.plot(z_vector, M_smooth_ver2)



# ======================================================================
# Section 13: smoothen out M using a bin averaging
# ======================================================================

# code the bin averaging

def bin_average_fast(z, f, bin_size):
    N = len(f)
    N_trim = (N // bin_size) * bin_size
    
    z_trim = z[:N_trim]
    f_trim = f[:N_trim]
    
    z_binned = z_trim.reshape(-1, bin_size).mean(axis=1)
    f_binned = f_trim.reshape(-1, bin_size).mean(axis=1)

    return z_binned, f_binned

z_smooth_ver3, M_smooth_ver3 = bin_average_fast(z_vector, M, 1000);
plt.plot(z_smooth_ver3, M_smooth_ver3)



