#!/usr/bin/env python3
"""
Memory-optimized adaptive Cartesian FFT split-step sweep in transverse Cartesian
coordinates (x, y), with propagation along z. The temporal pulse is included
through the artificial-plasma effective time factor, not as an explicit grid axis.

Compared with THOMAS_PCR_adaptive_FFT_sweep_xy_no_t.py, this version reduces GPU
memory by:
- storing 1D x/y/kx/ky vectors instead of full x_matrix/y_matrix/kx_matrix/ky_matrix;
- using broadcasting when building E, absorber, and FFT propagator;
- removing the persistent I_pow array;
- computing I, rho, and nonlinear in one CUDA RawKernel pass;
- computing l_Kerr and l_MPA from Imax instead of a full I**(K-1) array;
- deleting temporary arrays before remeshing and output when possible.

Equation implemented:
    dE/dz = i/(2k) (d_xx + d_yy) E
            + i*k*n2*I*E
            - 0.5*beta*I**(K-1)*E
            - 0.5*sigma*(1 + i*omega*tau)*rho*E

Artificial plasma model:
    b = beta * I**K / (K * hbar * omega * rho_neutral)
    rho = rho_neutral * (1 - exp(-b * 0.5 * sqrt(pi) * tp / sqrt(2*K)))

Run example:
    python Hermite_4D_FFT_artificial_plasma_simulation.py 1.1 2.0
"""

import argparse
import os
import gc
import numpy as np
import cupy as cp  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
from scipy.special import gamma, hermite

# ============================================================
# User switches
# ============================================================
NONLINEAR_ON = True

number_diag = 100
full_I_save_every = 10 * number_diag  # save full 2D I less often than scalar diagnostics

# RMS remeshing controls
REMESH_ON = True
target_points_x = 50
target_points_y = 50
max_Nx = 2 * 4096
max_Ny = 2 * 4096
remesh_check_every = 50

# Absorbing boundary controls
ABSORB_ON = True
absorb_frac_x = 0.15
absorb_frac_y = 0.15
absorb_strength_x = 8.0
absorb_strength_y = 8.0
absorb_power = 4

# Initial transverse grid
initial_Nx = 1000
initial_Ny = 1000
xmin0_factor = -5.0
xmax0_factor = 5.0
ymin0_factor = -5.0
ymax0_factor = 5.0

# z interval
zmin = 0.0
zmax = 4.0

# Adaptive DeltaZ control
DZ_LOW_FACTOR = 0.01
DZ_HIGH_FACTOR = 0.10
DZ_SET_FACTOR = 0.05

# ============================================================
# Physical constants
# ============================================================
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

# Hermite-in-time effective integration factor used in the artificial plasma model.
plasma_time_factor = 0.5 * np.sqrt(np.pi) * tp / np.sqrt(2.0 * K)

# ============================================================
# Memory-efficient nonlinear/plasma update kernel
# ============================================================
update_kernel = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void update_xy_no_t(
    const cuDoubleComplex* E,
    double* I,
    double* rho,
    cuDoubleComplex* nonlinear,
    long long Ntot,
    int nonlinear_on,
    double beta,
    int K,
    double hbar,
    double omega,
    double rho_neutral,
    double plasma_time_factor,
    double wavenumber,
    double n2,
    double sigma,
    double tau
)
{
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ntot) return;

    cuDoubleComplex e = E[idx];
    double er = cuCreal(e);
    double ei = cuCimag(e);
    double Ii = er * er + ei * ei;
    I[idx] = Ii;

    double Ipow = 1.0;
    for (int j = 0; j < K - 1; ++j) Ipow *= Ii;  // I^(K-1)

    double IK = Ipow * Ii;
    double b = beta * IK / ((double)K * hbar * omega * rho_neutral);
    double rhoi = rho_neutral * (1.0 - exp(-b * plasma_time_factor));
    if (rhoi < 0.0) rhoi = 0.0;
    if (rhoi > rho_neutral) rhoi = rho_neutral;
    rho[idx] = rhoi;

    if (nonlinear_on == 0) {
        nonlinear[idx] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    // nonlinear is the coefficient N in E_z = N * E.
    double real_part = -0.5 * beta * Ipow - 0.5 * sigma * rhoi;
    double imag_part =  wavenumber * n2 * Ii - 0.5 * sigma * omega * tau * rhoi;
    nonlinear[idx] = make_cuDoubleComplex(real_part, imag_part);
}
''', "update_xy_no_t")

def hermite_phys_gpu(n, z):
    if n == 0:
        return cp.ones_like(z)
    if n == 1:
        return 2.0 * z

    H_nm2 = cp.ones_like(z)
    H_nm1 = 2.0 * z

    for k in range(1, n):
        H_n = 2.0 * z * H_nm1 - 2.0 * k * H_nm2
        H_nm2, H_nm1 = H_nm1, H_n

    return H_nm1

def HG(x, y, Pin=1.0, width=0.1, l=0, m=0):
    xi = cp.sqrt(2.0) * x[:, None] / width
    eta = cp.sqrt(2.0) * y[None, :] / width

    H_l = hermite_phys_gpu(l, xi)
    H_m = hermite_phys_gpu(m, eta)

    A = cp.sqrt(
        2.0 * Pin
        / (
            cp.pi * width**2
            * 2.0**(l + m)
            * math.factorial(l)
            * math.factorial(m)
        )
    )

    E = A * H_l * H_m * cp.exp(
        -(x[:, None]**2 + y[None, :]**2) / width**2
    )

    return cp.ascontiguousarray(E.astype(cp.complex128))

def update_physics_gpu(E, I=None, rho=None, nonlinear=None, nonlinear_on=True):
    """Compute I, rho, nonlinear coefficient, and adaptive length scales.

    I, rho, and nonlinear may be passed in to reuse already-allocated arrays.
    No I**(K-1) array is stored.
    """
    Nx, Ny = E.shape
    Ntot = Nx * Ny

    if I is None or I.shape != E.shape:
        I = cp.empty(E.shape, dtype=cp.float64)
    if rho is None or rho.shape != E.shape:
        rho = cp.empty(E.shape, dtype=cp.float64)
    if nonlinear is None or nonlinear.shape != E.shape:
        nonlinear = cp.empty(E.shape, dtype=cp.complex128)

    threads = 256
    blocks = (Ntot + threads - 1) // threads
    update_kernel(
        (blocks,),
        (threads,),
        (
            E,
            I,
            rho,
            nonlinear,
            np.int64(Ntot),
            np.int32(1 if nonlinear_on else 0),
            float(beta_k),
            np.int32(K),
            float(reduced_planck),
            float(omega),
            float(rho_neutral),
            float(plasma_time_factor),
            float(wavenumber),
            float(n2),
            float(sigma),
            float(tau),
        ),
    )

    Imax = float(cp.max(I).get())
    rho_max = float(cp.max(rho).get())

    if Imax > 0.0:
        l_Kerr = 1.0 / (wavenumber * n2 * Imax)
        Ipow_max = Imax ** (K - 1)
        l_MPA = 2.0 / (beta_k * Ipow_max) if Ipow_max > 0.0 else 1.0e300
    else:
        l_Kerr = 1.0e300
        l_MPA = 1.0e300

    if rho_max > 0.0:
        plasma_abs = sigma * np.sqrt(1.0 + omega * omega * tau * tau)
        l_plasma = 2.0 / (plasma_abs * rho_max)
    else:
        l_plasma = 1.0e300

    return I, rho, nonlinear, l_MPA, l_Kerr, l_plasma

# ============================================================
# Grids, FFT propagator, absorber, remeshing
# ============================================================
def make_grid(Nx, Ny, xmin, xmax, ymin, ymax):
    DeltaX = (xmax - xmin) / (Nx - 1)
    DeltaY = (ymax - ymin) / (Ny - 1)

    x_vector = cp.linspace(xmin, xmax, Nx)
    y_vector = cp.linspace(ymin, ymax, Ny)
    kx_vector = 2.0 * cp.pi * cp.fft.fftfreq(Nx, d=DeltaX)
    ky_vector = 2.0 * cp.pi * cp.fft.fftfreq(Ny, d=DeltaY)

    l_x = 4.0 * wavenumber * DeltaX**2
    l_y = 4.0 * wavenumber * DeltaY**2
    return DeltaX, DeltaY, x_vector, y_vector, kx_vector, ky_vector, l_x, l_y


def make_A_half(kx_vector, ky_vector, DeltaZ):
    # Half-step linear propagator for dE/dz = i/(2k) (d_xx + d_yy) E.
    kx2 = kx_vector[:, None] ** 2
    ky2 = ky_vector[None, :] ** 2
    return cp.exp(-0.25j / wavenumber * (kx2 + ky2) * DeltaZ)


def make_absorption_mask_gpu(x_vector, y_vector, xmin, xmax, ymin, ymax):
    x = x_vector[:, None]
    y = y_vector[None, :]

    x_width = xmax - xmin
    x_abs_width = absorb_frac_x * x_width
    sx_left = cp.maximum((xmin + x_abs_width - x) / x_abs_width, 0.0)
    sx_right = cp.maximum((x - (xmax - x_abs_width)) / x_abs_width, 0.0)
    mask_x = cp.exp(-absorb_strength_x * sx_left**absorb_power) * cp.exp(-absorb_strength_x * sx_right**absorb_power)

    y_width = ymax - ymin
    y_abs_width = absorb_frac_y * y_width
    sy_left = cp.maximum((ymin + y_abs_width - y) / y_abs_width, 0.0)
    sy_right = cp.maximum((y - (ymax - y_abs_width)) / y_abs_width, 0.0)
    mask_y = cp.exp(-absorb_strength_y * sy_left**absorb_power) * cp.exp(-absorb_strength_y * sy_right**absorb_power)

    return (mask_x * mask_y).astype(cp.float64)


def compute_widths_gpu(I, x_vector, y_vector, DeltaX, DeltaY):
    norm = cp.sum(I) * DeltaX * DeltaY
    if float(norm.get()) <= 0.0:
        return waist, waist

    # Use 1D vectors and broadcasting. No full coordinate matrices are stored.
    x = x_vector[:, None]
    y = y_vector[None, :]
    x_mean = cp.sum(I * x) * DeltaX * DeltaY / norm
    y_mean = cp.sum(I * y) * DeltaX * DeltaY / norm
    x2_mean = cp.sum(I * (x - x_mean) ** 2) * DeltaX * DeltaY / norm
    y2_mean = cp.sum(I * (y - y_mean) ** 2) * DeltaX * DeltaY / norm

    # For I ~ exp(-2 x^2/w^2), rms = w/2, so w_eff = 2*rms.
    w_x_eff = 2.0 * cp.sqrt(cp.maximum(x2_mean, 0.0))
    w_y_eff = 2.0 * cp.sqrt(cp.maximum(y2_mean, 0.0))
    return float(w_x_eff.get()), float(w_y_eff.get())


def remesh_E_fft(E, Nx_new, Ny_new, xmin, xmax, ymin, ymax):
    Nx_old, Ny_old = E.shape
    x_old = cp.linspace(xmin, xmax, Nx_old)
    y_old = cp.linspace(ymin, ymax, Ny_old)
    x_new = cp.linspace(xmin, xmax, Nx_new)
    y_new = cp.linspace(ymin, ymax, Ny_new)

    # Interpolate real/imaginary parts. This uses temporary arrays only during remesh.
    tmp_real = cp.empty((Nx_new, Ny_old), dtype=cp.float64)
    tmp_imag = cp.empty((Nx_new, Ny_old), dtype=cp.float64)
    Er = E.real
    Ei = E.imag
    for j in range(Ny_old):
        tmp_real[:, j] = cp.interp(x_new, x_old, Er[:, j])
        tmp_imag[:, j] = cp.interp(x_new, x_old, Ei[:, j])

    new_real = cp.empty((Nx_new, Ny_new), dtype=cp.float64)
    new_imag = cp.empty((Nx_new, Ny_new), dtype=cp.float64)
    for i in range(Nx_new):
        new_real[i, :] = cp.interp(y_new, y_old, tmp_real[i, :])
        new_imag[i, :] = cp.interp(y_new, y_old, tmp_imag[i, :])

    del tmp_real, tmp_imag
    return cp.ascontiguousarray(new_real + 1j * new_imag)


def maybe_remesh(E, I, Nx, Ny, xmin, xmax, ymin, ymax, DeltaX, DeltaY, x_vector, y_vector):
    if not REMESH_ON:
        return False, E, Nx, Ny

    w_x_eff, w_y_eff = compute_widths_gpu(I, x_vector, y_vector, DeltaX, DeltaY)
    wanted_DeltaX = w_x_eff / target_points_x
    wanted_DeltaY = w_y_eff / target_points_y

    need_x = DeltaX > wanted_DeltaX
    need_y = DeltaY > wanted_DeltaY
    if not need_x and not need_y:
        return False, E, Nx, Ny

    Nx_new = Nx
    Ny_new = Ny
    if need_x:
        Nx_target = int(np.ceil((xmax - xmin) / wanted_DeltaX)) + 1
        Nx_new = min(max(Nx_target, int(1.5 * Nx)), max_Nx)
    if need_y:
        Ny_target = int(np.ceil((ymax - ymin) / wanted_DeltaY)) + 1
        Ny_new = min(max(Ny_target, int(1.5 * Ny)), max_Ny)

    if Nx_new == Nx and Ny_new == Ny:
        return False, E, Nx, Ny

    print(
        f"Remeshing: Nx {Nx} -> {Nx_new}, Ny {Ny} -> {Ny_new}, "
        f"w_x_eff/DeltaX = {w_x_eff / DeltaX:.1f}, w_y_eff/DeltaY = {w_y_eff / DeltaY:.1f}",
        flush=True,
    )
    E_new = remesh_E_fft(E, Nx_new, Ny_new, xmin, xmax, ymin, ymax)
    return True, E_new, Nx_new, Ny_new


def sanitize_pin(pin_factor):
    whole = int(np.floor(pin_factor + 1e-12))
    tenth = int(round((pin_factor - whole) * 10))
    return f"{whole:03d}p{tenth}"


def range_name(a, b):
    return f"{a:.1f}_to_{b:.1f}"

# ============================================================
# One run
# ============================================================
def run_one_pin(pin_factor, outdir):
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    Pin = pin_factor * Pcr
    xmin = xmin0_factor * waist
    xmax = xmax0_factor * waist
    ymin = ymin0_factor * waist
    ymax = ymax0_factor * waist
    Nx = initial_Nx
    Ny = initial_Ny

    DeltaX, DeltaY, x_vector, y_vector, kx_vector, ky_vector, l_x, l_y = make_grid(Nx, Ny, xmin, xmax, ymin, ymax)
    absorb_mask = make_absorption_mask_gpu(x_vector, y_vector, xmin, xmax, ymin, ymax)

    # 2D Hermite transverse beam. With this convention, integral |E|^2 dx dy ~= Pin.
    # amp = cp.sqrt(2.0 * Pin / (cp.pi * waist**2)) # this is not used anymore 
    # E = amp * cp.exp(-(x_vector[:, None] ** 2 + y_vector[None, :] ** 2) / waist**2) # this is not used anymore
    # E = cp.ascontiguousarray(E.astype(cp.complex128)) # this is not used anymore

    E = HG(x_vector, y_vector, Pin=Pin, width=waist, l=2, m=1)

    I = rho = nonlinear = None
    I, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, I, rho, nonlinear, nonlinear_on=NONLINEAR_ON)
    l_ref = min(l_x, l_y, l_MPA, l_Kerr, l_plasma)
    DeltaZ = DZ_SET_FACTOR * l_ref
    A_half = make_A_half(kx_vector, ky_vector, DeltaZ)

    z = zmin
    step = 0
    last_l_ref_for_remesh = l_ref

    z_diag = [z]
    I_peak = [float(cp.max(I).get())]
    # I_center = [float(I[Nx // 2, Ny // 2].get())] # I do not want this
    rho_peak = [float(cp.max(rho).get())]
    dz_diag = [DeltaZ]
    Nx_diag = [Nx]
    Ny_diag = [Ny]

    # Full 2D intensity snapshots are now kept in CPU RAM and written only once
    # at the end. This removes repeated disk I/O and compression from the loop.
    I_snapshot_z = []
    I_snapshot_steps = []
    I_snapshot_Nx = []
    I_snapshot_Ny = []
    I_snapshot_x = []
    I_snapshot_y = []
    I_snapshots = []

    def store_I_snapshot(z_value, step_value):
        I_snapshot_z.append(z_value)
        I_snapshot_steps.append(step_value)
        I_snapshot_Nx.append(Nx)
        I_snapshot_Ny.append(Ny)
        I_snapshot_x.append(cp.asnumpy(x_vector))
        I_snapshot_y.append(cp.asnumpy(y_vector))
        I_snapshots.append(cp.asnumpy(I))

    tag = sanitize_pin(pin_factor)
    store_I_snapshot(z, step)

    print(f"Pin/Pcr = {pin_factor:.1f}; start; I0 = {I_peak[0]:.6e}", flush=True)

    while z < zmax - 1e-15:
        l_ref = min(l_x, l_y, l_MPA, l_Kerr, l_plasma)
        if not (DZ_LOW_FACTOR * l_ref <= DeltaZ <= DZ_HIGH_FACTOR * l_ref):
            DeltaZ = DZ_SET_FACTOR * l_ref
            if z + DeltaZ > zmax:
                DeltaZ = zmax - z
            A_half = make_A_half(kx_vector, ky_vector, DeltaZ)

        if z + DeltaZ > zmax:
            DeltaZ = zmax - z
            A_half = make_A_half(kx_vector, ky_vector, DeltaZ)

        # Strang: half linear, full nonlinear, half linear.
        E = cp.fft.ifft2(cp.fft.fft2(E) * A_half)

        I, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, I, rho, nonlinear, nonlinear_on=NONLINEAR_ON)
        if NONLINEAR_ON:
            E *= cp.exp(nonlinear * DeltaZ)

        E = cp.fft.ifft2(cp.fft.fft2(E) * A_half)

        if ABSORB_ON:
            E *= absorb_mask

        I, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, I, rho, nonlinear, nonlinear_on=NONLINEAR_ON)

        z += DeltaZ
        step += 1

        rapid_change = l_ref < 0.7 * last_l_ref_for_remesh
        scheduled_check = step % remesh_check_every == 0
        if scheduled_check or rapid_change:
            did_remesh, E, Nx, Ny = maybe_remesh(E, I, Nx, Ny, xmin, xmax, ymin, ymax, DeltaX, DeltaY, x_vector, y_vector)
            last_l_ref_for_remesh = l_ref
            if did_remesh:
                # Old arrays have wrong shape after remesh; release them before allocating new ones.
                del I, rho, nonlinear, A_half, absorb_mask, x_vector, y_vector, kx_vector, ky_vector
                cp.get_default_memory_pool().free_all_blocks()

                DeltaX, DeltaY, x_vector, y_vector, kx_vector, ky_vector, l_x, l_y = make_grid(Nx, Ny, xmin, xmax, ymin, ymax)
                absorb_mask = make_absorption_mask_gpu(x_vector, y_vector, xmin, xmax, ymin, ymax)
                I = rho = nonlinear = None
                I, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, I, rho, nonlinear, nonlinear_on=NONLINEAR_ON)
                DeltaZ = DZ_SET_FACTOR * min(l_x, l_y, l_MPA, l_Kerr, l_plasma)
                if z + DeltaZ > zmax:
                    DeltaZ = zmax - z
                A_half = make_A_half(kx_vector, ky_vector, DeltaZ)

        if step % number_diag == 0 or z >= zmax - 1e-15:
            Imax_now = float(cp.max(I).get())
            rhomax_now = float(cp.max(rho).get())
            z_diag.append(z)
            I_peak.append(Imax_now)
            # I_center.append(float(I[Nx // 2, Ny // 2].get())) # I do not want this
            rho_peak.append(rhomax_now)
            dz_diag.append(DeltaZ)
            Nx_diag.append(Nx)
            Ny_diag.append(Ny)
            if step % full_I_save_every == 0 or z >= zmax - 1e-15:
                store_I_snapshot(z, step)

            print(
                f"Pin/Pcr={pin_factor:.1f}, step={step}, z={z:.6e}, "
                f"DeltaZ={DeltaZ:.3e}, Nx={Nx}, Ny={Ny}, "
                f"I_peak/I0={I_peak[-1]/I_peak[0]:.3e}, "
                # f"I_center/I0={I_center[-1]/I_center[0]:.3e}, " # I do not want this
                f"rho_peak={rho_peak[-1]:.3e}, "
                f"l_ref={l_ref:.3e}, l_x={l_x:.3e}, l_y={l_y:.3e}, "
                f"l_MPA={l_MPA:.3e}, l_Kerr={l_Kerr:.3e}, l_plasma={l_plasma:.3e}",
                flush=True,
            )

    npz_path = os.path.join(outdir, f"Pin_{tag}_Pcr_Hermite_4D_FFT_diagnostics.npz")
    png_path = os.path.join(outdir, f"Pin_{tag}_Pcr_Hermite_4D_FFT_diagnostics.png")

    x_cpu = cp.asnumpy(x_vector)
    y_cpu = cp.asnumpy(y_vector)
    I_final = cp.asnumpy(I)
    rho_final = cp.asnumpy(rho)

    def as_object_array(items):
        arr = np.empty(len(items), dtype=object)
        arr[:] = items
        return arr

    # Use uncompressed saving for speed. Object arrays are used because remeshing
    # can change Nx, Ny, and therefore different snapshots may have different shapes.
    np.savez(
        npz_path,
        pin_factor=pin_factor,
        Pcr=Pcr,
        z_diag=np.array(z_diag), # this is a 1d array that contains the values of z at which a cheap diagnostic was made
        I_peak=np.array(I_peak),
        # I_center=np.array(I_center), # I do not want this
        rho_peak=np.array(rho_peak),
        DeltaZ_diag=np.array(dz_diag),
        Nx_diag=np.array(Nx_diag),
        Ny_diag=np.array(Ny_diag),
        I_snapshot_z=np.array(I_snapshot_z), # this is a 1d array that contains the values of z at which an expensive diagnostic was made
        I_snapshot_steps=np.array(I_snapshot_steps), # this is a 1d array that contains the loop interation index at which an expensive diagnostic was made
        I_snapshot_Nx=np.array(I_snapshot_Nx), # this is a 1d array that contains the number of cells along the x axis when an expensive diagnostic was made
        I_snapshot_Ny=np.array(I_snapshot_Ny), # this is a 1d array that contains the number of cells along the y axis when an expensive diagnostic was made
        I_snapshot_x=as_object_array(I_snapshot_x), # this is a 2d array that contains the x_vector 
        I_snapshot_y=as_object_array(I_snapshot_y), # this is a 2d array that contains the y_vector
        I_snapshots=as_object_array(I_snapshots), # this is a 3d array, important, it contains the beam profile I(x, y) when an expensive diagnostic was made,
        # to make the plot of I(x, y) vs. z, make a heatmap of I_snapshots at a given z
        full_I_save_every=full_I_save_every, # this is a 0d array, i.e, a number
        x=x_cpu, # not important
        y=y_cpu, # not important
        I_final=I_final, # not important
        rho_final=rho_final, # not important
    )

    plt.figure(figsize=(7, 4))
    plt.plot(np.array(z_diag), np.array(I_peak) / I_peak[0], "x-", label="global max I / I0")
    # plt.plot(np.array(z_diag), np.array(I_center) / I_center[0], "o-", label="I(x=0,y=0) / I0") # I do not want this
    plt.xlabel("z (m)")
    plt.ylabel("normalized intensity")
    plt.grid(True)
    plt.legend()
    plt.title(f"Hermite beam, Pin/Pcr = {pin_factor:.1f}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved {npz_path}", flush=True)
    print(f"Saved {png_path}", flush=True)

    del E, I, rho, nonlinear, x_vector, y_vector, kx_vector, ky_vector, A_half, absorb_mask
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=float, help="first Pin/Pcr value, e.g. 1.1")
    parser.add_argument("end", type=float, help="last Pin/Pcr value, e.g. 2.0")
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    sweep = range_name(args.start, args.end)
    outdir = "Hermite_4D_FFT"
    os.makedirs(outdir, exist_ok=True)

    nvals = int(round((args.end - args.start) / args.step)) + 1
    values = [round(args.start + i * args.step, 1) for i in range(nvals)]
    for val in values:
        run_one_pin(val, outdir)


if __name__ == "__main__":
    main()
