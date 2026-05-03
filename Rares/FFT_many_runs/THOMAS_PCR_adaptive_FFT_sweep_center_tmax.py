#!/usr/bin/env python3
"""
Adaptive Cartesian FFT split-step sweep with:
- adaptive DeltaZ based on l_x, l_GVD, l_MPA, l_Kerr, l_plasma
- THOMAS/PCR-style plasma update using neutral-density variable u
- absorbing boundaries in x and t
- RMS remeshing in x and t
- one .png and one .npz per Pin/Pcr value
- diagnostic I_center_tmax = max_t I(z, x=0, t)

Run example:
python THOMAS_PCR_adaptive_FFT_sweep.py 1.1 2.0
"""

import argparse
import os
import gc
import numpy as np
import cupy as cp  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# User switches
# ============================================================
NONLINEAR_ON = True
SAVE_PROFILES = False

number_diag = 100
profile_save_every = 5 * number_diag

# RMS remeshing controls
REMESH_ON = True
target_points_x = 50
target_points_t = 50
max_Nx = 2 * 4096
max_Nt = 2 * 4096
remesh_check_every = 50

# Absorbing boundary controls
ABSORB_ON = True
absorb_frac_x = 0.15
absorb_frac_t = 0.15
absorb_strength_x = 8.0
absorb_strength_t = 8.0
absorb_power = 4

# Initial grid
initial_Nx = 1000
initial_Nt = 1000
xmin0_factor = -5.0
xmax0_factor = 5.0
tmin0_factor = -5.0
tmax0_factor = 5.0

# z interval
zmin = 0.0
zmax = 4.0

# Adaptive DeltaZ control requested by user
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
ddot_k = 2.0 * (1e-15) ** 2 * (1e-2) ** (-1)
tau = 3.5e-13
Eg = 11.0 * elem_charge
sigma = wavenumber * elem_charge**2 * tau / (omega * electron_mass * eps0) / (1.0 + omega**2 * tau**2)
Pcr = lam**2 / (2.0 * np.pi * nb * n2)
beta_k = 6.5e-104
K = 7

pressure = 101_325.0
temperature = 25.0 + 273.15
rho_neutral = 2.0 * pressure / (kB * temperature)

# ============================================================
# THOMAS/PCR-style plasma + nonlinear update kernel
# ============================================================
update_kernel = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void update_physics_cartesian(
    const cuDoubleComplex* E,
    double* I,
    double* Ipow,
    double* rho,
    cuDoubleComplex* nonlinear,
    double* row_l_mpa,
    double* row_l_kerr,
    double* row_l_plasma,
    int Nx,
    int Nt,
    int nonlinear_on,
    double beta,
    int K,
    double hbar,
    double omega,
    double rho_neutral,
    double wavenumber,
    double n2,
    double sigma,
    double tau,
    double DeltaT,
    double nb,
    double Eg
)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= Nx) return;

    double min_l_mpa = 1.0e300;
    double min_l_kerr = 1.0e300;
    double min_l_plasma = 1.0e300;
    double plasma_abs = sigma * sqrt(1.0 + omega * omega * tau * tau);

    for (int it = 0; it < Nt; ++it) {
        int idx = ix * Nt + it;
        cuDoubleComplex e = E[idx];
        double er = cuCreal(e);
        double ei = cuCimag(e);
        double Ii = er * er + ei * ei;
        I[idx] = Ii;

        double p = 1.0;
        for (int j = 0; j < K - 1; ++j) p *= Ii;  // I^(K-1)
        Ipow[idx] = p;

        if (Ii > 0.0) {
            double lk = 1.0 / (wavenumber * n2 * Ii);
            if (lk < min_l_kerr) min_l_kerr = lk;
        }
        if (p > 0.0) {
            double lm = 2.0 / (beta * p);
            if (lm < min_l_mpa) min_l_mpa = lm;
        }
    }

    // THOMAS/PCR neutral-density formulation.
    // u starts as rho_neutral, rho = rho_neutral - u.
    double u = rho_neutral;
    rho[ix * Nt] = 0.0;

    double a_prefactor = sigma / (nb * nb * Eg);
    double b_prefactor = beta / ((double)K * hbar * omega * rho_neutral);

    for (int it = 0; it < Nt - 1; ++it) {
        int idx0 = ix * Nt + it;
        int idx1 = ix * Nt + it + 1;

        double I0 = I[idx0];
        double I1 = I[idx1];

        double a_left = a_prefactor * I0;
        double a_right = a_prefactor * I1;

        // Ipow is I^(K-1), so Ipow * I = I^K
        double b_left = b_prefactor * Ipow[idx0] * I0;
        double b_right = b_prefactor * Ipow[idx1] * I1;

        double a_mid = 0.5 * (a_left + a_right);
        double b_mid = 0.5 * (b_left + b_right);

        double gamma = b_mid - a_mid;
        double source = -a_mid * rho_neutral;

        double g = exp(-DeltaT * gamma);
        double h;
        if (fabs(gamma) > 1.0e-30) {
            h = source * (1.0 - g) / gamma;
        } else {
            h = source * DeltaT;
        }

        u = g * u + h;
        double rhoi = rho_neutral - u;
        if (rhoi < 0.0) rhoi = 0.0;
        if (rhoi > rho_neutral) rhoi = rho_neutral;
        rho[idx1] = rhoi;

        if (rhoi > 0.0) {
            double lp = 2.0 / (plasma_abs * rhoi);
            if (lp < min_l_plasma) min_l_plasma = lp;
        }
    }

    for (int it = 0; it < Nt; ++it) {
        int idx = ix * Nt + it;
        if (nonlinear_on == 0) {
            nonlinear[idx] = make_cuDoubleComplex(0.0, 0.0);
            continue;
        }

        double Ii = I[idx];
        double Ip = Ipow[idx];
        double rhoi = rho[idx];

        // N is the exponent in E_z = N * E
        double real_part = -0.5 * beta * Ip - 0.5 * sigma * rhoi;
        double imag_part =  wavenumber * n2 * Ii - 0.5 * sigma * omega * tau * rhoi;
        nonlinear[idx] = make_cuDoubleComplex(real_part, imag_part);
    }

    row_l_mpa[ix] = min_l_mpa;
    row_l_kerr[ix] = min_l_kerr;
    row_l_plasma[ix] = min_l_plasma;
}
''', "update_physics_cartesian")


def update_physics_gpu(E, DeltaT, nonlinear_on=True):
    Nx, Nt = E.shape
    I = cp.empty((Nx, Nt), dtype=cp.float64)
    I_pow = cp.empty((Nx, Nt), dtype=cp.float64)
    rho = cp.empty((Nx, Nt), dtype=cp.float64)
    nonlinear = cp.empty((Nx, Nt), dtype=cp.complex128)
    row_l_mpa = cp.empty(Nx, dtype=cp.float64)
    row_l_kerr = cp.empty(Nx, dtype=cp.float64)
    row_l_plasma = cp.empty(Nx, dtype=cp.float64)

    threads = 128
    blocks = (Nx + threads - 1) // threads
    update_kernel(
        (blocks,),
        (threads,),
        (
            E,
            I,
            I_pow,
            rho,
            nonlinear,
            row_l_mpa,
            row_l_kerr,
            row_l_plasma,
            np.int32(Nx),
            np.int32(Nt),
            np.int32(1 if nonlinear_on else 0),
            float(beta_k),
            np.int32(K),
            float(reduced_planck),
            float(omega),
            float(rho_neutral),
            float(wavenumber),
            float(n2),
            float(sigma),
            float(tau),
            float(DeltaT),
            float(nb),
            float(Eg),
        ),
    )

    lengths = cp.asnumpy(cp.array([cp.min(row_l_mpa), cp.min(row_l_kerr), cp.min(row_l_plasma)]))
    return I, I_pow, rho, nonlinear, float(lengths[0]), float(lengths[1]), float(lengths[2])

# ============================================================
# Grids, FFT propagator, absorber, remeshing
# ============================================================
def make_grid(Nx, Nt, xmin, xmax, tmin, tmax):
    DeltaX = (xmax - xmin) / (Nx - 1)
    DeltaT = (tmax - tmin) / (Nt - 1)
    x_vector = cp.linspace(xmin, xmax, Nx)
    t_vector = cp.linspace(tmin, tmax, Nt)
    x_matrix, t_matrix = cp.meshgrid(x_vector, t_vector, indexing="ij")
    kx_vector = 2.0 * cp.pi * cp.fft.fftfreq(Nx, d=DeltaX)
    kt_vector = 2.0 * cp.pi * cp.fft.fftfreq(Nt, d=DeltaT)
    kx_matrix, kt_matrix = cp.meshgrid(kx_vector, kt_vector, indexing="ij")
    l_x = 4.0 * wavenumber * DeltaX**2
    l_GVD = 4.0 * DeltaT**2 / abs(ddot_k) if abs(ddot_k) > 0 else 1.0e300
    return DeltaX, DeltaT, x_matrix, t_matrix, kx_matrix, kt_matrix, l_x, l_GVD


def make_A_half(kx_matrix, kt_matrix, DeltaZ):
    return cp.exp(-0.25j / wavenumber * kx_matrix**2 * DeltaZ + 0.25j * ddot_k * kt_matrix**2 * DeltaZ)


def make_absorption_mask_gpu(x_matrix, t_matrix, xmin, xmax, tmin, tmax):
    mask = cp.ones_like(x_matrix, dtype=cp.float64)

    x_width = xmax - xmin
    x_abs_width = absorb_frac_x * x_width
    sx_left = cp.maximum((xmin + x_abs_width - x_matrix) / x_abs_width, 0.0)
    sx_right = cp.maximum((x_matrix - (xmax - x_abs_width)) / x_abs_width, 0.0)
    mask_x_left = cp.exp(-absorb_strength_x * sx_left**absorb_power)
    mask_x_right = cp.exp(-absorb_strength_x * sx_right**absorb_power)

    t_width = tmax - tmin
    t_abs_width = absorb_frac_t * t_width
    st_left = cp.maximum((tmin + t_abs_width - t_matrix) / t_abs_width, 0.0)
    st_right = cp.maximum((t_matrix - (tmax - t_abs_width)) / t_abs_width, 0.0)
    mask_t_left = cp.exp(-absorb_strength_t * st_left**absorb_power)
    mask_t_right = cp.exp(-absorb_strength_t * st_right**absorb_power)

    return (mask * mask_x_left * mask_x_right * mask_t_left * mask_t_right).astype(cp.float64)


def compute_widths_gpu(I, x_matrix, t_matrix, DeltaX, DeltaT):
    norm = cp.sum(I) * DeltaX * DeltaT
    if float(norm.get()) <= 0.0:
        return waist, tp

    x_mean = cp.sum(I * x_matrix) * DeltaX * DeltaT / norm
    t_mean = cp.sum(I * t_matrix) * DeltaX * DeltaT / norm
    x2_mean = cp.sum(I * (x_matrix - x_mean) ** 2) * DeltaX * DeltaT / norm
    t2_mean = cp.sum(I * (t_matrix - t_mean) ** 2) * DeltaX * DeltaT / norm

    # For I ~ exp(-2 x^2/w^2), rms = w/2, so w_eff = 2*rms.
    w_eff = 2.0 * cp.sqrt(cp.maximum(x2_mean, 0.0))
    tp_eff = 2.0 * cp.sqrt(cp.maximum(t2_mean, 0.0))
    return float(w_eff.get()), float(tp_eff.get())


def remesh_E_fft(E, Nx_new, Nt_new, xmin, xmax, tmin, tmax):
    Nx_old, Nt_old = E.shape
    x_old = cp.linspace(xmin, xmax, Nx_old)
    t_old = cp.linspace(tmin, tmax, Nt_old)
    x_new = cp.linspace(xmin, xmax, Nx_new)
    t_new = cp.linspace(tmin, tmax, Nt_new)

    # Interpolate real and imaginary parts. First along x for all old t, then along t for all new x.
    tmp_real = cp.empty((Nx_new, Nt_old), dtype=cp.float64)
    tmp_imag = cp.empty((Nx_new, Nt_old), dtype=cp.float64)
    Er = E.real
    Ei = E.imag
    for j in range(Nt_old):
        tmp_real[:, j] = cp.interp(x_new, x_old, Er[:, j])
        tmp_imag[:, j] = cp.interp(x_new, x_old, Ei[:, j])

    new_real = cp.empty((Nx_new, Nt_new), dtype=cp.float64)
    new_imag = cp.empty((Nx_new, Nt_new), dtype=cp.float64)
    for i in range(Nx_new):
        new_real[i, :] = cp.interp(t_new, t_old, tmp_real[i, :])
        new_imag[i, :] = cp.interp(t_new, t_old, tmp_imag[i, :])

    return cp.ascontiguousarray(new_real + 1j * new_imag)


def maybe_remesh(E, I, Nx, Nt, xmin, xmax, tmin, tmax, DeltaX, DeltaT, x_matrix, t_matrix):
    if not REMESH_ON:
        return False, E, Nx, Nt

    w_eff, tp_eff = compute_widths_gpu(I, x_matrix, t_matrix, DeltaX, DeltaT)
    wanted_DeltaX = w_eff / target_points_x
    wanted_DeltaT = tp_eff / target_points_t

    need_x = DeltaX > wanted_DeltaX
    need_t = DeltaT > wanted_DeltaT
    if not need_x and not need_t:
        return False, E, Nx, Nt

    Nx_new = Nx
    Nt_new = Nt
    if need_x:
        Nx_target = int(np.ceil((xmax - xmin) / wanted_DeltaX)) + 1
        Nx_new = min(max(Nx_target, int(1.5 * Nx)), max_Nx)
    if need_t:
        Nt_target = int(np.ceil((tmax - tmin) / wanted_DeltaT)) + 1
        Nt_new = min(max(Nt_target, int(1.5 * Nt)), max_Nt)

    if Nx_new == Nx and Nt_new == Nt:
        return False, E, Nx, Nt

    print(
        f"Remeshing: Nx {Nx} -> {Nx_new}, Nt {Nt} -> {Nt_new}, "
        f"w_eff/DeltaX = {w_eff / DeltaX:.1f}, tp_eff/DeltaT = {tp_eff / DeltaT:.1f}",
        flush=True,
    )
    E_new = remesh_E_fft(E, Nx_new, Nt_new, xmin, xmax, tmin, tmax)
    return True, E_new, Nx_new, Nt_new


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
    tmin = tmin0_factor * tp
    tmax = tmax0_factor * tp
    Nx = initial_Nx
    Nt = initial_Nt

    DeltaX, DeltaT, x_matrix, t_matrix, kx_matrix, kt_matrix, l_x, l_GVD = make_grid(Nx, Nt, xmin, xmax, tmin, tmax)
    absorb_mask = make_absorption_mask_gpu(x_matrix, t_matrix, xmin, xmax, tmin, tmax)

    amp = cp.sqrt(2.0 * Pin / (cp.pi * waist**2))
    E = amp * cp.exp(-x_matrix**2 / waist**2 - t_matrix**2 / tp**2).astype(cp.complex128)
    E = cp.ascontiguousarray(E)

    I, I_pow, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, DeltaT, nonlinear_on=NONLINEAR_ON)
    l_ref = min(l_x, l_GVD, l_MPA, l_Kerr, l_plasma)
    DeltaZ = DZ_SET_FACTOR * l_ref
    A_half = make_A_half(kx_matrix, kt_matrix, DeltaZ)

    z = zmin
    step = 0
    last_l_ref_for_remesh = l_ref

    z_diag = [z]
    I_peak = [float(cp.max(I).get())]
    I_center_tmax = [cp.max(I[Nx // 2, :]).item()]
    dz_diag = [DeltaZ]
    Nx_diag = [Nx]
    Nt_diag = [Nt]

    print(f"Pin/Pcr = {pin_factor:.1f}; start; I0 = {I_peak[0]:.6e}", flush=True)

    while z < zmax - 1e-15:
        l_ref = min(l_x, l_GVD, l_MPA, l_Kerr, l_plasma)
        if not (DZ_LOW_FACTOR * l_ref <= DeltaZ <= DZ_HIGH_FACTOR * l_ref):
            DeltaZ = DZ_SET_FACTOR * l_ref
            if z + DeltaZ > zmax:
                DeltaZ = zmax - z
            A_half = make_A_half(kx_matrix, kt_matrix, DeltaZ)

        if z + DeltaZ > zmax:
            DeltaZ = zmax - z
            A_half = make_A_half(kx_matrix, kt_matrix, DeltaZ)

        # Strang: half linear, full nonlinear, half linear
        E = cp.fft.ifft2(cp.fft.fft2(E) * A_half)

        I, I_pow, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, DeltaT, nonlinear_on=NONLINEAR_ON)
        if NONLINEAR_ON:
            E = E * cp.exp(nonlinear * DeltaZ)

        E = cp.fft.ifft2(cp.fft.fft2(E) * A_half)

        if ABSORB_ON:
            E *= absorb_mask

        I, I_pow, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, DeltaT, nonlinear_on=NONLINEAR_ON)

        z += DeltaZ
        step += 1

        rapid_change = l_ref < 0.7 * last_l_ref_for_remesh
        scheduled_check = step % remesh_check_every == 0
        if scheduled_check or rapid_change:
            did_remesh, E, Nx, Nt = maybe_remesh(E, I, Nx, Nt, xmin, xmax, tmin, tmax, DeltaX, DeltaT, x_matrix, t_matrix)
            last_l_ref_for_remesh = l_ref
            if did_remesh:
                DeltaX, DeltaT, x_matrix, t_matrix, kx_matrix, kt_matrix, l_x, l_GVD = make_grid(Nx, Nt, xmin, xmax, tmin, tmax)
                absorb_mask = make_absorption_mask_gpu(x_matrix, t_matrix, xmin, xmax, tmin, tmax)
                I, I_pow, rho, nonlinear, l_MPA, l_Kerr, l_plasma = update_physics_gpu(E, DeltaT, nonlinear_on=NONLINEAR_ON)
                DeltaZ = DZ_SET_FACTOR * min(l_x, l_GVD, l_MPA, l_Kerr, l_plasma)
                if z + DeltaZ > zmax:
                    DeltaZ = zmax - z
                A_half = make_A_half(kx_matrix, kt_matrix, DeltaZ)

        if step % number_diag == 0 or z >= zmax - 1e-15:
            z_diag.append(z)
            # I_peak.append(float(cp.max(I).get())) # I do not want to save this yet
            I_center_tmax.append(cp.max(I[Nx // 2, :]).item())
            dz_diag.append(DeltaZ)
            Nx_diag.append(Nx)
            Nt_diag.append(Nt)
            print(
                f"Pin/Pcr={pin_factor:.1f}, step={step}, z={z:.6e}, "
                f"DeltaZ={DeltaZ:.3e}, Nx={Nx}, Nt={Nt}, "
                f"I/I0={I_center_tmax[-1]/I_center_tmax[0]:.3e}, "
                f"l_ref={l_ref:.3e}, l_x={l_x:.3e}, l_GVD={l_GVD:.3e}, "
                f"l_MPA={l_MPA:.3e}, l_Kerr={l_Kerr:.3e}, l_plasma={l_plasma:.3e}",
                flush=True,
            )

    tag = sanitize_pin(pin_factor)
    npz_path = os.path.join(outdir, f"Pin_{tag}_Pcr.npz")
    png_path = os.path.join(outdir, f"Pin_{tag}_Pcr.png")

    x_cpu = cp.asnumpy(x_matrix[:, 0])
    t_cpu = cp.asnumpy(t_matrix[0, :])
    I_final = cp.asnumpy(I)

    np.savez_compressed(
        npz_path,
        pin_factor=pin_factor,
        Pcr=Pcr,
        z_diag=np.array(z_diag),
        I_peak=np.array(I_peak),
        I_center_tmax=np.array(I_center_tmax),
        DeltaZ_diag=np.array(dz_diag),
        Nx_diag=np.array(Nx_diag),
        Nt_diag=np.array(Nt_diag),
        x=x_cpu,
        t=t_cpu,
        I_final=I_final,
    )

    plt.figure(figsize=(7, 4))
    # plt.plot(np.array(z_diag), np.array(I_peak) / I_peak[0], "x-", label="global max I") # I do not need this now
    plt.plot(np.array(z_diag), np.array(I_center_tmax) / I_center_tmax[0], "o-", label="max_t I(x=0,t) / I0")
    plt.xlabel("z (m)")
    plt.ylabel("normalized intensity")
    plt.grid(True)
    plt.legend()
    plt.title(f"Pin/Pcr = {pin_factor:.1f}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved {npz_path}", flush=True)
    print(f"Saved {png_path}", flush=True)

    del E, I, I_pow, rho, nonlinear, x_matrix, t_matrix, kx_matrix, kt_matrix, absorb_mask
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=float, help="first Pin/Pcr value, e.g. 1.1")
    parser.add_argument("end", type=float, help="last Pin/Pcr value, e.g. 2.0")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    sweep = range_name(args.start, args.end)
    outdir = f"pin_sweep_{sweep}"
    os.makedirs(outdir, exist_ok=True)

    nvals = int(round((args.end - args.start) / args.step)) + 1
    values = [round(args.start + i * args.step, 1) for i in range(nvals)]
    for val in values:
        run_one_pin(val, outdir)


if __name__ == "__main__":
    main()
