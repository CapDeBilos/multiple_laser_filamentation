# Multiphoton Absorption (MPA) Term — Implementation Notes


## What was added

The file `added_MPA.py` extends the existing GNLSE solver (diffraction + GVD + Kerr) by including the **multiphoton absorption / ionization term** from Mlejnek, Wright & Moloney, *Opt. Lett.* **23**, 382 (1998).

---

## Physics

### The propagation equation

In the co-moving frame the GNLSE reads (Mlejnek OL 1998, eq. 1):

```
dE/dz = (i / 2k₀) ∇⊥² E
        - (i k''₀ / 2) ∂²E/∂t²
        + i k₀ n₂ |E|² E
        - (β_K / 2) |E|^(2K−2) E
```

The first two lines were already implemented. The new term is the last line.

### Where the MPA term comes from

Each ionization event in the multiphoton regime requires absorbing K photons simultaneously. The probability per unit time per unit volume of such an event scales as the intensity to the K-th power: W ∝ |E|^(2K). The power extracted from the field per unit volume is W × Kℏω₀ × ρ_neutral. Working through the slowly-varying envelope reduction (see Couairon et al., EPJST **199**, 5 (2011), section 3.4, eq. 107–110) one obtains an effective current acting on the envelope:

```
J_env / (ε₀ c) = n₀ β_K |E|^(2K−2) E
```

Inserting this into the NLS equation gives the −(β_K/2)|E|^(2K-2) E term. The factor of 1/2 comes from writing the full field as E_real = Re[E e^{i(k₀z − ω₀t)}] and time-averaging. The term is **purely real** (no imaginary unit in front), so it is dissipative — it removes energy from the field without adding phase.

### Parameter values (air at STP, λ = 775 nm)

| Symbol | Value | Units | Source |
|--------|-------|-------|--------|
| K | 7 | — | 11 eV gap, 1.6 eV/photon → 7 photons |
| β^(7) | 6.5 × 10⁻¹⁰⁴ | m¹¹ W⁻⁶ | Mlejnek OL 1998 (from Keldysh theory) |

Units of β_K are m^(2K−3) W^(1−K), which for K=7 gives m¹¹ W⁻⁶. This ensures β_K|E|^(2K−2) has units of m⁻¹ (consistent with a z-derivative of E).

---

## Numerical implementation

### Where it sits in the split-step scheme

The operator splitting already used for Kerr is:

```
E^{n+1} = [L_−^{delta}]^{−1} · ([L_+^{delta}] E^n + ΔzN(E^n)) · [L_+^d] · [L_−^d]^{−T}
```

where N(E) is the pointwise nonlinear term. MPA is simply added into N:

```python
def nonlinear_term(E):
    I = np.abs(E)**2
    kerr_term = 1j * k0 * n2 * I * E
    mpa_term  = -(beta_K / 2.0) * I**(K_mpa - 1) * E
    return kerr_term + mpa_term
```

This is correct because:
1. MPA is local (no spatial coupling between grid points), so it can be handled explicitly without touching the tridiagonal solvers.
2. The Crank-Nicolson matrices for diffraction and GVD are **unchanged** — MPA only modifies the right-hand side.

### Time-stepping of the nonlinear term

Same Adams-Bashforth 2nd-order scheme already used for Kerr:

- Step 0: forward Euler, `NL = ΔzN(E_0)`
- Step k > 0: `NL = Δz(3/2 N(E_k) − 1/2 N(E_{k−1}))`

### On the magnitude of β_K

β^(7) = 6.5 × 10⁻¹⁰⁴ m¹¹ W⁻⁶ looks absurdly small. This is expected: it is compensated by |E|^(2K−2) = |E|^12, which at filament intensities (~10¹³ W/cm² = 10¹⁷ W/m²) gives |E|^12 ~ (2I/ε₀c)^6 ~ 10¹¹⁰ m⁻¹² V¹² → the product β_K|E|^12 ~ 10⁶ m⁻¹, which is a strong attenuation rate, as expected physically.

---

## What to expect in the simulation

Without MPA the Kerr self-focusing drives a catastrophic intensity collapse at z ≈ z_focus. With MPA the collapse is **arrested**: once the intensity is high enough, the MPA term drains energy rapidly, preventing the singularity. The on-axis peak intensity plot (Fig. 2 in the output) should show a plateau or oscillation at high intensity rather than a divergence — the signature of filamentation.

---

## Connection to the current / plasma term

The plasma current term in Mlejnek eq. 1 is `−(σ/2)(1 + iωτ)ρE`. This requires solving the Drude rate equation for ρ alongside the field. MPA feeds into that equation as a source: `∂ρ/∂t = ... + β_K|E|^(2K) / (Kℏω)`. The two terms are therefore **coupled but separable** in the split-step scheme: MPA acts on E directly, while the plasma current acts on E through ρ which is updated from a separate ODE. Both can be added to N(E) once ρ is available.

---

## Files in the project

| File | Contents |
|------|----------|
| `transverse_1.py` | Baseline: diffraction only (1D radial, no time) |
| `added_GVD.py` | + GVD, validated against analytic solution |
| `added_GVD_Kerr.py` | + Kerr nonlinearity |
| `added_MPA.py` | **+ MPA / multiphoton ionization** ← this file |
