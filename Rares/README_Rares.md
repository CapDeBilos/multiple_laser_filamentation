# multiple_laser_filamentation
Electrodynamics project: Simulations of multiple laser filamentation for the PHY_2S004_EP electodynamics course.

Description of what was done to transverse_1.py
Objective

We aim to simulate solutions to Maxwell’s equations relevant to laser filamentation in plasmas (starting with air).

Due to computational constraints (time and memory), we solve for the envelope of the electric field, which varies slowly in space and time.

Field Representation

The electric field is written as:

$$
E(r,z,t) = \varepsilon(r,z,t), e^{i(k_0 z - \omega_0 t)} \\\\ (1)
$$

where:

$E(r,z,t)$ is the electric field
$\varepsilon(r,z,t)$ is the slowly varying envelope
$e^{i(k_0 z - \omega_0 t)}$ is the fast oscillating carrier

This assumes:

The pulse is centered around frequency $\omega_0$ and wavenumber $k_0$
The polarization is transverse (perpendicular to the $Oz$ axis)
The polarization remains fixed
Change of Variables

Introduce the moving frame:

$$
t = \tau + \frac{\zeta}{v_g}, \qquad z = \zeta
$$

Define:

$$
\bar{k} = k_0 - \frac{\omega_0}{v_g}
$$

Using approximations:

$\bar{k} \approx 0$
$\bar{k}^2 \approx 0$
$v_g \approx c$
$\partial_\tau \approx 0$

we obtain:

\frac{i}{2k_0} \nabla_\perp \varepsilon
\tag{2}
$$

Paraxial Equation

Equation (2) is the paraxial wave equation for the envelope:

$$
\varepsilon = \varepsilon(r,\zeta,\tau)
$$

It is computationally much more tractable than the full Maxwell system.

Transverse Laplacian

In cylindrical or planar symmetry:

$$
\nabla_\perp =
\partial_X^2 + \frac{d-1}{X},\partial_X
$$

where:

$X = r$ (cylindrical) or $X = x$ (planar)
$d$ is the number of transverse dimensions
Boundary Conditions
Cylindrical geometry:

$$
\partial_r \varepsilon(r=0) = 0, \qquad
\varepsilon(r=r_{\max}, \zeta) = 0
$$

Planar geometry:

$$
\partial_x \varepsilon(x=0) = 0, \qquad
\varepsilon(x=x_{\max}, \zeta) = 0
$$

Initial Condition

The initial condition is the beam profile:

$$
\varepsilon(r, 0, \tau)
$$

Simulation Log
🔹 Explicit Scheme

Parameters:

$\lambda_0 = 775\ \text{nm}$
$w_0 = 0.7\ \text{mm}$
Rayleigh length: $z_0 \approx 2\ \text{m}$
$t_p = 85\ \text{fs}$
$P_{\text{cr}} = 1.7\ \text{GW}$
$P_{\text{in}}$ in units of $P_{\text{cr}}$

Simulation domain:

$z_{\max} = 1\ \text{m}$
$r_{\max} = 10,w_0$

Observation:

Good agreement with analytic solution for very small propagation distances
Becomes unstable quickly
Extremely large numerical error:

$$
\text{max relative error} \sim 10^{38}%
$$

Conclusion:

❌ Explicit scheme is unusable for this problem
(even for pure transverse diffraction)

🔹 Crank–Nicolson Scheme

Switch to implicit method:

$$
L_- \varepsilon^{n+1} = L_+ \varepsilon^n
$$

Procedure:

Define $\delta$
Construct matrices $L_+$ and $L_-$
Compute:
$$
L = L_-^{-1} L_+
$$
Iterate over propagation steps

Observation:

Much more stable than explicit scheme
Can propagate over significantly longer distances
Still limited by computational cost

Best result obtained:

$$
\text{max relative error} \approx 160%
$$

before hitting:

memory limits
computation time limits
Next Steps
Improve computational efficiency
Optimize solver (tridiagonal / sparse methods)
Increase accessible propagation distance
TODO

🚀 Obtain more computational power
