import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# %% physical constants of the simulation
epsilon0 = 8.85e-12
c = 3e8
lam0 = 775e-9
k0 = 2 * np.pi / lam0
w0 = 0.7e-3
tp = 85e-15
Pcr = 1.7e9
Pin = Pcr * 1

# %% grid
NR = 1000 # here NR + 2 represents the number of points in the radial direction 
NZ = int(1e4) # here NR + 1 represents the number of points in the Z direction (the direction of the propagation of the beam)
nu = 1  # cylindrical geometry (if this confuses you, check Couairon_Kolesik_EPJST2011_199_5)

rmin = 0.0 # this is the origin for the radial axis
rmax = 5 * w0 # this is the maximum radial value 
DeltaR = (rmax - rmin) / (NR + 1) # this is the spacing between two consecutive points
r = np.linspace(rmin, rmax, NR + 2) # this is the radial vector 

zmin = 0.0 # same as above, but for the Z direction not for the radial direction
zmax = 10.0
DeltaZ = (zmax - zmin) / NZ
z = np.linspace(zmin, zmax, NZ + 1)

# %% coefficients
delta = DeltaZ / (4 * k0 * DeltaR**2) # this is a coefficient relevant for the construction of the matrices, see Couairon_Kolesik_EPJST2011_199_5

idx = np.arange(1, NR + 1, dtype=float)
u = np.ones(NR) - 0.5 * nu / idx
v = np.ones(NR) + 0.5 * nu / idx

# %% tridiagonal representation of Lplus and Lminus
# Matrix size is (NR+2) x (NR+2)
# If you have skipped to this point and have not read Couairon_Kolesik_EPJST2011_199_5, please do so, otherwise you will not understant anything

# Lplus diagonals
diag_plus = np.zeros(NR + 2, dtype=complex)
upper_plus = np.zeros(NR + 1, dtype=complex)  # A[i, i+1]
lower_plus = np.zeros(NR + 1, dtype=complex)  # A[i+1, i]

diag_plus[0] = 1 - 4j * delta
diag_plus[1:NR + 1] = 1 - 2j * delta
diag_plus[NR + 1] = 0  # keep MATLAB boundary condition

upper_plus[0] = 4j * delta
upper_plus[1:NR + 1] = 1j * delta * v

lower_plus[0:NR] = 1j * delta * u

# Lminus diagonals
diag_minus = np.zeros(NR + 2, dtype=complex)
upper_minus = np.zeros(NR + 1, dtype=complex)
lower_minus = np.zeros(NR + 1, dtype=complex)

diag_minus[0] = 1 + 4j * delta
diag_minus[1:NR + 1] = 1 + 2j * delta
diag_minus[NR + 1] = 1  # keep MATLAB boundary condition

upper_minus[0] = -4j * delta
upper_minus[1:NR + 1] = -1j * delta * v

lower_minus[0:NR] = -1j * delta * u

# scipy banded format:
# ab[0,1:]  = upper diagonal
# ab[1,:]   = main diagonal
# ab[2,:-1] = lower diagonal
ab_minus = np.zeros((3, NR + 2), dtype=complex)
ab_minus[0, 1:] = upper_minus
ab_minus[1, :] = diag_minus
ab_minus[2, :-1] = lower_minus

# %% field propagation
# here we compute the actual field
E = np.zeros((NR + 2, NZ + 1), dtype=complex)

E0 = np.sqrt(2 * Pin / (np.pi * w0**2))
E[:, 0] = E0 * np.exp(-r**2 / w0**2)

for n in range(NZ):
    rhs = diag_plus * E[:, n]
    rhs[:-1] += upper_plus * E[1:, n]
    rhs[1:] += lower_plus * E[:-1, n]

    E[:, n + 1] = solve_banded((1, 1), ab_minus, rhs)

# %% intensity (self explanatory, compute it from the electric field)
I = 0.5 * epsilon0 * c * np.abs(E)**2

fig1 = plt.figure()
plt.imshow(
    I,
    extent=[z.min(), z.max(), r.min(), r.max()],
    aspect='auto',
    origin='lower'
)
plt.xlabel('z coordinate (m)')
plt.ylabel('r coordinate (m)')
plt.colorbar(label='Intensity (W/m^2)')
plt.title('Beam intensity at t = 0')

# %% exact solution
n_medium = 1
zR = np.pi * w0**2 * n_medium / lam0

Zmesh, Rmesh = np.meshgrid(z, r)
wz = w0 * np.sqrt(1 + (Zmesh / zR)**2)
Eexact = E0 * (w0 / wz) * np.exp(-Rmesh**2 / wz**2)
Iexact = 0.5 * epsilon0 * c * np.abs(Eexact)**2

# %% relative error
relative_error = 100 * np.abs(I - Iexact) / Iexact

fig2 = plt.figure()
plt.imshow(
    relative_error,
    extent=[z.min(), z.max(), r.min(), r.max()],
    aspect='auto',
    origin='lower'
)
plt.xlabel('z coordinate (m)')
plt.ylabel('r coordinate (m)')
plt.colorbar(label='Relative error (%)')
plt.title('Relative error from the exact solution at t = 0')
plt.show()