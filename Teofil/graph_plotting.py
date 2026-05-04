# ============ This code takes some diagnostics from the simulation and ============
# ============ creates different graphs for visualizing the effects     ============
import numpy as np
import matplotlib.pyplot as plt
from beam_profiles import *
from constants import *

'''
Problems to address:
    - variable mesh sizes
    - how to import data form the file
    - file managing system, with references to all the other files
'''

########## For simulations of the form x, y, z, with artificial time
# read from a file
def read_npz(filename):
    '''
    read the arrays, we need a list with z coordinates and a list of 2D arrays (x, y) with the intensity
    '''
    import_data = np.load(filename)
    # arr = import_data['array1'] # needs some details here
    # data = xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Emesh
    # return (data) # returns the parameters and the 3D np.array in z, y, and x of the field envelope



########## Plots for xyz simulations, artificial time
# plot the max intensity (on-axis) over time
def on_axis_max_I_vs_z(data: np.ndarray) -> None:
    xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Emesh = data
    z = np.linspace(zmin, zmax, Nz)
    midx = Emesh.shape[2] // 2
    midy = Emesh.shape[1] // 2
    E2 = Emesh[:, midy, midx] ** 2
    # E2n /= Emesh[0, midy, midx] ** 2 # in case you need normalized values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, E2, color='green')
    # ax.plot(z, E2n, color='blue')

    ax.set_title('Max intensity over time, on $z$ axis, for artificial time')
    ax.set_xlabel('$z$ (m)')
    ax.set_ylabel('$\\varepsilon_{max}^2$ ($V^2/m^2$)')
    # ax.set_ylabel('$\\varepsilon_{max}^2 / \\varepsilon_0^2$ (1)') # normalized

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('xyz_sim_I_max_vs_z.pdf', dpi=150)
    plt.show()

# plot field in time (in the reference frame of v_g)
def field_in_time(data: np.ndarray, tmin: float, tmax: float, Nt: int, z: float) -> None:
    xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Emesh = data
    midx = Emesh.shape[2] // 2
    midy = Emesh.shape[1] // 2
    z_idx = np.argmin(np.abs(np.linspace(zmin, zmax, Nz) - z))
    E_ref = Emesh[z_idx, midy, midx]
    
    t = np.linspace(tmin, tmax, Nt)
    En = np.exp(- t**2 / tp**2)
    E = E_ref * En

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, E, color='red')
    # ax.plot(z, En, color='magenta')

    ax.set_title(f'$\\varepsilon(x=0, y=0, z={z}, t)$ vs artificial time')
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$\\varepsilon$ ($V/m$)')
    # ax.set_ylabel('$\\varepsilon / \\varepsilon_0$ (1)') # normalized

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('xyz_sim_E_vs_t_artificial.pdf', dpi=150)
    plt.show()




# plot y-z beam profile
def beam_profile_yz(data: np.ndarray) -> None:
    xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Emesh = data
    E = np.array(Emesh[:, :, Nx // 2]).T

    # plot figure
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.imshow(E, cmap='inferno', aspect='auto',
              extent=[zmin, zmax, ymin, ymax],
              origin='lower')
    ax.set_title('Intensity distribution in y and z for artificial time')
    fig.colorbar(c, ax=ax, label='$\\varepsilon$ (V/m)')
    ax.set_xlabel('$z$ (m)')
    ax.set_ylabel('$y$ (m)')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('xyz_sim_beam_profile_yz.pdf', dpi=150)
    plt.show()



########## Test data
def data1():  # easy test
    xmin, xmax = -0.1, 0.1
    Nx = 50
    ymin, ymax = -0.1, 0.1
    Ny = 50
    zmin, zmax = 0.0, 10.0
    Nz = 100

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)
    X, Y = np.meshgrid(x, y)
    Emesh = np.zeros((Nz, Ny, Nx))
    for i in range(Nz):
        Emesh[i] = supergaussian(np.sqrt(X ** 2 + Y ** 2), Pin=gaussian(r=z[i] - zmax / 3), width=0.05, p=5)
    return (xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Emesh)

def data2(): # heavier test
    xmin, xmax = -0.1, 0.1
    Nx = 500
    ymin, ymax = -0.1, 0.1
    Ny = 500
    zmin, zmax = 0.0, 10.0
    Nz = 1000

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)
    X, Y = np.meshgrid(x, y)
    Imesh = np.zeros((Nz, Ny, Nx))

    for i in range(Nz):
        Imesh[i] = gaussian(np.sqrt(X ** 2 + Y ** 2), Pin=gaussian(r=z[i] - zmax / 3), width=0.05)
    return (xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Imesh)

args = data1()
# on_axis_max_I_vs_z(args)
# beam_profile_yz(args)
# field_in_time(args, tmin=-1e-12, tmax=1e-12, Nt=1000, z=3.4)