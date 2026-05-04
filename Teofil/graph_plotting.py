# ============ This code takes some diagnostics from the simulation and ============
# ============ creates different graphs for visualizing the effects     ============
import numpy as np
import matplotlib.pyplot as plt
import os
from beam_profiles import *
from constants import *

'''
Problems to address:
    - variable mesh sizes
    - how to import/read data from the file
    - file managing system, with references to all the other files
    - if you want to add multiple plots on the same graph, you need to pass ax=None as an
    argument then address this with an if statement in the function
'''

########## For simulations in x, y, z, with artificial time
class BeamSimulation:
    def __init__(self, filename: str):
        self.load(filename)

    def load(self, filename: str):
        data = np.load(filename, allow_pickle=True)
        self.pin_factor         = data['pin_factor']
        self.Pcr                = data['Pcr']
        self.z_diag             = data['z_diag']          # 1d array, z values at cheap diagnostic
        self.I_peak             = data['I_peak']
        self.I_center           = data['I_center']
        self.rho_peak           = data['rho_peak']
        self.DeltaZ_diag        = data['DeltaZ_diag']
        self.Nx_diag            = data['Nx_diag']
        self.Ny_diag            = data['Ny_diag']
        self.snap_z             = data['I_snapshot_z']     # 1d array, z values at expensive diagnostic
        self.snap_steps         = data['I_snapshot_steps'] # 1d array, loop iteration indices
        self.snap_Nx            = data['I_snapshot_Nx']    # 1d array, Nx at expensive diagnostic
        self.snap_Ny            = data['I_snapshot_Ny']    # 1d array, Ny at expensive diagnostic
        self.snap_x             = data['I_snapshot_x']     # 2d array, x vectors
        self.snap_y             = data['I_snapshot_y']     # 2d array, y vectors
        self.snaps              = data['I_snapshots']      # 3d array, beam profile I(x,y) at each snap, shape (Nz, Nx, Ny)
        self.full_I_save_every  = data['full_I_save_every']
        self.x_cpu              = data['x']
        self.y_cpu              = data['y']
        self.I_final            = data['I_final']
        self.rho_final          = data['rho_final']
        self.I_peak_n       = self.I_peak / self.I_peak[0]      # normalize quantities
        self.I_center_n     = self.I_center / self.I_peak[0]
        self.snaps_n        = self.snaps / self.I_peak[0]
        self.save_dir       = os.path.dirname(os.path.abspath(filename))        # other attributes
        self.file_name      = os.path.splitext(os.path.basename(filename))[0]

    def on_axis_max_vs_z(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.z_diag, self.I_peak_n, color='green', label='$I_{max} / I_0$')
        ax.plot(self.z_diag, self.I_center_n, color='blue', label='$I_{r=0} / I_0$')

        ax.set_title('$I_{max}(x=0, y=0, z)/I_0$, artificial time')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|\\varepsilon_{max} / \\varepsilon_0|^2$ (1)')
        ax.legend()

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{self.file_name}_I_max_z_t_artif.pdf', dpi=150)
        # plt.show()
    
    def beam_profile_x(self, z: float):
        z_idx = np.clip(np.searchsorted(self.snap_z, z), 0, len(self.snap_z) - 1)
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        x = self.snap_x[z_idx]
        snap = self.snaps_n[z_idx]
        I_n = snap[:, snap.shape[1] // 2]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, I_n, color='red', label='$I_{max} / I_0$')
        # ax.set_ylim(0, 10)
        ax.set_title(f'Intensity profile $I(x, y=0, z = {z_val:.3f})/I_0$, artificial time')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$|\\varepsilon / \\varepsilon_0|^2$ (1)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{self.file_name}_profile_x_t_artif_z_{z_str}.pdf', dpi=150)
        # plt.show()

    def beam_profile_xz(self):
        I_n = np.array([snap[:, snap.shape[1] // 2] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        c = ax.imshow(I_n, cmap='inferno', aspect='auto', # vmin=0, vmax=5,
                extent=[self.snap_z[0], self.snap_z[-1], self.snap_x[0][0], self.snap_x[0][-1]],
                origin='lower')
        ax.set_title('Intensity profile $I(x, y=0, z)/I_0$, artificial time')
        fig.colorbar(c, ax=ax, label='$|\\varepsilon / \\varepsilon_0|^2$ (1)')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$x$ (m)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{self.file_name}_profile_xz_t_artif.pdf', dpi=150)
        # plt.show()
    
    def beam_profile_xy(self, z: float):
        z_idx = np.clip(np.searchsorted(self.snap_z, z), 0, len(self.snap_z) - 1)
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        I_n = self.snaps_n[z_idx].T # because imshow treats array as (rows, cols) = (y, x)

        fig, ax = plt.subplots(figsize=(6, 6))
        c = ax.imshow(I_n, cmap='hot', aspect='equal', # vmin=0, vmax=5,
                extent=[self.snap_x[0][0], self.snap_x[0][-1], self.snap_y[0][0], self.snap_y[0][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x, y, z = {z_val:.3f})/I_0$, artificial time')
        fig.colorbar(c, ax=ax, label='$|\\varepsilon / \\varepsilon_0|^2$ (1)', fraction=0.046, pad=0.04)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{self.file_name}_profile_xy_t_artif_z_{z_str}.pdf', dpi=150)
        # plt.show()
        



########## Some tests
sim_test = BeamSimulation('/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Teofil/test_runs/Pin_013p0_Pcr_gaussian_4D_FFT_diagnostics.npz')
# sim_test.on_axis_max_vs_z()
# sim_test.beam_profile_xz()
# sim_test.beam_profile_xy(0.5)
# for z in sim_test.snap_z:
    # sim_test.beam_profile_xy(z)
# sim_test.beam_profile_x(0.5)
# for z in sim_test.snap_z:
    # sim_test.beam_profile_x(z)



'''
########## Old test data
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

# args = data1()
# on_axis_max_I_vs_z(args)
# beam_profile_yz(args)
# field_in_time(args, tmin=-200e-15, tmax=200e-15, Nt=1000, z=3.4)
'''