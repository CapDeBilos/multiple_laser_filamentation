# ============ This code takes some diagnostics from the simulation and ============
# ============ creates different graphs for visualizing the effects     ============
import numpy as np
import matplotlib.pyplot as plt
import os
from beam_profiles import *
from constants import *

'''
Problems to address:
    - if you want to add multiple plots on the same graph, you need to pass ax=None as an
    argument then address this with an if statement in the function
'''

########## For simulations in z, x, y, with artificial time
class BeamSimulationZXY:
    def __init__(self, filepath: str, simulations_root: str, results_root: str):
        self.load(filepath)
        self.sim_dir    = os.path.dirname(os.path.abspath(filepath))
        # self.file_name  = os.path.splitext(os.path.basename(filepath))[0]
        self.file_name  = f'Pin_{self.pin_factor:4.1f}Pcr'.replace('.', 'p')
        self.res_dir    = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.file_name)
        os.makedirs(self.res_dir, exist_ok=True)

    def load(self, filepath: str):
        data                    = np.load(filepath, allow_pickle=True)
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
        self.snaps_n        = np.array([snap / self.I_peak[0] for snap in self.snaps])
        # self.I0_n_fft       = np.abs(np.fft.fft2(self.snaps[0])).max() # the max intensity of the spectrum at z=0, not a good method

    def save(self, fig, name: str):
        fig.savefig(os.path.join(self.res_dir, f'{name}.pdf'), dpi=150)
        plt.close(fig)

    def on_axis_max_vs_z(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.z_diag, self.I_peak_n, color='green', label='$I_{max} / I_0$')
        ax.plot(self.z_diag, self.I_center_n, color='blue', label='$I_{r=0} / I_0$')

        ax.set_title(f'$I_{{max}}(x=0, y=0, z)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I_{{max}}/I_0|$ (1)')
        ax.legend()

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'max_I_vs_z')
        # plt.show()
    
    def profile_x(self, z: float, fig=None, ax=None, save=True):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}m'.replace('.', 'p')
        x = self.snap_x[z_idx]
        snap = self.snaps_n[z_idx]
        I_n = snap[:, snap.shape[1] // 2]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x, I_n, label=f'$I(x, y=0, z={z_val:5.3f}m) / I_0$')
        # ax.set_ylim(0, 10)
        ax.set_title(f'Intensity profile $I(x, y=0, z = {z_val:.3f}m)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            self.save(fig, f'profile_x_z_{z_str}')
        # plt.show()
        return(fig, ax)
    
    def profile_y(self, z: float, fig=None, ax=None, save=True):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        y = self.snap_y[z_idx]
        snap = self.snaps_n[z_idx]
        I_n = snap[snap.shape[0] // 2, :]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(y, I_n, label='$I(x=0, y) / I_0$')
        # ax.set_ylim(0, 10)
        ax.set_title(f'Intensity profile $I(x=0, y, z = {z_val:.3f})/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$y$ (m)')
        ax.set_ylabel('$|I / I_0|$ (1)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, f'profile_y_z_{z_str}')
        # plt.show()
        return(fig, ax)

    def profile_xy(self, z: float):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}m'.replace('.', 'p')
        I_n = self.snaps_n[z_idx].T # because imshow treats array as (rows, cols) = (y, x)

        fig, ax = plt.subplots(figsize=(8, 5))

        c = ax.imshow(I_n, cmap='hot', aspect='equal', # vmin=0, vmax=5,
                extent=[self.snap_x[z_idx][0], self.snap_x[z_idx][-1], self.snap_y[z_idx][0], self.snap_y[z_idx][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x, y, z = {z_val:.3f}m)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        fig.colorbar(c, ax=ax, label='$|I/I_0|$ (1)', fraction=0.046, pad=0.04)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_xlim(-0.0015, 0.0015)
        ax.set_ylim(-0.0015, 0.0015)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'profile_xy_z_{z_str}')
        # plt.show()

    def profile_zx(self):
        I_n = np.array([snap[:, snap.shape[1] // 2] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        c = ax.imshow(I_n, cmap='hot', aspect='auto', # vmin=0, vmax=5,
                extent=[self.snap_z[0], self.snap_z[-1], self.snap_x[0][0], self.snap_x[0][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x, y=0, z)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        fig.colorbar(c, ax=ax, label='$|I/I_0|$ (1)')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$x$ (m)')
        ax.set_ylim(-0.0015, 0.0015)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'profile_zx')
        # plt.show()
    
    def profile_zy(self):
        I_n = np.array([snap[snap.shape[0] // 2, :] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        c = ax.imshow(I_n, cmap='hot', aspect='auto', # vmin=0, vmax=5,
                extent=[self.snap_z[0], self.snap_z[-1], self.snap_y[0][0], self.snap_y[0][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x=0, y, z)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        fig.colorbar(c, ax=ax, label='$|I/I_0|$ (1)')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_ylim(-0.0015, 0.0015)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'profile_zy')
        # plt.show()

    ''' This is an attempt to plot the frequency. It is not correct because we need the
    # phase of E. Can't just do sqrt(I). Leave it for another time.
    def spectrum_xy(self, z: float):
        z_idx = np.clip(np.searchsorted(self.snap_z, z), 0, len(self.snap_z) - 1)
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')

        E_n = np.sqrt(self.snaps[z_idx])
        E_n_fft = np.fft.fftshift(np.fft.fft2(E_n))
        I_n_fft = np.abs(E_n_fft) ** 2

        # I_n_fft = np.abs(np.fft.fftshift(np.fft.fft2(self.snaps[z_idx]))) / self.I0_n_fft
                
        dx = self.snap_x[z_idx][1] - self.snap_x[z_idx][0]
        dy = self.snap_y[z_idx][1] - self.snap_y[z_idx][0]
        kx = np.fft.fftshift(np.fft.fftfreq(self.snaps[z_idx].shape[0], d=dx)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(self.snaps[z_idx].shape[1], d=dy)) * 2 * np.pi

        fig, ax = plt.subplots(figsize=(6, 6))
        c = ax.imshow(I_n_fft.T, aspect='equal', origin='lower', # vmin=0, vmax=5,
                extent=[kx[0], kx[-1], ky[0], ky[-1]],
                cmap='viridis') # norm=LogNorm()
        fig.colorbar(c, ax=ax, label='$|I / I_0|$ (1)', fraction=0.046, pad=0.04)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')

        ax.set_title(f'Intensity spectrum $|I(k_x, k_y, z={z_val:.3f}) / I_0|$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$k_x$ (1/m)')
        ax.set_ylabel('$k_y$ (1/m)')
        ax.set_xlim(-1.5e5, 1.5e5)
        ax.set_ylim(-1.5e5, 1.5e5)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{self.file_name}_spectrum_xy_t_artif_z_{z_str}.png', dpi=150)
        # plt.show()
    '''


class BeamSimulationZXY_Noise:
    def __init__(self, filepath: str, simulations_root: str, results_root: str):
        self.load(filepath)
        self.sim_dir   = os.path.dirname(os.path.abspath(filepath))
        self.file_name = f'Pin_{self.pin_factor:4.1f}Pcr_noise'.replace('.', 'p')
        self.res_dir   = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.file_name)
        os.makedirs(self.res_dir, exist_ok=True)

    def load(self, filepath: str):
        data                         = np.load(filepath, allow_pickle=True)
        self.pin_factor              = data['pin_factor']
        self.Pcr                     = data['Pcr']
        self.z_diag                  = data['z_diag']
        self.I_peak                  = data['I_peak']
        self.I_center                = data['I_center']
        self.rho_peak                = data['rho_peak']
        self.DeltaZ_diag             = data['DeltaZ_diag']
        self.Nx_diag                 = data['Nx_diag']
        self.Ny_diag                 = data['Ny_diag']
        self.x_cpu                   = data['x']
        self.y_cpu                   = data['y']
        self.I_final                 = data['I_final']
        self.rho_final               = data['rho_final']
        self.NOISE_ON                = data['NOISE_ON']
        self.NOISE_SEED              = data['NOISE_SEED']
        self.MULTIPLICATIVE_NOISE_ON = data['MULTIPLICATIVE_NOISE_ON']
        self.amp_noise_rms           = data['amp_noise_rms']
        self.PHASE_NOISE_ON          = data['PHASE_NOISE_ON']
        self.phase_noise_rms         = data['phase_noise_rms']
        self.noise_corr_x_factor     = data['noise_corr_x_factor']
        self.noise_corr_y_factor     = data['noise_corr_y_factor']
        self.amp_noise_clip_sigma    = data['amp_noise_clip_sigma']
        self.I_peak_n                = self.I_peak / self.I_peak[0]
        self.I_center_n              = self.I_center / self.I_peak[0]

    def save(self, fig, name: str):
        fig.savefig(os.path.join(self.res_dir, f'{name}.pdf'), dpi=150)
        plt.close(fig)

    def on_axis_max_vs_z(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.z_diag, self.I_peak_n,   color='green', label='$I_{{max}} / I_0$')
        ax.plot(self.z_diag, self.I_center_n, color='blue',  label='$I_{r=0} / I_0$')

        ax.set_title(f'$I_{{max}}(x=0, y=0, z)/I_0$, Pin={self.pin_factor}Pcr, noisy input, artificial time')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I_{{max}}/I_0|$ (1)')
        ax.legend()

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, 'max_I_vs_z')


class BeamSimulationZXY_hermite:
    def __init__(self, filepath: str, simulations_root: str, results_root: str):
        self.load(filepath)
        self.sim_dir    = os.path.dirname(os.path.abspath(filepath))
        # self.file_name  = os.path.splitext(os.path.basename(filepath))[0]
        self.file_name  = f'Pin_{self.pin_factor:4.1f}Pcr'.replace('.', 'p')
        self.res_dir    = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.file_name)
        os.makedirs(self.res_dir, exist_ok=True)

    def load(self, filepath: str):
        data                    = np.load(filepath, allow_pickle=True)
        self.pin_factor         = data['pin_factor']
        self.Pcr                = data['Pcr']
        self.z_diag             = data['z_diag']          # 1d array, z values at cheap diagnostic
        self.I_peak             = data['I_peak']
        # self.I_center           = data['I_center']
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
        # self.I_center_n     = self.I_center / self.I_peak[0]
        self.snaps_n        = np.array([snap / self.I_peak[0] for snap in self.snaps])
        # self.I0_n_fft       = np.abs(np.fft.fft2(self.snaps[0])).max() # the max intensity of the spectrum at z=0, not a good method

    def save(self, fig, name: str):
        fig.savefig(os.path.join(self.res_dir, f'{name}.pdf'), dpi=150)
        plt.close(fig)

    def on_axis_max_vs_z(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.z_diag, self.I_peak_n, color='green', label='$I_{max} / I_0$')
        # ax.plot(self.z_diag, self.I_center_n, color='blue', label='$I_{r=0} / I_0$')

        ax.set_title(f'$I_{{max}}(x=0, y=0, z)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I_{{max}}/I_0|$ (1)')
        ax.legend()

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'max_I_vs_z')
        # plt.show()
    
    def profile_x(self, z: float, fig=None, ax=None, save=True):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        x = self.snap_x[z_idx]
        snap = self.snaps_n[z_idx]
        I_n = snap[:, snap.shape[1] // 2]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x, I_n, label='$I(x, y=0) / I_0$')
        # ax.set_ylim(0, 10)
        ax.set_title(f'Intensity profile $I(x, y=0, z = {z_val:.3f})/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, f'profile_x_z_{z_str}')
        # plt.show()
        return(fig, ax)
    
    def profile_y(self, z: float, fig=None, ax=None, save=True):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        y = self.snap_y[z_idx]
        snap = self.snaps_n[z_idx]
        I_n = snap[snap.shape[0] // 2, :]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(y, I_n, label='$I(x=0, y) / I_0$')
        # ax.set_ylim(0, 10)
        ax.set_title(f'Intensity profile $I(x=0, y, z = {z_val:.3f})/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        ax.set_xlabel('$y$ (m)')
        ax.set_ylabel('$|I / I_0|$ (1)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, f'profile_y_z_{z_str}')
        # plt.show()
        return(fig, ax)

    def profile_xy(self, z: float):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        I_n = self.snaps_n[z_idx].T # because imshow treats array as (rows, cols) = (y, x)

        fig, ax = plt.subplots(figsize=(6, 6))
        c = ax.imshow(I_n, cmap='hot', aspect='equal', # vmin=0, vmax=5,
                extent=[self.snap_x[z_idx][0], self.snap_x[z_idx][-1], self.snap_y[z_idx][0], self.snap_y[z_idx][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x, y, z = {z_val:.3f})/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        fig.colorbar(c, ax=ax, label='$|I/I_0|$ (1)', fraction=0.046, pad=0.04)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_xlim(-0.0035, 0.0035)
        ax.set_ylim(-0.0035, 0.0035)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'profile_xy_z_{z_str}')
        # plt.show()

    def profile_zx(self):
        I_n = np.array([snap[:, snap.shape[1] // 2] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        c = ax.imshow(I_n, cmap='hot', aspect='auto', # vmin=0, vmax=5,
                extent=[self.snap_z[0], self.snap_z[-1], self.snap_x[0][0], self.snap_x[0][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x, y=0, z)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        fig.colorbar(c, ax=ax, label='$|I/I_0|$ (1)')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$x$ (m)')
        ax.set_ylim(-0.0035, 0.0035)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'profile_zx')
        # plt.show()
    
    def profile_zy(self):
        I_n = np.array([snap[snap.shape[0] // 2, :] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        c = ax.imshow(I_n, cmap='hot', aspect='auto', # vmin=0, vmax=5,
                extent=[self.snap_z[0], self.snap_z[-1], self.snap_y[0][0], self.snap_y[0][-1]],
                origin='lower')
        ax.set_title(f'Intensity profile $I(x=0, y, z)/I_0$, Pin={self.pin_factor}Pcr, artificial time')
        fig.colorbar(c, ax=ax, label='$|I/I_0|$ (1)')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_ylim(-0.0035, 0.0035)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, f'profile_zy')
        # plt.show()


class BeamSimulationZRT:
    def __init__(self, filepath: str, simulations_root: str, results_root: str):
        self.load(filepath)
        self.sim_dir   = os.path.dirname(os.path.abspath(filepath))
        self.file_name = f'Pin_{self.pin_factor:4.1f}Pcr'.replace('.', 'p')
        self.res_dir   = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.file_name)
        os.makedirs(self.res_dir, exist_ok=True)

    def load(self, filepath: str):
        data             = np.load(filepath, allow_pickle=True)
        self.pin_factor  = data['pin_factor']
        self.z           = data['z']
        self.I_axis_max_t = data['I_axis_max_t']
        self.I_ratio     = data['I_ratio']

    def save(self, fig, name: str):
        fig.savefig(os.path.join(self.res_dir, f'{name}.pdf'), dpi=150)
        plt.close(fig)

    def on_axis_max_vs_z(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.z, self.I_ratio, color='green', label='$I_{max} / I_0$')

        ax.set_title(f'$I_{{max}}(r=0, z)/I_0$, Pin={self.pin_factor}Pcr')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I_{{max}}/I_0|$ (1)')
        ax.legend()

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save(fig, 'max_I_vs_z')


class BeamSimulationZXT:
    def __init__(self, filepath: str, simulations_root: str, results_root: str):
        self.load(filepath)
        self.sim_dir   = os.path.dirname(os.path.abspath(filepath))
        self.file_name = f'Pin_{self.pin_factor:4.1f}Pcr'.replace('.', 'p')
        self.res_dir   = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.file_name)
        os.makedirs(self.res_dir, exist_ok=True)

    def load(self, filepath: str):
        data                 = np.load(filepath, allow_pickle=True)
        self.pin_factor      = data['pin_factor']
        self.Pcr             = data['Pcr']
        self.z_diag          = data['z_diag']
        self.I_peak          = data['I_peak'].item()       # scalar
        self.I_center_tmax   = data['I_center_tmax']       # 1d array over z
        self.DeltaZ_diag     = data['DeltaZ_diag']
        self.Nx_diag         = data['Nx_diag']
        self.Nt_diag         = data['Nt_diag']
        self.x_cpu           = data['x']
        self.t_cpu           = data['t']
        self.I_final         = data['I_final']
        self.I_center_tmax_n = self.I_center_tmax / self.I_peak   # normalize by initial peak

    def save(self, fig, name: str):
        fig.savefig(os.path.join(self.res_dir, f'{name}.pdf'), dpi=150)
        plt.close(fig)
    
    def on_axis_max_vs_z(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.plot(self.z_diag, self.I_center_tmax_n, color='green', label='$I(x=0, t_{max}) / I_0$')
            ax.set_title(f'$I(x=0, t_{{max}}, z)/I_0$, Pin={self.pin_factor}Pcr')
            ax.set_xlabel('$z$ (m)')
            ax.set_ylabel('$|I/I_0|$ (1)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            self.save(fig, 'max_I_vs_z')
        except Exception as e:
            plt.close(fig)
            raise e