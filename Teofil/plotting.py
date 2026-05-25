# ============ This code takes some diagnostics from the simulation and ============
# ============ creates different graphs for visualizing the effects     ============
import numpy as np
import matplotlib.pyplot as plt
import os
from beam_profiles import *
from constants import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

'''
Problems to address:
    - if you want to add multiple plots on the same graph, you need to pass ax=None as an
    argument then address this with an if statement in the function
'''

########## For simulations in z, x, y, with artificial time
class BeamSimulationZXY:
    def __init__(self, filepath: str, simulations_root: str, results_root: str = None):
        self.load(filepath)
        if results_root is None:
            results_root = simulations_root
        self.sim_dir = os.path.dirname(os.path.abspath(filepath))
        self.res_dir = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.pin_str)

    def load(self, filepath: str):
        data                   = np.load(filepath, allow_pickle=True)
        self.pin_factor        = data['pin_factor']
        self.Pcr               = data['Pcr']
        self.z_diag            = data['z_diag']
        self.I_peak            = data['I_peak']
        self.I_center          = data['I_center']
        self.rho_peak          = data['rho_peak']
        self.DeltaZ_diag       = data['DeltaZ_diag']
        self.Nx_diag           = data['Nx_diag']
        self.Ny_diag           = data['Ny_diag']
        self.snap_z            = data['I_snapshot_z']
        self.snap_steps        = data['I_snapshot_steps']
        self.snap_Nx           = data['I_snapshot_Nx']
        self.snap_Ny           = data['I_snapshot_Ny']
        self.snap_x            = data['I_snapshot_x']
        self.snap_y            = data['I_snapshot_y']
        self.snaps             = data['I_snapshots']
        self.full_I_save_every = data['full_I_save_every']
        self.x_cpu             = data['x']
        self.y_cpu             = data['y']
        self.I_final           = data['I_final']
        self.rho_final         = data['rho_final']
        self.pin_str           = f'{self.pin_factor:04.1f}Pcr'.replace('.', 'p')
        self.I_peak_n          = self.I_peak / self.I_peak[0]
        self.I_center_n        = self.I_center / self.I_peak[0]
        self.snaps_n           = np.array([snap / self.I_peak[0] for snap in self.snaps])

    def save(self, fig, name: str, res_dir: str = None):
        out = res_dir if res_dir is not None else self.res_dir
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_axis_max_vs_z(self, fig=None, ax=None, save=True, res_dir=None, ylim=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
            
        ax.plot(self.z_diag, self.I_peak_n,   label=f'$I_{{max}}/I_0$, Pin={self.pin_factor:.1f}Pcr')
        ax.plot(self.z_diag, self.I_center_n, label=f'$I_{{r=0}}/I_0$, Pin={self.pin_factor:.1f}Pcr', linestyle='--')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$I/I_0$ (1)')
        if ylim:
            ax.set_ylim(0, ylim)
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.grid(True, alpha=0.3)
        if save:
            ax.set_title(f'$I_{{max}}$ vs $z$, Pin={self.pin_factor:.1f}Pcr, artificial time')
            ax.legend()
            self.save(fig, name=f'max_I_vs_z_{self.pin_str}', res_dir=res_dir)
        return fig, ax

    def profile_x(self, z: float, fig=None, ax=None, save=True, res_dir=None):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        I_n   = self.snaps_n[z_idx][:, self.snaps_n[z_idx].shape[1] // 2]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.snap_x[z_idx], I_n, label=f'$I(x, y=0, z={z_val:5.3f}m) / I_0$')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, name=f'profile_x_z_{z_str}', res_dir=res_dir)
        return fig, ax

    def profile_y(self, z: float, fig=None, ax=None, save=True, res_dir=None):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        I_n   = self.snaps_n[z_idx][self.snaps_n[z_idx].shape[0] // 2, :]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.snap_y[z_idx], I_n, label=f'$I(x=0, y, z={z_val:5.3f}m) / I_0$')
        ax.set_xlabel('$y$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, name=f'profile_y_z_{z_str}', res_dir=res_dir)
        return fig, ax

    def profile_xy(self, z: float, save=True, res_dir=None, vmax=True):
        z_idx  = np.argmin(np.abs(self.snap_z - z))
        z_val  = self.snap_z[z_idx]
        z_str  = f'{z_val:5.3f}'.replace('.', 'p')
        I_n    = self.snaps_n[z_idx].T
        _vmax = 0.7 * np.max(self.I_peak_n) if vmax else None

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.10)  # fixed margins

        c = ax.imshow(I_n, cmap='hot', aspect='equal', vmin=0, vmax=_vmax,
                      extent=[self.snap_x[z_idx][0], self.snap_x[z_idx][-1],
                               self.snap_y[z_idx][0], self.snap_y[z_idx][-1]],
                      origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, label='$|I/I_0|$ (1)')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

        ax.set_title(f'$I(x, y, z={z_val:.3f}m)/I_0$, Pin={self.pin_factor:.1f}Pcr')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_xlim(-0.0015, 0.0015)
        ax.set_ylim(-0.0015, 0.0015)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # fixed exponent format
        ax.grid(True, alpha=0.3)

        if save:
            self.save(fig, name=f'profile_xy_z_{z_str}', res_dir=res_dir)

    def profile_zx(self, save=True, res_dir=None, vmax=None):
        I_n = np.array([snap[:, snap.shape[1] // 2] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.10)  # fixed margins

        c = ax.imshow(I_n, cmap='hot', aspect='auto', vmin=0, vmax=vmax,
                      extent=[self.snap_z[0], self.snap_z[-1],
                               self.snap_x[0][0], self.snap_x[0][-1]],
                      origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, label='$|I/I_0|$ (1)')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

        ax.set_title(f'$I(x, y=0, z)/I_0$, Pin={self.pin_factor:.1f}Pcr')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$x$ (m)')
        ax.set_ylim(-0.0015, 0.0015)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # fixed exponent format
        ax.grid(True, alpha=0.3)

        if save:
            self.save(fig, name=f'profile_zx_{self.pin_str}', res_dir=res_dir)

    def profile_zy(self, save=True, res_dir=None, vmax=None):
        I_n = np.array([snap[snap.shape[0] // 2, :] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.10)  # fixed margins

        c = ax.imshow(I_n, cmap='hot', aspect='auto', vmin=0, vmax=vmax,
                      extent=[self.snap_z[0], self.snap_z[-1],
                               self.snap_y[0][0], self.snap_y[0][-1]],
                      origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, label='$|I/I_0|$ (1)')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

        ax.set_title(f'$I(x=0, y, z)/I_0$, Pin={self.pin_factor:.1f}Pcr')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_ylim(-0.0015, 0.0015)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # fixed exponent format
        ax.grid(True, alpha=0.3)
        if save:
            self.save(fig, name=f'profile_zy_{self.pin_str}', res_dir=res_dir)

class BeamSimulationZXY_Noise:
    def __init__(self, filepath: str, simulations_root: str, results_root: str = None):
        self.load(filepath)
        if results_root is None:
            results_root = simulations_root
        self.sim_dir = os.path.dirname(os.path.abspath(filepath))
        self.res_dir = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.pin_str)

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
        self.pin_str                 = f'{self.pin_factor:04.1f}Pcr'.replace('.', 'p')
        self.I_peak_n                = self.I_peak / self.I_peak[0]
        self.I_center_n              = self.I_center / self.I_peak[0]

    def save(self, fig, name: str, res_dir: str = None):
        out = res_dir if res_dir is not None else self.res_dir
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_axis_max_vs_z(self, fig=None, ax=None, save=True, res_dir=None, ylim=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

        ax.plot(self.z_diag, self.I_peak_n,   label=f'$I_{{max}}/I_0$, Pin={self.pin_str} (noise)')
        ax.plot(self.z_diag, self.I_center_n, label=f'$I_{{r=0}}/I_0$, Pin={self.pin_str} (noise)', linestyle='--')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$I/I_0$ (1)')
        if ylim:
            ax.set_ylim(0, ylim)
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.grid(True, alpha=0.3)
        if save:
            ax.set_title(f'$I_{{max}}$ vs $z$, Pin={self.pin_str}, noisy, artificial time')
            ax.legend()
            out = os.path.join(res_dir, 'max_I_z') if res_dir is not None else os.path.join(self.res_dir, 'max_I_z')
            self.save(fig, name=f'max_I_vs_z_{self.pin_str}', res_dir=out)
        return fig, ax

class BeamSimulationZXY_hermite:
    def __init__(self, filepath: str, simulations_root: str, results_root: str = None):
        self.load(filepath)
        if results_root is None:
            results_root = simulations_root
        self.sim_dir = os.path.dirname(os.path.abspath(filepath))
        self.res_dir = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.pin_str)

    def load(self, filepath: str):
        data                   = np.load(filepath, allow_pickle=True)
        self.pin_factor        = data['pin_factor']
        self.Pcr               = data['Pcr']
        self.z_diag            = data['z_diag']
        self.I_peak            = data['I_peak']
        self.rho_peak          = data['rho_peak']
        self.DeltaZ_diag       = data['DeltaZ_diag']
        self.Nx_diag           = data['Nx_diag']
        self.Ny_diag           = data['Ny_diag']
        self.snap_z            = data['I_snapshot_z']
        self.snap_steps        = data['I_snapshot_steps']
        self.snap_Nx           = data['I_snapshot_Nx']
        self.snap_Ny           = data['I_snapshot_Ny']
        self.snap_x            = data['I_snapshot_x']
        self.snap_y            = data['I_snapshot_y']
        self.snaps             = data['I_snapshots']
        self.full_I_save_every = data['full_I_save_every']
        self.x_cpu             = data['x']
        self.y_cpu             = data['y']
        self.I_final           = data['I_final']
        self.rho_final         = data['rho_final']
        self.pin_str           = f'{self.pin_factor:04.1f}Pcr'.replace('.', 'p')
        self.I0                = np.max(self.snaps[0])
        self.I_peak_n          = self.I_peak / self.I_peak[0]
        self.snaps_n           = np.array([snap / self.I_peak[0] for snap in self.snaps])

    def save(self, fig, name: str, res_dir: str = None):
        out = res_dir if res_dir is not None else self.res_dir
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_axis_max_vs_z(self, fig=None, ax=None, save=True, res_dir=None, ylim=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
        
        ax.plot(self.z_diag, self.I_peak_n, label=f'$I_{{max}}/I_0$, Pin={self.pin_str}')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I_{{max}}/I_0|$ (1)')
        if ylim:
            ax.set_ylim(0, ylim)
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.grid(True, alpha=0.3)
        if save:
            ax.set_title(f'$I_{{max}}(x=0, y=0, z)/I_0$, Pin={self.pin_str}, artificial time')
            ax.legend()
            self.save(fig, name=f'max_I_vs_z_{self.pin_str}', res_dir=res_dir)
        return fig, ax

    def profile_x(self, z: float, fig=None, ax=None, save=True, res_dir=None):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        I_n   = self.snaps_n[z_idx][:, self.snaps_n[z_idx].shape[1] // 2]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.snap_x[z_idx], I_n, label=f'$I(x, y=0, z={z_val:5.3f}m) / I_0$')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, name=f'profile_x_z_{z_str}', res_dir=res_dir)
        return fig, ax

    def profile_y(self, z: float, fig=None, ax=None, save=True, res_dir=None):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:.3f}'.replace('.', 'p')
        I_n   = self.snaps_n[z_idx][self.snaps_n[z_idx].shape[0] // 2, :]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.snap_y[z_idx], I_n, label=f'$I(x=0, y, z={z_val:5.3f}m) / I_0$')
        ax.set_xlabel('$y$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            self.save(fig, name=f'profile_y_z_{z_str}', res_dir=res_dir)
        return fig, ax

    def profile_xy(self, z: float, save=True, res_dir=None, vmax=False):
        z_idx = np.argmin(np.abs(self.snap_z - z))
        z_val = self.snap_z[z_idx]
        z_str = f'{z_val:5.3f}'.replace('.', 'p')
        I_n   = self.snaps_n[z_idx].T
        _vmax = 0.7 * np.max(self.I_peak_n) if vmax else None

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.10)  # fixed margins

        c = ax.imshow(I_n, cmap='hot', aspect='equal', vmin=0, vmax=_vmax,
                      extent=[self.snap_x[z_idx][0], self.snap_x[z_idx][-1],
                               self.snap_y[z_idx][0], self.snap_y[z_idx][-1]],
                      origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, label='$|I/I_0|$ (1)')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        
        ax.set_title(f'$I(x, y, z={z_val:.3f}m)/I_0$, Pin={self.pin_str}')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_xlim(-0.0025, 0.0025)
        ax.set_ylim(-0.0025, 0.0025)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # fixed exponent format
        ax.grid(True, alpha=0.3)
        if save:
            self.save(fig, name=f'profile_xy_z_{z_str}', res_dir=res_dir)

    def profile_zx(self, save=True, res_dir=None, vmax=None):
        I_n = np.array([snap[:, snap.shape[1] // 2] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.10)  # fixed margins

        c = ax.imshow(I_n, cmap='hot', aspect='auto', vmin=0, vmax=vmax,
                      extent=[self.snap_z[0], self.snap_z[-1],
                               self.snap_x[0][0], self.snap_x[0][-1]],
                      origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, label='$|I/I_0|$ (1)')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        
        ax.set_title(f'$I(x, y=0, z)/I_0$, Pin={self.pin_str}')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$x$ (m)')
        ax.set_ylim(-0.0030, 0.0030)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # fixed exponent format
        ax.grid(True, alpha=0.3)
        if save:
            self.save(fig, name=f'profile_zx_{self.pin_str}', res_dir=res_dir)

    def profile_zy(self, save=True, res_dir=None, vmax=None):
        I_n = np.array([snap[snap.shape[0] // 2, :] for snap in self.snaps_n]).T

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.10)  # fixed margins

        c = ax.imshow(I_n, cmap='hot', aspect='auto', vmin=0, vmax=vmax,
                      extent=[self.snap_z[0], self.snap_z[-1],
                               self.snap_y[0][0], self.snap_y[0][-1]],
                      origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, label='$|I/I_0|$ (1)')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

        ax.set_title(f'$I(x=0, y, z)/I_0$, Pin={self.pin_str}')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_ylim(-0.0030, 0.0030)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # fixed exponent format
        ax.grid(True, alpha=0.3)
        if save:
            self.save(fig, name=f'profile_zy_{self.pin_str}', res_dir=res_dir)

class BeamSimulationZXT:
    def __init__(self, filepath: str, simulations_root: str, results_root: str = None):
        self.load(filepath)
        if results_root is None:
            results_root = simulations_root
        self.sim_dir  = os.path.dirname(os.path.abspath(filepath))
        self.res_dir  = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.pin_str)

    def load(self, filepath: str):
        data                 = np.load(filepath, allow_pickle=True)
        self.pin_factor      = data['pin_factor']
        self.Pcr             = data['Pcr']
        self.z_diag          = data['z_diag']
        self.I_peak          = data['I_peak'].item()
        self.I_center_tmax   = data['I_center_tmax']
        self.DeltaZ_diag     = data['DeltaZ_diag']
        self.Nx_diag         = data['Nx_diag']
        self.Nt_diag         = data['Nt_diag']
        self.x_cpu           = data['x']
        self.t_cpu           = data['t']
        self.I_final         = data['I_final']
        self.I_center_tmax_n = self.I_center_tmax / self.I_peak
        self.pin_str         = f'{self.pin_factor:04.1f}Pcr'.replace('.', 'p')

    def save(self, fig, name: str, res_dir: str = None):
        out = res_dir if res_dir is not None else self.res_dir
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'{name}.pdf'), dpi=150, bbox_inches="tight")
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_axis_max_vs_z(self, fig=None, ax=None, save=True, res_dir=None, ylim=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

        ax.plot(self.z_diag, self.I_center_tmax_n,
                label=f'$I_{{max}}(x=0, z)/I_0$, Pin={self.pin_factor:.1f}Pcr')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I/I_0|$ (1)')
        if ylim:
            ax.set_ylim(0, ylim)
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.grid(True, alpha=0.3)
        if save:
            ax.set_title(f'$I(x=0, t_{{max}}, z)/I_0$, Pin={self.pin_factor:.1f}Pcr')
            ax.legend()
            out = os.path.join(res_dir, 'max_I_z') if res_dir is not None else os.path.join(self.res_dir, 'max_I_z')
            self.save(fig, name=f'max_I_vs_z_{self.pin_str}', res_dir=out)
        return fig, ax

class BeamSimulationZRT:
    def __init__(self, filepath: str, simulations_root: str, results_root: str = None):
        self.load(filepath)
        if results_root is None:
            results_root = simulations_root
        self.sim_dir = os.path.dirname(os.path.abspath(filepath))
        self.res_dir = os.path.join(results_root, os.path.relpath(self.sim_dir, simulations_root), self.pin_str)

    def load(self, filepath: str):
        data              = np.load(filepath, allow_pickle=True)
        self.pin_factor   = data['pin_factor']
        self.z            = data['z']
        self.I_axis_max_t = data['I_axis_max_t']
        self.I_ratio      = data['I_ratio']
        self.pin_str      = f'{self.pin_factor:04.1f}Pcr'.replace('.', 'p')

    def save(self, fig, name: str, res_dir: str = None):
        out = res_dir if res_dir is not None else self.res_dir
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'{name}.pdf'), dpi=150, bbox_inches="tight")
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_axis_max_vs_z(self, fig=None, ax=None, save=True, res_dir=None, ylim=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
        
        ax.plot(self.z, self.I_ratio, label=f'$I_{{max}}/I_0$, Pin={self.pin_factor:.1f}Pcr')
        ax.set_xlabel('$z$ (m)')
        ax.set_ylabel('$|I_{max}/I_0|$ (1)')
        if ylim:
            ax.set_ylim(0, ylim)
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
        ax.grid(True, alpha=0.3)
        if save:
            ax.set_title(f'$I_{{max}}(r=0, z)/I_0$, Pin={self.pin_factor:.1f}Pcr')
            ax.legend()
            out = os.path.join(res_dir, 'max_I_z') if res_dir is not None else os.path.join(self.res_dir, 'max_I_z')
            self.save(fig, name=f'max_I_vs_z_{self.pin_str}', res_dir=out)
        return fig, ax