import glob
from plotting import *

########## Run the code on real data
# run the plots for all subdirs of the Simulations directory, saving automatically to the Results directory
def run_ZXY(directory: str, results_root: str, zs, subdir: str = '', ylim_I=None, vmax_zx=None, vmax_zy=None, vmax_xy=False):
    for filepath in glob.glob(os.path.join(directory, '*.npz')):
        print(f'Processing {filepath}...')
        try:
            sim = BeamSimulationZXY(filepath, simulations_root=directory, results_root=results_root)
            out_dir = os.path.join(results_root, subdir) if subdir else results_root

            sim.on_axis_max_vs_z(res_dir=os.path.join(out_dir, 'max_I_z'), ylim=ylim_I)
            sim.profile_zx(res_dir=os.path.join(out_dir, 'zx'), vmax=vmax_zx)
            sim.profile_zy(res_dir=os.path.join(out_dir, 'zy'), vmax=vmax_zy)

            for z in zs:
                sim.profile_xy(z=z, res_dir=os.path.join(out_dir, 'xy', f'profile_xy_{sim.pin_str}'), vmax=vmax_xy)

            print(f'Done — saved in {out_dir}')
        except Exception as e:
            print(f'Error processing {filepath}: {e}')

def run_ZXY_Noise(directory: str, results_root: str, subdir: str = 'max_I_z', ylim=None):
    out_dir = os.path.join(results_root, subdir)
    for filepath in glob.glob(os.path.join(directory, '*.npz')):
        print(f'Processing {filepath}...')
        try:
            sim = BeamSimulationZXY_Noise(filepath, simulations_root=directory, results_root=results_root)
            sim.on_axis_max_vs_z(res_dir=out_dir, ylim=ylim)
            print(f'Done — saved in {out_dir}')
        except Exception as e:
            print(f'Error processing {filepath}: {e}')

def run_ZXY_hermite(directory: str, results_root: str, zs, subdir: str = '', ylim_I=None, vmax_zx=None, vmax_zy=None, vmax_xy=False):
    for filepath in glob.glob(os.path.join(directory, '*.npz')):
        print(f'Processing {filepath}...')
        try:
            sim = BeamSimulationZXY_hermite(filepath, simulations_root=directory, results_root=results_root)
            out_dir = os.path.join(results_root, subdir) if subdir else results_root

            sim.on_axis_max_vs_z(res_dir=os.path.join(out_dir, 'max_I_z'), ylim=ylim_I)
            sim.profile_zx(res_dir=os.path.join(out_dir, 'zx'), vmax=vmax_zx)
            sim.profile_zy(res_dir=os.path.join(out_dir, 'zy'), vmax=vmax_zy)

            for z in zs:
                sim.profile_xy(z=z, res_dir=os.path.join(out_dir, 'xy', f'profile_xy_{sim.pin_str}'), vmax=vmax_xy)

            print(f'Done — saved in {out_dir}')
        except Exception as e:
            print(f'Error processing {filepath}: {e}')

def run_ZXT(directory: str, results_root: str, subdir: str = 'max_I_z', ylim=None):
    out_dir = os.path.join(results_root, subdir)
    for filepath in glob.glob(os.path.join(directory, '*.npz')):
        print(f'Processing {filepath}...')
        try:
            sim = BeamSimulationZXT(filepath, simulations_root=directory, results_root=results_root)
            sim.on_axis_max_vs_z(res_dir=out_dir, ylim=ylim)
            print(f'Done — saved in {out_dir}')
        except Exception as e:
            print(f'Error processing {filepath}: {e}')

def run_ZRT(directory: str, results_root: str, subdir: str = 'max_I_z', ylim=None):
    out_dir = os.path.join(results_root, subdir)
    for filepath in glob.glob(os.path.join(directory, '*.npz')):
        print(f'Processing {filepath}...')
        try:
            sim = BeamSimulationZRT(filepath, simulations_root=directory, results_root=results_root)
            sim.on_axis_max_vs_z(res_dir=out_dir, ylim=ylim)
            print(f'Done — saved in {out_dir}')
        except Exception as e:
            print(f'Error processing {filepath}: {e}')


# ZXY, gaussian, no noise
# run_ZXY(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/no_noise/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/no_noise/',
    # np.arange(0.0, 4.0, 0.079611),
    # subdir='dynamic',
    # ylim_I=None, vmax_zx=None, vmax_zy=None, vmax_xy=False,
# )

# ZXY, gaussian, with noise
# run_ZXY_Noise(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/noise/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/noise/',
    # subdir='dynamic',
    # ylim=None, # 90
# )

# ZXY, square
# run_ZXY(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/square/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/square/',
    # np.arange(0.0, 4.0, 0.079611),
    # subdir='dynamic',
    # ylim_I=None, vmax_zx=None, vmax_zy=None, vmax_xy=False, # 35, 20, 20, False
# )

# ZXY, hermite
# run_ZXY_hermite(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/hermite_2_1/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/hermite_2_1/',
    # np.arange(0.0, 4.0, 0.079611),
    # subdir='dynamic',
    # ylim_I=None, vmax_zx=None, vmax_zy=None, vmax_xy=False # 40, 10, 10, True
# )

# ZXT gaussian or square
# run_ZXT(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/square/',
    # subdir='dynamic',
    # ylim=None,
# )

# ZRT, gaussian, no plasma
# run_ZRT(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/no_plasma/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/no_plasma/',
    # subdir='dynamic',
    # ylim=None,
# )

# ZRT, gaussian, plasma
# run_ZRT(
    # '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/',
    # '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/plasma/',
    # subdir='dynamic',
    # ylim=None,
# )





########## Manual plotting
def compare_profile_x_gaussian(pin_factor, zs):
    pin_str = f"{pin_factor:05.1f}".replace(".", "p")
    sim = BeamSimulationZXY(f'/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/no_noise/many_diags/Pin_{pin_str}_Pcr_gaussian_4D_FFT_diagnostics.npz',
                            '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/no_noise/many_diags/',
                            '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/no_noise/many_diags/a_manual')
    fig, ax = sim.profile_x(z=0, save=False)
    for z in zs: # np.arange(0.239, 1.435, 0.239)
        fig, ax = sim.profile_x(z, fig=fig, ax=ax, save=False)
    ax.set_title(f'Intensity profile $I(x, y=0, z)/I_0$, Pin={sim.pin_factor}Pcr, artificial time')
    ax.legend()
    sim.save(fig, 'profile_x_sweep_z')

def compare_profile_x_square(pin_factor, zs):
    pin_str = f"{pin_factor:05.1f}".replace(".", "p")
    sim = BeamSimulationZXY(f'/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/square/Pin_{pin_str}_Pcr_square_4D_FFT_diagnostics.npz',
                            '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/square',
                            '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/square/a_manual')
    fig, ax = sim.profile_x(z=0, save=False)
    for z in zs: # np.arange(0.239, 1.435, 0.239)
        fig, ax = sim.profile_x(z, fig=fig, ax=ax, save=False)
    ax.set_xlim(-0.002, 0.002)
    ax.set_title(f'Intensity profile $I(x, y=0, z)/I_0$, Pin={sim.pin_factor}Pcr, artificial time')
    ax.legend()
    sim.save(fig, 'profile_x_sweep_z')

def compare_on_axis_max_vs_z(sims: list, res_dir: str, name: str = 'compare_max_I_vs_z'):
    """
    Parameters
    ----------
    sims     : list of BeamSimulationZXY or BeamSimulationZXY_Noise instances
    res_dir  : directory where the combined figure is saved
    name     : filename (without extension)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for sim in sims:
        fig, ax = sim.plot_on_axis_max_vs_z(fig=fig, ax=ax, save=False)

    ax.set_title('$I_{max}$ and $I_{r=0}$ vs $z$, comparison')
    ax.legend()
    plt.tight_layout()

    os.makedirs(res_dir, exist_ok=True)
    fig.savefig(os.path.join(res_dir, f'{name}.pdf'), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(res_dir, f'{name}.png'), dpi=150, bbox_inches="tight")
    plt.close(fig)

def compare_on_axis_max_vs_z_ZXT(sims: list, res_dir: str, name: str = 'compare_max_I_vs_z'):
    """
    Parameters
    ----------
    sims    : list of BeamSimulationZXT instances
    res_dir : directory where the combined figure is saved
    name    : filename (without extension)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for sim in sims:
        fig, ax = sim.plot_on_axis_max_vs_z(fig=fig, ax=ax, save=False)

    ax.set_title('$I(x=0, t_{max}, z)/I_0$ — comparison')
    ax.legend()
    plt.tight_layout()

    os.makedirs(res_dir, exist_ok=True)
    fig.savefig(os.path.join(res_dir, f'{name}.pdf'), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(res_dir, f'{name}.png'), dpi=150, bbox_inches="tight")
    plt.close(fig)

def compare_on_axis_max_vs_z_ZRT(sims: list, res_dir: str, name: str = 'compare_max_I_vs_z'):
    """
    Parameters
    ----------
    sims    : list of BeamSimulationZRT instances
    res_dir : directory where the combined figure is saved
    name    : filename (without extension)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for sim in sims:
        fig, ax = sim.on_axis_max_vs_z(fig=fig, ax=ax, save=False)

    ax.set_title('$I(x=0, t_{max}, z)/I_0$ — comparison')
    ax.legend()
    plt.tight_layout()

    os.makedirs(res_dir, exist_ok=True)
    fig.savefig(os.path.join(res_dir, f'{name}.pdf'), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(res_dir, f'{name}.png'), dpi=150, bbox_inches="tight")
    plt.close(fig)

'''
for pin_factor in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    compare_profile_x_gaussian(pin_factor, zs=[0.239, 0.955, 1.194, 1.433])
compare_profile_x_square(pin_factor=20.0, zs=[0.159, 0.318, 0.955])
'''

'''
sim_clean = BeamSimulationZXY(
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/no_noise/many_diags/Pin_018p0_Pcr_gaussian_4D_FFT_diagnostics.npz',
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/no_noise/many_diags/',
    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/a_manual/'
)
sim_noisy = BeamSimulationZXY_Noise(
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/noise/Pin_018p0_Pcr_xy_no_t_memory_optimized_noise.npz',
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/noise/',
    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/a_manual/'
)
compare_on_axis_max_vs_z(
    sims=[sim_clean, sim_noisy],
    res_dir='/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/a_manual/',
    name='compare_18p0Pcr'
)
# '''

'''
sim_1 = BeamSimulationZXT(
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/Pin_000p1_Pcr.npz',
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/',
    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/square/a_manual/'
)
sim_2 = BeamSimulationZXT(
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/Pin_006p7_Pcr.npz',
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/',
    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/square/a_manual/'
)
sim_3 = BeamSimulationZXT(
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/Pin_020p0_Pcr.npz',
    '/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/square/',
    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/square/a_manual/'
)

compare_on_axis_max_vs_z_ZXT(
    sims=[sim_1, sim_2, sim_3],
    res_dir='/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/square/a_manual',
    name='compare_square'
)
# '''

# graphs similar to Mlejnek 1998
sim_1 = BeamSimulationZRT(
    filepath            = '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/Pin_002p0_Pcr.npz',
    simulations_root    = '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/',
    results_root        = '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/plasma/a_manual/',
)

sim_2 = BeamSimulationZRT(
    filepath            = '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/Pin_002p9_Pcr.npz',
    simulations_root    = '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/',
    results_root        = '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/plasma/a_manual/',
)

sim_3 = BeamSimulationZRT(
    filepath            = '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/Pin_003p8_Pcr.npz',
    simulations_root    = '/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/plasma/',
    results_root        = '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/plasma/a_manual/',
)

compare_on_axis_max_vs_z_ZRT(
    sims        = [sim_1, sim_2, sim_3],
    res_dir     = '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/plasma/a_manual/',
    name        = 'mlejnek_comparison',
)

