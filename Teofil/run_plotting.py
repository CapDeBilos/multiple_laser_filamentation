from plotting import *

########## Run the code on real data
# run the plots for all subdirs of the Simulations directory, saving automatically to the Results directory
# this works for zxy gaussian and zxy square
def run_ZXY(simulations_root: str, results_root: str, zs):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('01p2_Pcr_gaussian_4D_FFT_diagnostics.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZXY(filepath, simulations_root, results_root)
                    sim.on_axis_max_vs_z()
                    sim.profile_zx()
                    sim.profile_zy()
                    # for z in zs:
                    sim.profile_xy(0.0)
                        # sim.profile_x(z)
                        # sim.profile_y(z)
                        # sim.profile_xy(z)
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

def run_ZXY_Noise(simulations_root: str, results_root: str):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZXY_Noise(filepath, simulations_root, results_root)
                    sim.on_axis_max_vs_z()
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

def run_ZXY_hermite(simulations_root: str, results_root: str, zs):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZXY_hermite(filepath, simulations_root, results_root)
                    # sim.on_axis_max_vs_z()
                    # sim.profile_zx()
                    # sim.profile_zy()
                    # for z in zs:
                    sim.profile_xy(0.0)
                        # sim.profile_x(z)
                        # sim.profile_y(z)
                        # sim.profile_xy(z)
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

def run_ZXT(simulations_root: str, results_root: str):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZXT(filepath, simulations_root, results_root)
                    sim.on_axis_max_vs_z()
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

def run_ZRT(simulations_root: str, results_root: str):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZRT(filepath, simulations_root, results_root)
                    sim.on_axis_max_vs_z()
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')


# gaussian
run_ZXY('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/no_noise/',
           '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/no_noise/',
           np.arange(0.0, 4.0, 0.079611))

# square
# run_ZXY('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/square/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/square/',
        #    np.arange(0.0, 4.0, 0.159222)) # 3 * 0.079611

# gaussian with noise
# run_ZXY_Noise('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/noise/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/noise/') 

# hermite
# run_ZXY_hermite('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/hermite_2_1/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/hermite_2_1/',
        #    np.arange(0.0, 4.0, 0.079611))

# XZT gaussian or square
# run_ZXT('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxt/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/')

# gaussian, plasma
# run_ZRT('/home/teofil/Desktop/Eldyn_sims/Simulations/FD/zrt/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FD/zrt/')


########## Some tests
# sim_test = BeamSimulationZXY_hermite('/home/teofil/Desktop/Eldyn_sims/test/Pin_014p0_Pcr_Hermite_4D_FFT_diagnostics.npz', '/home/teofil/Desktop/Eldyn_sims/test/', '/home/teofil/Desktop/Eldyn_sims/test/')
# sim_test.on_axis_max_vs_z()
# print(sim_test.snap_z)
# sim_test.profile_zx()
# sim_test.profile_zy()
# sim_test.profile_xy(1.5)
# for z in sim_test.snap_z:
    # sim_test.profile_xy(z)
# sim_test.profile_x(0.5)
# for z in sim_test.snap_z:
    # sim_test.profile_x(z)
# sim_test.spectrum_xy(0)
# print(sim_test.snap_z)
# print(np.size(sim_test.snap_z))




########## Manual plotting
def manual1(pin_factor, zs):
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

def manual2(pin_factor, zs):
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


manual1(pin_factor=1.2, zs=[0.239, 0.955, 1.194, 1.433])
# manual2(pin_factor=20.0, zs=[0.159, 0.318, 0.955])


# plot all files in all the dirs in a list of paths
'''
def plot_from_dirs(paths: list[str]):
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.npz'):
                sim = BeamSimulationXYZ(os.path.join(path, file))
                sim.on_axis_max_vs_z()
                sim.profile_xz()
                for z in np.arange(0.0, 4.0, 1.0): # default = 0.079611
                    sim.profile_x(z)
                    sim.profile_xy(z)
                    # sim.spectrum_xy(z)

plot_from_dirs([
    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Teofil/test_runs/'
])
#'''


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
# profile_yz(args)
# field_in_time(args, tmin=-200e-15, tmax=200e-15, Nt=1000, z=3.4)
'''