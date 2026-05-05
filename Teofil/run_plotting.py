from plotting import *

########## Run the code on real data
# run the plots for all subdirs of the Simulations directory, saving automatically to the Results directory
def run_ZXY(simulations_root: str, results_root: str, zs):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZXY(filepath, simulations_root, results_root)
                    sim.on_axis_max_vs_z()
                    sim.profile_zx()
                    sim.profile_zy()
                    for z in zs:
                        sim.profile_x(z)
                        sim.profile_y(z)
                        sim.profile_xy(z)
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

# run_ZXY('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/noise/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/noise/',
        #    np.arange(0.0, 4.0, 0.079611))

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

# run_ZXY_Noise('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/gaussian/noise/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxy/gaussian/noise/') 

def run_ZXY_hermite(simulations_root: str, results_root: str, zs):
    for dirpath, _, files in os.walk(simulations_root):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(dirpath, file)
                print(f'Processing {filepath}...')
                try:
                    sim = BeamSimulationZXY_hermite(filepath, simulations_root, results_root)
                    sim.on_axis_max_vs_z()
                    sim.profile_zx()
                    sim.profile_zy()
                    for z in zs:
                        sim.profile_x(z)
                        sim.profile_y(z)
                        sim.profile_xy(z)
                    print(f'Done with {filepath}')
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

# run_ZXY_hermite('/home/teofil/Desktop/Eldyn_sims/Simulations/FFT/zxy/hermite_2_1/',
        #    '/media/teofil/Data/Teofil/Ecole/_S04/ELDYN/Project/Our_project/Code/multiple_laser_filamentation/Results/FFT/zxt/hermite_2_1/',
        #    np.arange(0.0, 4.0, 0.079611))

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