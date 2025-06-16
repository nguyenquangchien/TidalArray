import os
import sys
import warnings
import numpy as np
import datetime

from thetis import *

from firedrake.petsc import PETSc
from firedrake.output.vtk_output import VTKFile

import tools.export_hidromod
import tools.detectors
import tools.tidal_forcing
import tools.thetis_support_scripts
from tools.power import TidalPowerCallback
from tools.transect import generate_transect
import inputs.simulation_parameters as inputs

sys.path.append('../')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

starttime = datetime.datetime.now()
if MPI.COMM_WORLD.rank == 0:
    print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))

inputdir = 'inputs'
outputdir = inputs.run_output_folder
CASEDIR = f"case_{os.environ['SIM_CASE']}"
out_DT_folder = inputs.output_DT_folder
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(out_DT_folder):
        os.makedirs(out_DT_folder)

path_to_ramp_export = os.path.join(inputs.ramp_output_folder, inputs.ramp_filename)
with CheckpointFile(path_to_ramp_export, 'r') as CF:
    mesh2d = CF.load_mesh()
    bathymetry_2d = CF.load_function(mesh2d, name="bathymetry")
    h_viscosity = CF.load_function(mesh2d, name="viscosity")
    mu_manning = CF.load_function(mesh2d, name='manning')
    uv_init = CF.load_function(mesh2d, name="velocity")
    elev_init = CF.load_function(mesh2d, name="elevation")

PETSc.Sys.Print(f'Loaded mesh {mesh2d.name}')
PETSc.Sys.Print(f'Exporting to {outputdir}')

identifier = 0  # simulation ID
PETSc.Sys.Print('Simulation identifier: ' + str(identifier))

if 'SIM_DURATION' in os.environ:
    # Override the ramp time read from input file
    st = os.environ['SIM_DURATION']
    # parse the number and unit (last char.)
    number = float(st[:-1])
    unit = st[-1]
    if unit.isdigit():  # default unit is day
        unit = 'd'
        number = float(st)
    elif unit == 'h':
        t_end = number * 3600
    elif unit == 'd':
        t_end = number * 24 * 3600
    elif unit == 'c':
        t_end = number * 12.42 * 3600

t_start = identifier * t_end  # = 0, Simulation start time relative to tidal_forcing
t_export = inputs.run_exp_interval   # Export time if necessary
wd_alpha = inputs.alpha              # Wetting and drying
lat_coriolis = inputs.lat_cor        # Coriolis calculation parameters


P1_2d = FunctionSpace(mesh2d, 'CG', 1)  # Continuous Galerkin 1st order
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, lat_coriolis)

if inputs.opt_include_turbines:
    site_ID = 2  # mesh PhysID for subdomain where turbines are to be sited
    turbine_density = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_density').assign(0.0)
    farm_options = DiscreteTidalTurbineFarmOptions()
    farm_options.turbine_type = 'table'
    farm_options.turbine_options.thrust_coefficients = np.load(f'inputs/{inputs.thrust_coefficients_file}').tolist()
    farm_options.turbine_options.thrust_speeds = np.load(f'inputs/{inputs.thrust_speeds_file}').tolist()  # [m/s]
    if inputs.include_support_structure:
        farm_options.turbine_options.C_support = inputs.support_structure_thrust_coefficient
        farm_options.turbine_options.A_support = inputs.support_structure_area
    farm_options.turbine_options.diameter = inputs.turbine_diameter  # in metres

    farm_options.upwind_correction = True
    farm_options.turbine_density = turbine_density

    # Even with average turbine density!
    if inputs.use_turbine_coordinates_file:
        farm_options.turbine_coordinates = \
            [[Constant(x), Constant(y)]
            for x, y in np.load(f'inputs/{inputs.turbine_coordinates_file}')]
    else:
        farm_options.turbine_coordinates = \
                [[Constant(x), Constant(y)] for x, y in inputs.turbine_coordinates]

with timed_stage('initialisation'):
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.cfl_2d = 1.0
    options.use_nonlinear_equations = True
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.coriolis_frequency = coriolis_2d
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['elev_2d', 'uv_2d']
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
    options.element_family = "dg-dg"
    options.swe_timestepper_type = 'CrankNicolson'
    options.swe_timestepper_options.implicitness_theta = 1.0
    options.swe_timestepper_options.use_semi_implicit_linearization = True
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = Constant(wd_alpha)
    options.manning_drag_coefficient = mu_manning
    options.horizontal_viscosity = h_viscosity
    options.use_grad_div_viscosity_term = True
    options.use_grad_depth_viscosity_term = False
    options.timestep = inputs.dt  # override dt for CrankNicolson (semi-implicit)
    if inputs.opt_include_turbines:
        options.discrete_tidal_turbine_farms[site_ID] = [farm_options]
    options.swe_timestepper_options.solver_parameters = {
        'snes_type': 'newtonls',
        'snes_rtol': 1e-3,
        'snes_linesearch_type': 'bt',
        'snes_max_it': 20,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
    }

# Boundary conditions
tidal_elev = Function(bathymetry_2d.function_space())
solver_obj.bnd_functions['shallow_water'] = {
    inputs.open_bnd: {'elev': tidal_elev},
    inputs.land_bnd: {'un': 0.0}
}

extra_detectors = list(np.load(inputs.additional_detector_files[0] + '.npy'))
extra_detector_names = list(np.load(inputs.additional_detector_files[0] + '_names.npy'))

# Simulation preliminaries
solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)
det_xy, det_names = tools.detectors.get_detectors(mesh2d, maximum_distance=inputs.max_dist)

# Adding detectors callback for monitor points
cb = DetectorsCallback(solver_obj, det_xy, ['elev_2d', 'uv_2d'],
                       name='detectors-adcp', detector_names=det_names)
solver_obj.add_callback(cb, 'timestep')


# Transect detectors
for transect_location in inputs.transects:
    [Xs, Ys], [Xe, Ye], Npts = inputs.transects[transect_location]
    detectors_coords = generate_transect([Xs, Ys], [Xe, Ye], Npts,
                                        save_csv=False, save_npy=True, 
                                        filename=f'transect_{transect_location}')
    cb_transect = DetectorsCallback(solver_obj, detectors_coords, ['elev_2d', 'uv_2d'],
                                    name=f'detectors_{transect_location}')
    solver_obj.add_callback(cb_transect, 'timestep')


if inputs.opt_include_turbines:
    # Examining turbine density distribution
    dens = solver_obj.tidal_farms[0].turbine_density
    field = Function(P1_2d)
    field.project(dens)
    VTKFile(os.path.join(outputdir, 'dens.pvd')).write(field)    

    turbine_names = [f'T{i}' for i, _ in enumerate(farm_options.turbine_coordinates)]
    if MPI.COMM_WORLD.rank == 0:
        print(farm_options)

    PETSc.Sys.Print("Customized power callback")
    cb_turbines = turbines.TurbineFunctionalCallback(solver_obj)  # farm power, built-in callback
    cb_pow = TidalPowerCallback(solver_obj, site_ID,
                                    farm_options, ['pow'], 
                                    cb_name='array', turbine_names=turbine_names)
    #~ cb_uv_turb = TidalPowerCallback(solver_obj, site_ID,
                                    #~ farm_options, ['uv_2d'], 
                                    #~ cb_name='array', turbine_names=turbine_names)
    solver_obj.add_callback(cb_turbines, 'timestep')
    solver_obj.add_callback(cb_pow, 'timestep')

uv, elev = solver_obj.timestepper.solution.subfunctions
powers = []
uv_turb = []

# Vorticity calculation
vorticity = Function(P1v_2d, name="vorticity").assign(0)
vorticity_calc = thetis.diagnostics.VorticityCalculator2D(uv, vorticity)
solver_obj.solve()

if MPI.COMM_WORLD.rank == 0:
    print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))
    #~ start = datetime.datetime(inputs.s_year, inputs.s_month, inputs.s_day, inputs.s_hour, inputs.s_min)
    #~ tools.export.init(cb.detector_names, start, inputs.dt)
    start = datetime.fromisoformat(os.environ['SIM_START'])
    tools.export_hidromod.init(cb.detector_names, start, dt,'Wl_Tser')
    tools.export_hidromod.init(cb.detector_names, start, dt,'Cv_Tser')
    tools.export_hidromod.init(cb.detector_names, start, dt,'Pow_Tser')
    tools.export_hidromod.init(cb.detector_names, start, dt,'DirCv_Tser')


def intermediate_steps(t):
    """
    Export hdf5 fields for further processing.
    :param t: time
    :return:
    """
    # Export final state that can be picked up later aka. hot-start
    if t >= t_end:
        path_to_run_export = os.path.join(inputs.run_output_folder, "end_of_run_export.h5")
        with CheckpointFile(path_to_run_export, 'w') as f:
            f.save_mesh(mesh2d)
            f.save_function(bathymetry_2d, name="bathymetry")
            f.save_function(h_viscosity, name="viscosity")
            f.save_function(mu_manning, name="manning")
            f.save_function(uv, name="velocity")
            f.save_function(elev, name="elevation")
            f.save_function(vorticity, name="vorticity")
            f.close()

        VTKFile(f'{CASEDIR}/velocity_endofrun.pvd').write(uv)
        VTKFile(f'{CASEDIR}/elevation_endofrun.pvd').write(elev)

    # Export turbine power series regularly - monitoring purposes
    if t % inputs.power_export_interval == 0:
        try:
            arr_powers = np.asarray(powers)
            n_times, n_turbines, _ = arr_powers.shape
            flat_arr_powers = arr_powers.reshape(n_times, n_turbines)
            np.savetxt(os.path.join(out_DT_folder, 'powers.csv'), flat_arr_powers, delimiter=',')
            flat_arr_powers.tofile(os.path.join(out_DT_folder, 'powers.dat'))  # binary file
        except:
            print('Could not export turbine power series.')
            print('Option include turbines:', inputs.opt_include_turbines)
            print('Shape of the array data:', arr_powers.shape)


def update_forcings(t):
    """ Update tidal elevation at the model boundaries
        :param t: time
        :return:
    """
    with timed_stage('update forcings'):
        global time_layer
        intermediate_steps(t)
        PETSc.Sys.Print(f"t = {t_start + t}")
        e_vector = tools.tidal_forcing.set_tidal_field(tidal_elev, t + int(t_start))
        new_elev_bnd_2d = Function(tidal_elev.function_space(), name='bnd elevation')
        new_elev_bnd_2d.dat.data_with_halos[:] = e_vector.dat.data_with_halos
        tidal_elev.assign(new_elev_bnd_2d)
        tidal_elev.assign(e_vector)
        if MPI.COMM_WORLD.rank == 0:
            timestamp = (start + timedelta(seconds=t)).isoformat()
            tools.export_hidromod.export_TS(timestamp, cb.detector_locations, elev,'Wl_Tser')
            tools.export_hidromod.export_CVTS(timestamp, cb.detector_locations, uv,'Cv_Tser')
            tools.export_hidromod.export_DIRTS(timestamp, cb.detector_locations, uv,'DirCv_Tser')
            tools.export_hidromod.export_field(t, dt, elev)
            # tools.export.export_TS(t, cb.detector_locations, elev)
            # tools.export.export_field(t, inputs.dt, elev)
        if inputs.opt_include_turbines and inputs.mode not in ['PF0-G', 'PF0-A']:
            #~ power_from_callback = cb_turbines()
            #~ uv_from_callback = cb_uv_turb()
            #~ powers.append(power_from_callback)
            #~ uv_turb.append(uv_from_callback)
            powers.append(cb_turbines.integrated_power[0] / 100 - sum(powers))
            powers_out = cb_pow().transpose()
            tools.export_hidromod.export_pow_TS(timestamp, list(np.round(powers_out[0], 4)), 'Pow_Tser')


# Simulation iteration
solver_obj.iterate(update_forcings=update_forcings)

if inputs.opt_include_turbines:
    np.save(os.path.join(outputdir, 'farm_power.npy'), np.array(powers))

endtime = datetime.datetime.now()
simulationtime = endtime - starttime

if MPI.COMM_WORLD.rank == 0:
    print('End time:', endtime.strftime("%d/%m/%Y %H:%M:%S"))
    print('Simulation time =', simulationtime)


exit(0)

# Flatten the list 
try:
    arr_powers = np.asarray(powers)
    n_times, n_turbines, _ = arr_powers.shape
    flat_arr_powers = arr_powers.reshape(n_times, n_turbines)
    np.savetxt(os.path.join(out_DT_folder, 'powers.csv'), flat_arr_powers, delimiter=',')
    flat_arr_powers.tofile(os.path.join(out_DT_folder, 'powers.dat'))  # binary file
    arr_uv = np.asarray(uv_turb)
    n_times, n_turbines, _ = arr_uv.shape
    flat_arr_uv = arr_uv.reshape(n_times, n_turbines)
    np.savetxt(os.path.join(out_DT_folder, 'uv.csv'), flat_arr_uv, delimiter=',')
    flat_arr_uv.tofile(os.path.join(out_DT_folder, 'uv.dat'))  # binary file

except:
    print('Could not export turbine power series.')
    print('Option include turbines:', inputs.opt_include_turbines)
    print('Shape of the array data:', arr_powers.shape)
finally:
    if MPI.COMM_WORLD.rank == 0:
        print('End time:', endtime.strftime("%d/%m/%Y %H:%M:%S"))
        print('Simulation time =', simulationtime)
