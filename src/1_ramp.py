from thetis import (Function, FunctionSpace, Constant,
                    CheckpointFile, solver2d, timed_stage, 
                    MPI, as_vector)
from datetime import datetime
from firedrake.petsc import PETSc
from firedrake.output.vtk_output import VTKFile

import tools.tidal_forcing
import tools.thetis_support_scripts
import inputs.simulation_parameters as inputs

import os
import sys
import warnings
from datetime import datetime

sys.path.append('../')
warnings.simplefilter(action="ignore", category=DeprecationWarning)

starttime = datetime.now()
if MPI.COMM_WORLD.rank == 0:
    print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))

CASEDIR = f"case_{os.environ['SIM_CASE']}"
inputdir = 'inputs'
outputdir = inputs.ramp_output_folder
datadir = 'data'

path_to_preproc_file = os.path.join(inputs.preproc_folder, inputs.preproc_filename)
with CheckpointFile(path_to_preproc_file, 'r') as CF:
    mesh2d = CF.load_mesh()
    bathymetry_2d = CF.load_function(mesh2d, name="bathymetry")
    h_viscosity = CF.load_function(mesh2d, name="viscosity")
    mu_manning = CF.load_function(mesh2d, name='manning')
    CF.close()

PETSc.Sys.Print(f'Loaded mesh {mesh2d.name}')
PETSc.Sys.Print(f'Exporting to {outputdir}')

identifier = -1  # simulation ID
PETSc.Sys.Print(f'Simulation identifier : {identifier}')

if 'RAMP_DURATION' in os.environ:
    # Override the ramp time read from input file
    st = os.environ['RAMP_DURATION']
    # parse the number and unit (last char.)
    number = float(st[:-1])
    unit = st[-1]
    if unit.isdigit():  # default unit is day
        unit = 'd'
        number = float(st)

    if unit == 'h':
        ramptime = number * 3600
    elif unit == 'd':
        ramptime = number * 24 * 3600
    elif unit == 'c':
        ramptime = number * 12.42 * 3600

t_start = - ramptime               # Simulation start time relative to tidal_forcing
t_end = 0                          # ramptime + t_start
CG_2d = FunctionSpace(mesh2d, 'CG', 1)
coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, inputs.lat_cor)

with timed_stage('initialisation'):
    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.cfl_2d = 1.0
    options.use_nonlinear_equations = True
    options.simulation_export_time = inputs.ramp_exp_interval
    options.simulation_end_time = ramptime
    options.coriolis_frequency = coriolis_2d
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.fields_to_export_hdf5 = []
    options.element_family = "dg-dg"
    options.swe_timestepper_type = 'CrankNicolson'
    options.swe_timestepper_options.implicitness_theta = 1.0
    options.swe_timestepper_options.use_semi_implicit_linearization = True
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = Constant(inputs.alpha)
    options.manning_drag_coefficient = mu_manning
    options.horizontal_viscosity = h_viscosity
    options.use_grad_div_viscosity_term = True
    options.use_grad_depth_viscosity_term = False
    options.timestep = inputs.dt  # override dt for CrankNicolson (semi-implicit)
    options.swe_timestepper_options.solver_parameters = {
        'snes_type': 'newtonls',
        'snes_rtol': 1e-3,
        'snes_linesearch_type': 'bt',
        'snes_max_it': 20,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
    }

tidal_elev = Function(bathymetry_2d.function_space())
solver_obj.bnd_functions['shallow_water'] = {
    inputs.open_bnd: {'elev': tidal_elev}}

elev_init = Function(CG_2d)
elev_init.assign(0.0)

solver_obj.assign_initial_conditions(uv=as_vector((1e-3, 0.0)), elev=elev_init)

uv, elev = solver_obj.timestepper.solution.subfunctions


def intermediate_steps(t):
    if inputs.opt_spatial_harmonics_distribution and t % inputs.ramp_exp_interval == 0:
        # Exporting to data file - useful for quick sampling etc.
        PETSc.Sys.Print("Exporting elevation field for harmonic analysis")
        elev_CG = Function(CG_2d, name='elev_CG').project(elev)
        with CheckpointFile(os.path.join(outputdir, f'elev_{t}')) as f:
            f.store(elev_CG)
            f.close()

    if t == t_end:  # export final state (hot-start)
        path_to_ramp_file = os.path.join(inputs.ramp_output_folder, inputs.ramp_filename)
        with CheckpointFile(path_to_ramp_file, 'w') as f:
            f.save_mesh(mesh2d)
            f.save_function(bathymetry_2d, name="bathymetry")
            f.save_function(h_viscosity, name="viscosity")
            f.save_function(mu_manning, name="manning")
            f.save_function(uv, name="velocity")
            f.save_function(elev, name="elevation")
            f.close()

        VTKFile(f'{CASEDIR}/velocityout.pvd').write(uv)
        VTKFile(f'{CASEDIR}/elevationout.pvd').write(elev)


def update_forcings(t):
    intermediate_steps(float(t + t_start))
    PETSc.Sys.Print(f"Updating tidal field at t={t_start + t}")
    # tools.tidal_forcing_ramp.set_tidal_field(tidal_elev, t + int(t_start), t_start)
    e_vector = tools.tidal_forcing.set_tidal_field(tidal_elev, t - inputs.ramptime, ramp_duration=inputs.ramptime)
    new_elev_bnd_2d = Function(tidal_elev.function_space(), name='bnd elevation')
    new_elev_bnd_2d.dat.data_with_halos[:] = e_vector.dat.data_with_halos
    tidal_elev.assign(new_elev_bnd_2d)


solver_obj.iterate(update_forcings=update_forcings)

endtime = datetime.now()
simulationtime = endtime - starttime

if MPI.COMM_WORLD.rank == 0:
    print('End time:', endtime.strftime("%d/%m/%Y %H:%M:%S"))
    print('Simulation time =', simulationtime)
