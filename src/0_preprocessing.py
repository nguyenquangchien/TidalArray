#!/usr/bin/env python
"""
Pre-processing script. This script:
- Interpolates bathymetry to an hdf5 file that can be imported later
- Adds viscosity sponges in boundary conditions
- Can be used to edit bathymetry/ friction and other fields that then feed into the simulations

In modifying the regions close to boundary the Eikonal equation is solved
using Firedrake (initial script from Roan, modified by Stephan and Than)
"""

from thetis import *
from firedrake.petsc import PETSc
from tools import field_tools, tidal_amplitude, bathymetry, boundary_distance
import inputs.simulation_parameters as inputs
from firedrake.output.vtk_output import VTKFile

import os
import utm
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

starttime = datetime.now()
CASEDIR = f"case_{os.environ['SIM_CASE']}"
if MPI.COMM_WORLD.rank == 0:
    print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))
    if not os.path.exists(f'{CASEDIR}'):
        os.makedirs(f'{CASEDIR}')
    if not os.path.exists(inputs.preproc_folder):
        os.makedirs(inputs.preproc_folder)
    if not os.path.exists(inputs.ramp_output_folder):
        os.makedirs(inputs.ramp_output_folder)
    if not os.path.exists(inputs.run_output_folder):
        os.makedirs(inputs.run_output_folder)
MPI.COMM_WORLD.barrier()  # stop until the pre-processing folder is generated

outputdir = f"{CASEDIR}/outputs"
inputdir = "inputs"
mesh = Mesh(inputs.mesh_file)

# Step 0 - Calculate lowest astronomical tide if bathymetry dataset is based on Lowest Astronomical Tide
V = FunctionSpace(mesh, 'CG', 1)
lat = Function(V)
tidal_amplitude.get_lowest_astronomical_tide(lat)
VTKFile(os.path.join(outputdir, 'lat.pvd')).write(lat)

# Step 1 - Calculate distance for viscosity
PETSc.Sys.Print("Calculate distance for viscosity")

# Boundary conditions
bcs = [DirichletBC(V, 0.0, inputs.open_bnd)]

## Code from PFClean
# # Step 1 - Calculate distance to open boundary for viscosity
# PETSc.Sys.Print("Calculate distance to open (ocean) boundary")  # af
# u_open = boundary_distance.calculate_distance_to_boundary(mesh, inputs.open_bnd, inputs.i_L, inputs.i_epss)
# VTKFile(outputdir + "/dist_open_bnd.pvd").write(u_open)

L = inputs.L
v = TestFunction(V)
u_open = field_tools.eik(V, inputs.open_bnd, 
                    outfilename=os.path.join(outputdir, "dist.pvd"))

# Adding viscosity sponge
h_viscosity = Function(V, name="viscosity")
h_viscosity.interpolate(max_value(inputs.visc_amb, 1000 * (1 - u_open / 2e4)))
VTKFile(os.path.join(outputdir, 'viscosity.pvd')).write(h_viscosity)


# Creating a Manning/Quadratic drag/other type friction field to be used in the simulations

if inputs.use_friction_data:
    manning_data = np.load(inputs.friction_data)
    interpolator = np.vectorize(interp1d(manning_data[0, :], manning_data[1, :],
                                         fill_value=(manning_data[1, 0], manning_data[1, -1]),
                                         bounds_error=False))
    manning_2d = bathymetry.get_manning_class(inputs.bed_classification_file, mesh, interpolator)
    manning_2d *= inputs.friction_multiplier
    bathymetry.smoothen_bathymetry(manning_2d)
else:
    manning_2d = Function(V, name='manning')
    manning_2d.interpolate(max_value(inputs.i_manning, 0.1 * (1. - u_open / 5e4)))

VTKFile(os.path.join(outputdir, 'manning.pvd')).write(manning_2d)
print_output('Exported manning')

# Interpolating bathymetry
bath = None
add_lat = False
for i, (f, source) in enumerate(inputs.bathymetries):
    if source in ['digimap', 'emod']:
        if i == 0:
            bath = bathymetry.get_bathymetry(f, mesh, source=source)
        else:
            bath = bathymetry.get_bathymetry_iteration(f, mesh, bath, source=source)
        add_lat = True
    else:
        if add_lat:
            bath.assign(bath + lat)
            add_lat = False
        if i == 0:
            bath = bathymetry.get_bathymetry(f, mesh, source=source)
        else:
            bath = bathymetry.get_bathymetry_iteration(f, mesh, bath, source=source)
    if add_lat:
        bath.assign(bath + lat)

## New chunk from PFClean - maybe this help solve the error run MPI?
# Set any remaining np.NaNs to the minimum value.
xvector = mesh.coordinates.dat.data
bvector = bath.dat.data
utm_zone = 30
utm_band = 'N'
for a, xy in enumerate(xvector):
    lat, lon = utm.to_latlon(xy[0], xy[1], utm_zone, utm_band)
    if np.isnan(bvector[a]):
        bvector[a] = 0


# Smoothing bathymetry
bathymetry.smoothen_bathymetry(bath)
# bathymetry.smoothen_bathymetry(bath)  # sometimes need to do this twice for better smoothing on new mesh

## OLD CODE
# def zbedf(bathy, distance):
#     """
#     Function used to edit bathymetry close to a boundary (e.g. when wetting and drying needs to be avoided)
#     :param bathy:  Bathymetry field
#     :param distance:  Distance from particular boundary determined by the Eikonal equation
#     :return: Modified bathymetry field
#     """
#     zbed = conditional(ge(bathy, 25 * (1. - distance / 10000.)), bathy, 25 * (1. - distance / 10000.))
#     return zbed


# # Applying bathymetry correction at the boundary
# bath.interpolate(max_value(zbedf(bath, u), inputs.min_depth))


## NEW CODE - PFClean
# Applying bathymetry correction at the boundary - closed boundary interpolation removes cliffs
if inputs.apply_closed_boundary_interp:
    PETSc.Sys.Print("Calculate distance to closed (land) boundary")  # af
    u_land = boundary_distance.calculate_distance_to_boundary(mesh, inputs.land_bnd, inputs.L, inputs.epss)
    VTKFile(outputdir + "/dist_land_bnd.pvd").write(u_land)

    # determine the gradient of the distance to land boundaries to identify narrow channels
    V = FunctionSpace(mesh, 'CG', 1)
    grad_u_land_func = boundary_distance.calculate_gradient_magnitude(u_land, V)
    VTKFile(outputdir + "/grad_dist_land_bnd.pvd").write(grad_u_land_func)

    bath.interpolate(max_value(
        boundary_distance.zbedf_narrow(bath, u_land, grad_u_land_func,
                                       dist_from_bnd=inputs.land_bnd_interp_dist,
                                       max_depth=inputs.max_land_bnd_depth,
                                       narrow_chnl_slope_threshold=inputs.narrow_chnl_slope_threshold,
                                       narrow_chnl_dist_from_bnd=inputs.narrow_chnl_dist_from_bnd,
                                       narrow_chnl_max_depth=inputs.narrow_chnl_max_depth),
        inputs.i_min_depth))
# open boundary interpolation increases depth near open boundary if we don't want wetting and drying
if inputs.apply_open_boundary_interp:
    bath.interpolate(max_value(
        boundary_distance.zbedf(bath, u_open, inputs.open_bnd_interp_dist, inputs.min_open_bnd_depth),
        inputs.i_min_depth))


VTKFile(os.path.join(outputdir, 'bath.pvd')).write(bath)

with CheckpointFile(os.path.join(inputs.preproc_folder, inputs.preproc_filename), "w") as CF:
    CF.save_mesh(mesh)
    CF.save_function(h_viscosity, name="viscosity")
    CF.save_function(bath, name="bathymetry")
    CF.save_function(manning_2d, name='manning')
    CF.close()

endtime = datetime.now()
simulationtime = endtime - starttime

if MPI.COMM_WORLD.rank == 0:
    print('End time:', endtime.strftime("%d/%m/%Y %H:%M:%S"))
    print('Simulation time =', simulationtime)
