#!/usr/bin/env python
""" Simulation parameters; avoid modifying file structure. """

import os
from pathlib import Path

mode = 'PF1A' # 'PF0N', 'PF0G', 'PF0A', 'PF1N', 'PF1G', 'PF1A', 'PF2N', 'PF2G', 'PF2A'
mode = mode.upper()

if mode in ['PF0N', 'PF0G']:
    mesh_file = "inputs/meshes/PF0G.msh"
elif mode == 'PF0A':
    mesh_file = "inputs/meshes/PF0A.msh"
elif mode in ['PF1N', 'PF1G']:
    mesh_file = "inputs/meshes/PF1G.msh"
elif mode == 'PF1A':
    mesh_file = "inputs/meshes/PF1A.msh"
elif mode[:3] == 'PF2':
    mesh_file = "inputs/meshes/PF2_JHR.msh"
else:
    raise ValueError(f'Unknown mode: {mode}')


model_data_dir = f'{Path.home()}/Documents/model_data'
model_data_dir = f'../model_data'  # for Docker runs

# Bathyfile(s): order by resolution (highest -> lowest)
# list of (file_path, data_source)
bathymetries = [ # (f"{model_data_dir}/Meygen_bathymetry_WGS84.nc", 'emod'),
                (f"{model_data_dir}/bathymetry/orkney_digimap.nc", 'digimap'),
                (f"{model_data_dir}/bathymetry/digimap_pentland.nc", 'digimap'),
                (f"{model_data_dir}/bathymetry/Pentland_firth_Gebco.nc", 'Gebco')]

## New in PFClean
# Minimum depth in bathymetry
i_min_depth = -5.0     # a minimum depth that is imposed for the bathymetry
# Interpolations to alter boundary values
apply_closed_boundary_interp = True  # to set maximum depths at the land boundaries to avoid cliffs
land_bnd_interp_dist = 75.  # distance over which to overwrite depth if needed
max_land_bnd_depth = 2.  # maximum depth at the land boundary
narrow_chnl_slope_threshold = 0.5  # gradient threshold to identify narrow channels which need less modification
narrow_chnl_dist_from_bnd = 30.  # distance over which to overwrite depth in narrow channels
narrow_chnl_max_depth = 5.  # maximum depth at narrow channel land boundaries
apply_open_boundary_interp = False
open_bnd_interp_dist = 10000.  # Distance from open boundary at which min. depth = 0
min_open_bnd_depth = 25.  # Min. dpeth at boundary if open boundary interpolation is applied

# Forcing
grid_forcing_file = f'{model_data_dir}/forcing/gridES2008.nc'
hf_forcing_file = f'{model_data_dir}/forcing/hfES2008.nc'
range_forcing_coords = ((-8., 0.5), (56., 61))
constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']

# Detectors
tidegauge_file = 'inputs/detectors/useful_gauges.csv'
elevation_detectors = []
additional_detector_files = ['inputs/detectors/extra_detectors']
transects = {
    'flood_inlet':  ([489348.21, 6501814.71], [492411.84, 6504011.77], 1000),
    'flood_centre': ([491478.98, 6499808.73], [492653.66, 6503338.39], 1000),
    'flood_outlet': ([494234.05, 6500721.97], [493785.47, 6503399.66],  700),
    'ebb_inlet':  ([493441.01, 6503286.84], [494556.89, 6500735.17], 800),
    'ebb_centre': ([492033.22, 6502959.], [492330.8, 6500069.28], 900),
    'ebb_outlet': ([491842.06, 6503213.44], [489593.67, 6501262.28], 700),
}

# Friction
use_friction_data = True
i_manning = 0.03  # Uniform Manning USED IF use_friction_data = False
bed_classification_file = f'{model_data_dir}/friction/PF_sediment_ID.nc'  # regions of friction (sedimentological)
friction_data = f'{model_data_dir}/friction/dmax_mannings_coeffs.npy'  # corresponding Manning value mapping
friction_multiplier = 1.0

# Viscosity
visc_amb = 5.

# Outputs folders
CASEDIR = f'case_{os.environ["SIM_CASE"]}'
preproc_folder = f'{CASEDIR}/preproc'
preproc_filename = 'preprocessing.h5'
ramp_output_folder = f'{preproc_folder}/ramp'
ramp_filename = 'ramp.h5'
run_output_folder = f'{CASEDIR}/outputs'
output_DT_folder = f'{CASEDIR}/out_DT'

## UTM parameters - geographical
zone = 30  # 30N = northern hemisphere
band = 'V'  # V = North Sea

## Export region (netCDF) specification
ext_lon = 0.01
ext_lat = 0.005
n_sample_lon = 20
n_sample_lat = 30
lat_origin = 58.655
lon_origin = -3.145

## Simulation start time parameters (for TPXO)
s_year = 2017
s_month = 8
s_day = 9
s_hour = 0
s_min = 0

## Detector parameters
max_dist = 5e3  # maximum distance from detectors

## Eikonal and preprocessing stage parameters 
L = 1e3  # characteristic length for eikonal eq.
epss = [100000., 10000., 5000., 2500., 1500., 1000.]  # accuracy [m] of the final "distance to boundary" func.

## Boundary values (from QGIS)
open_bnd = 1
land_bnd = 1000

## Bathymetry parameters
min_depth = -5.0     # minimum depth in bathymetry
alt_name = 'z'  # name of bathymetry param in the nc file

## Ramp and run parameters
ramptime = 2 * 24 * 3600      # Ramptime in seconds
dt = 100                      # Crank-Nicolson timestep in seconds
alpha = 1.5                   # Wetting and drying parameter
manning = 0.03                # Manning parameter
lat_cor = 59                  # Coriolis calculation parameters, latitude in degrees

opt_spatial_harmonics_distribution = False

# Export specs (times are in second)
ramp_exp_interval = 86400.          # Ramp output interval field variables
run_exp_interval = 86400.           # Run output interval field variables
int_expt_ff = 1200.                 # Export flow field to aggregate during ebb/flood
power_export_interval = 2 * 3600    # Power output interval
t_end = 60 * 24 * 3600              # Simulation duration in seconds

# Indices of ebb/flood periods corresponding to int_expt_ff
# Requirement: ebb and flood are strictly distinct periods
# Flood can occur either after or before ebb.
# Here each period spans 18 intervals or 6 hours.
idx_start_ebb = 1
idx_end_ebb = 18
idx_start_fld = 19
idx_end_fld = 38

## Turbine specification
if mode.endswith('N'):
    opt_include_turbines = False
else:
    opt_include_turbines = True
    site_ID = 2
    turbine_diameter = 18
    thrust_coefficients_file = 'turbine_data/AR1500/thrusts_AR1500.npy'
    thrust_speeds_file = 'turbine_data/AR1500/speeds_AR1500.npy'
    include_support_structure = True
    support_structure_thrust_coefficient = 0.7  # support structure drag coefficient
    support_structure_area = 2.6 * 14.0  # cross-sectional area of support structure - diameter x height
    turbine_coordinates = [(491818.59, 6502208.52), 
                        (491776.28, 6502123.85), 
                        (492027.64, 6502159.18), 
                        (492037.85, 6502002.53)]
    turbine_coordinates_file = 'turbine_data/JHR2024_gridded_20m_test_case.npy'
    
    if mode == 'PF0G':
        opt_average_density = True   # perform average density over ROI marked with site_ID
        use_turbine_coordinates_file = False
    elif mode == 'PF0A':
        opt_average_density = True   # perform average density over ROI marked with site_ID
        use_turbine_coordinates_file = True
    elif mode in ['PF1G', 'PF2G']:
        opt_average_density = False  # bump density function formulation
        use_turbine_coordinates_file = False
    elif mode in ['PF1A', 'PF2A']:
        opt_average_density = False  # bump density function formulation
        use_turbine_coordinates_file = True
    else:
        raise ValueError(f'Unknown mode: {mode}')
