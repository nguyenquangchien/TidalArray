# TidalArray
Simulating tidal stream turbines for an operational: real-world case of Pentland Firth, Scotland

## Installation

Prerequisite: Thetis and Firedrake must be installed on your system:

```bash
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install  
python3 firedrake-install --install thetis  
```

(or otherwise consult Firedrake's Github)


## Inputs

### General

* `PF0_assess.msh`: Computational mesh for the domain, Case PF0, for resource assessment
<!-- * `PF1_demo.msh`: Computational mesh for the domain, Case PF1, for operational demo
* `PF2_design.msh`: Computational mesh for the domain, Case PF2, for design purpose -->
* `bathy.txt`: Bathymetry data, interpolated on the mesh
* `lat.txt`: Lowest Astronomical Tide, used for adjusting bathymetry
* `extra_detectors.npy` and `extra_detectors_names.npy`: List of detector coordinates and names
* `n_max_125.npy`:  Distribution of Manning friction
<!-- * `speeds_AR1500.npy`, `thrusts_AR1500.npy`: Speeds and thrusts of the Simec Atlantis AR1500 tidal turbines -->
* `useful_gauges.csv`: Location and tidal constituent information for various gauges around the UK and from the Meygen site.
* `simulation_parameters.py`: Simulation parameters configuration, see below.

### Model Data

Most model data is available on request (or can be obtained online from various sources). This should include:

* Bed class file:  `bed_class_pentland_rev.nc`  
* Bathymetry files:  
    - `digimap_pentland.nc`: low resolution data covering all Scottish coasts, 
    - `orkney_digimap.nc`:  medium-resolution data covering the north-east of Scotland and the Orkneys and Shetland  
    - `Pentland_firth_Gebco.nc`: very low resolution data covering North UK and Scandinavia  

Forcing files:  `gridES2008.nc`  and  `hfES2008.nc`.

### Simulation Parameters

#### Input paths
1. Mesh: mesh to interpolate data to and calculate on.
2. Bathymetry: bathymetry files, in order from least to highest resolution. See `bathymetry.py` in `tools`.
3. Forcing: tidal signal forcing data.
4. Detectors: detector locations.
5. Bed morphology: bed morphology data file.
6. Friction: friction file.

#### Output paths
1. Output folders: choice of output folder location and name for the ramp and run.


#### Simulation parameters
1. UTM zone and band: 30V
2. Simulation start time: year, month, day, hour, minute
3. Detector:
4. Eikonal:
5. Boundary values: boundary tags on the mesh that represent the open boundaries (i.e. the North Sea) and the land boundaries.
6. Bathymetry: Minimum depth and name of the bathymetry parameter from the NetCDF file e.g. `'z'`, `'elevation'` etc.
7. Time: Ramp duration, timestep for the simulation, wetting and drying, Manning parameter, Coriolis parameter, export times and run simulation.
8. Physical parameters: gravitational acceleration and seawater density.


## Tools

A series of files to aid the preprocessing and model runs.

* `utm.py`:  Converts from lat/lon to UTM and vice versa.
* `bathymetry.py`:  Interpolation of the bathymetry.
* `detectors.py`:  Takes the expected tidal amplitude across the domain and outputs it for a transformation between datums.
* `thetis_support_scripts.py`:  Functions for use in pre-processing and running of the model e.g. Coriolis term inclusion, re-initialisation of flow/parameter fields and exporting states.
* `tidal_amplitude.py`:  Transforms data (e.g. bathymetry) to lowest astronical tide (/chart datum) data.
* `tidal_forcing.py`:  Sets the tidal field.
* `tidal_forcing_ramp.py`:  Sets the tidal field for the ramp.
* `power.py`:  Calculates the power output of the turbine, override Thetis' default module.


## Batch-run

This method offers maximum autonomous control during operational use. 

Information about the command is given by typing into the command prompt (terminal):

```
$ sim.py -h
usage: sim.py [-h] [-c CASE] [-s START] [-d DURATION] [-r RAMP_DURATION] [-p NUM_PROC]

Time information for tidal dynamic simulation

options:
  -h, --help            Show this help message and exit
  -s CASE, --case CASE
                        Name of the simulation case, used for output folder name
  -s START, --start START
                        Start date time (YYYY-MM-DDTHH:MM) of simulation
  -d DURATION, --duration DURATION
                        Duration length of simulation, number and units e.g. "30d", "120h", "14c" (here "c" is semi-diurnal cycle)
  -r RAMP_DURATION, --ramp-duration RAMP_DURATION
                        Duration length of ramp-up, number and units e.g. "2d", "24", "4c" (here "c" is semi-diurnal cycle)
  -p NUM_PROC, --num-proc NUM_PROC
                        Number of processes used in parallel simulation
```

An example where running a "base" case with a one-day long simulation, starting at 00:00 20 Oct 2002, preceded by a ramping-up time period of three hours, using 8 CPU cores:

```
$ python sim.py  -c BASE  -s 2002-10-20T00:00  -r 3h  -d 1d  -p 8
```

You may want to install the Python package `utm` beforehand:
```
$ pip install utm
```

## Output

Output produced in two forms collected in the folder `out_DT`: 

- time-series of elevation at a selected station (Wick, 58.44°N, 3.09°W or X = 497120 m, Y = 6477545 m in UTM 30V) (tab-separated text file `Tser.tsv`)
- field elevation of a small region between Stroma and Scotland mainland (longitude range 3.145°W to 3.135°W, latitude range 58.655°N to 58.660°N) (`field.nc`)

Live plot is produced during the run with `tools/simple_plot.py` which generates `tools/plot.html` and this can be viewed using Firefox browser if available.

## Dissecting running stages

### Parallel runs

Due to the size of the model and the nature of the problem, this model lends itself to parallel processing. This is typically achieved with the [Open MPI commands](https://www.open-mpi.org/). If running in parallel, it is important that all the scripts are run on the same number of cores.

> NOTE: It is recommended to use the script `sim.py` to run the script. Even if you were to adjust, arrange the processing steps, it is easier to do in a batch file like `sim.py` instead of typing the commands into the terminal.

### 0_preprocessing.py

CAUTION: Performing MPI parallel run at this stage is not recommended because while creating checkpoint (`hdf5`) files, parallel runs easily consume all system memory!

Before ramping up the model, we must process the model data from its initial form (e.g. a NetCDF (`.nc`) file) to a `.h5` file which Thetis uses. The bathymetry is interpolated to the mesh, viscosity ``sponges'' are added near the boundaries for model stability and the Manning friction fields are interpolated to the mesh.

Interpolates bathymetry to an hdf5 file that can be imported later
- Adds viscosity sponges in boundary conditions
- Can be used to edit bathymetry/ friction and other fields that then feed into the simulations

Generates:  
- `inputs/bathymetry2D.h5`: bathymetry data for running the model
- `inputs/manning2D.h5`: friction data for running the model
- `inputs/viscosity.h5`: viscosity data for running the model
- `outputs_CASE/outputs/bath.pvd` & corresponding `.pvtu` and `.vtu` files: bathymetry data to be viewed with ParaView
- `outputs_CASE/outputs/dist.pvd` & corresponding `.pvtu` and `.vtu` files:
- `outputs_CASE/outputs/lat.pvd` & corresponding `.pvtu` and `.vtu` files:
- `outputs_CASE/outputs/manning.pvd` & corresponding `.pvtu` and `.vtu` files: friction data to be viewed with ParaView
- `outputs_CASE/outputs/viscosity.pvd` & corresponding `.pvtu` and `.vtu` files: viscosity data to be viewed with ParaView

### 1_ramp.py

Before running the model, we need to ramp up the elevation and velocity fields up to the desired start point. Once running for a simulated day or two, the data files are saved for the main run.

When run, the simulation parameters should be output followed by the remaining time in the simulation.

Generates:  
- `outputs_CASE/outputs_ramp/Elevation2d`: elevation outputs  
- `outputs_CASE/outputs_ramp/init_bathymetry_2d`: bathymetry data to be viewed with ParaView  
- `outputs_CASE/outputs_ramp/Velocity2d`: velocity outputs  
- `outputs_CASE/outputs_ramp/bath.pvd` & corresponding `.pvtu` and `.vtu` files: bathymetry data to be viewed with ParaView  
- `outputs_CASE/outputs_ramp/diagnostic_volume2d.hdf5`  
- `outputs_CASE/outputs_ramp/manning.pvd` & corresponding `.pvtu` and `.vtu` files: friction data to be viewed with ParaView  
- `outputs_CASE/outputs_ramp/viscosity.pvd` & corresponding `.pvtu` and `.vtu` files: viscosity data to be viewed with ParaView  
- `outputs_CASE/elevationout.pvd` & corresponding `.pvtu` and `.vtu` files: final elevation data to be viewed with ParaView  
- `outputs_CASE/velocityout.pvd` & corresponding `.pvtu` and `.vtu` files: final velocity data to be viewed with ParaView

### 2_run.py

The main model can now be run. Unless altered, velocities and elevation fields will be outputted at the desired export times. For optimisation use, some `.h5` files will also be output at certain time periods.

Generates:  
- `outputs_CASE/outputs_run/Elevation2d`: elevation outputs  
- `outputs_CASE/outputs_run/hdf5`: hdf5 files for manipulation later  
- `outputs_CASE/outputs_run/init_bathymetry_2d`: bathymetry data to be viewed with ParaView  
- `outputs_CASE/outputs_run/Velocity2d`: velocity outputs  
- `outputs_CASE/outputs_run/bath.pvd` & corresponding `.pvtu` and `.vtu`files: bathymetry data to be viewed with ParaView  
- `outputs_CASE/outputs_run/diagnostic_detectors-adcp.hdf5`  
- `outputs_CASE/outputs_run/diagnostic_turbine.hdf5`  
- `outputs_CASE/outputs_run/diagnostic_volume2d.hdf5`  
- `outputs_CASE/outputs_run/elevation_imported.pvd` & corresponding `.pvtu` and `.vtu` files: imported elevation data to be viewed with ParaView  
- `outputs_CASE/outputs_run/manning.pvd` & corresponding `.pvtu` and `.vtu` files: friction data to be viewed with ParaView  
- `outputs_CASE/outputs_run/velocity_imported.pvd` & corresponding `.pvtu` and `.vtu` files: imported velocity data to be viewed with ParaView  
- `outputs_CASE/outputs_run/viscosity.pvd & corresponding` `.pvtu` and `.vtu` files: viscosity data to be viewed with ParaView.

### 3_postproc.py

Assuming that we finish simulation, and the results are stored in different folders according to various scenarios:

* Base case, named e.g. `outputs_BASECASE`
* Coarse-grid case, e.g. `outputs_COARSEGRID`
* Fine-grid case, e.g. `outputs_FINEGRID`
* No-turbine case, e.g. `outputs_NOTRB`
* Full-array case, e.g. `outputs_ARRAY`

Then rename the corresponding folders in the [code file](https://github.com/nguyenquangchien/TidalArray/blob/7fb01756158157f087cd34467658f2aea37e7c7b/src/3_postproc.py#L31). 

Running this script can be done in two steps:

* Step 1: to produce the intermediate flux files `raw_flux_output/sthuv_base_ext.npy` and `raw_flux_output/Q_all_ext.npy`
* Step 2: indicate the corresponding files in the [code file](https://github.com/nguyenquangchien/TidalArray/blob/7fb01756158157f087cd34467658f2aea37e7c7b/src/3_postproc.py#L558) and run to produce the plots similar to that presented in the present paper. 

### 4_plot_diff.py

The script is for plotting the difference in flow field and energy flux between the base (e.g. ambient) case and the modified (altered) case. The plots will be shown as filled contours.

As with the above script, be careful first to get the correct names for output (simulated) folders, and filled in the code statements: 

* [`compare_and_plot`](https://github.com/nguyenquangchien/TidalArray/blob/7fb01756158157f087cd34467658f2aea37e7c7b/src/4_plot_diff.py#L379): for plotting the difference in velocity.
* `compare_and_plot_flux`: for plotting the difference in energy flux.

The parameters for these functions/statements are:
* `levels`: for example `(-80, 80, 17)` will show difference ranging from -80% to +80% with 17 levels of contour lines.
* `folder`: name of folder to save the image files in.
* `fformat`: file format for such image files, e.g. `png` or `pdf` (for simple plots).
* `quantity` and `scenario`: to generate appropriate file names.
* `subfig`: annotates on the plots. Specifically when preparing paper manuscripts, to indicate the order within a figure.


## Validation

The validation scripts can be used to compare tracked velocities and elevations in the model to measured data. This includes plotting water level and velocity-time series, peak velocities and turbine powers. A number of predefined metrics to quantify the goodness-of-fit between the model and the data is established, including root-mean-square error (RMSE), mean absolute error (MAE), bias, and coefficient of determination (R^2^).

* `gof.py`: Performing godness-of-fit between a time series of water level (or generally, a scalar quantity) against observed data.
* `gof_vel.py`: Performing godness-of-fit between a time series of velocity (or generally, a vector quantity) against observed data. Also separates flood and ebb phases, analyses velocity peaks and histograms.

The scripts can be invoked after simulation stopped, e.g. after step 2 mentioned above.