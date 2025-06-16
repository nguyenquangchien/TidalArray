""" Calculate discharge through a transect. """

import os
import h5py
import numpy as np

from datetime import datetime as date_time
from datetime import timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go

os.environ['SIM_CASE'] = 'PF1_G'  # old cases are OK; just need to import the transects
from inputs.simulation_parameters import transects, dt

from thetis import *

WORK_DIR = 1
base_path = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src')
base_file = 'outputs/diagnostic_detectors_flood_centre.hdf5'
path_hdf5_PF1 = os.path.join(base_path, 'case_PF1-G_visc_5', base_file)  # case_PF1-G_nM_10  case_PF1-G_visc_5
path_hdf5_PF0 = os.path.join(base_path, 'case_PF0-G_visc_5', base_file)  # case_PF0-G_nM_10  case_PF0-G_visc_5
path_hdf5_PF2 = os.path.join(base_path, 'case_PF2-G_nM_11', base_file)
path_hdf5_NNN = os.path.join(base_path, 'case_PF1-N_mann1.0', base_file)
path_hdf5_AAA = os.path.join(base_path, 'case_PF1-A_nM_10_hotstart', base_file)

# Collect the paths for all cases, the first one is the ``base case''
all_paths = (path_hdf5_PF1, 
            path_hdf5_PF0, path_hdf5_PF2,
            path_hdf5_NNN, path_hdf5_AAA,
            )
N_CASES = len(all_paths)

T_START = date_time(2017, 8, 1, 0, 0, 0)  # simulation start time
T_PLOT_BEGIN = date_time(2017, 8, 3)
T_PLOT_END = date_time(2017, 8, 10)  # limit plotting -- not necessary simulation end time
times_plot_transect = [
    date_time(2017, 8, 11, 3, 30),   # Spring tide 1st, Ebb -- out_idx[438]
    date_time(2017, 8, 11, 9, 30),   # Spring tide 1st, Fld -- out_idx[449]
    date_time(2017, 8, 16,21, 20),   # Neap tide, Ebb -- out_idx[686]
    date_time(2017, 8, 17, 3, 20),   # Neap tide, Fld -- out_idx[697]
    date_time(2017, 8, 23, 1, 15),   # Spring tide 2nd, Ebb -- out_idx[952]
    date_time(2017, 8, 23, 8, 10),   # Spring tide 2nd, Fld -- out_idx[965]
]  # selected ebb and flood phases
dtsim = 100  # simulation time step
step_output = 20  # skipping data points to run faster. You can set > 1 here for STEP 1, but in STEP 2 you must set set_output=1.

Q_MIN = 500  # lower threshold for calculating difference in flow discharge
DIFF_CUTOFF = 50  # max. percentage diff. considered
WIN_SIZE = 108  # moving average window for difference estimation
T_CYCLE = 12*3600 + 25*60  # time of one semi-diurnal cycle [in sec]
N_DAYS = 24  # 24  # run duration 
MAX_TLEN = int(N_DAYS * 86400 / dt)
# Bathymetry along the transect
path_to_ramp_file = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF2-G_nM_11/preproc/ramp/ramp.h5')
with CheckpointFile(path_to_ramp_file, 'r') as CF:
    mesh2d = CF.load_mesh()
    bathy = CF.load_function(mesh2d, name='bathymetry')


(xs, ys), (xe, ye), npts = transects['flood_centre'] # ([491478.98, 6499808.73], [492653.66, 6503338.39], 1000)  # from inputs.simulation.transects
translen = np.sqrt((xs-xe)**2 + (ys-ye)**2)
nvec = ((ye-ys)/translen, (xs-xe)/translen)
delta_L = translen/npts
ds = [i * delta_L for i in range(npts)]

transectA_coords = np.load('inputs/transect_flood_inlet.npy')
transectB_coords = np.load('inputs/transect_flood_centre.npy')
transectC_coords = np.load('inputs/transect_flood_outlet.npy')

profile_B = np.zeros(npts)
for i, (x, y) in enumerate(transectB_coords):
    profile_B[i] = bathy.at(x, y)

# For I/O Energy Fluxes
(xsA, ysA), (xeA, yeA), nptsA = transects['flood_inlet'] #    'flood_inlet':  ([489348.21, 6501814.71], [492411.84, 6504011.77], 1000),
(xsC, ysC), (xeC, yeC), nptsC = transects['flood_outlet'] #    'flood_outlet': ([494234.05, 6500721.97], [493785.47, 6503399.66],  700),

translenA = np.sqrt((xsA-xeA)**2 + (ysA-yeA)**2)
translenC = np.sqrt((xsC-xeC)**2 + (ysC-yeC)**2)
nvecA = ((yeA-ysA)/translenA, (xsA-xeA)/translenA)
nvecC = ((yeC-ysC)/translenC, (xsC-xeC)/translenC)
deltaL_A = translenA/npts
deltaL_C = translenC/npts
dsA = [i * deltaL_A for i in range(nptsA)]
dsC = [i * deltaL_C for i in range(nptsC)]

profile_A = np.zeros(nptsA)
for i, (x, y) in enumerate(transectA_coords):
    profile_A[i] = bathy.at(x, y)

profile_C = np.zeros(nptsA)
for i, (x, y) in enumerate(transectC_coords):
    profile_C[i] = bathy.at(x, y)


with h5py.File(path_hdf5_PF1, 'r') as f:
    """ Read time series
        So that we can create an array to store the results,
        whose size must be known before iterating over time.
    """
    keys = list(f.keys())
    assert keys[-1] == 'time', (
            f'ERROR reading file {path_hdf5_PF1}, time column not detected!')
    times = np.array(f[keys[-1]])
    ntimes = len(times)



time_idx_out = [
    (7879, "Spring, Ebb"),
    (8099, "Spring, Fld"),
    (13721, "Neap, Ebb"),
    (13941, "Neap, Fld"),
]


def process_flux(list_of_paths, fluxtype='Q', transect='B', filename_datacube=None, filename_flux=None):
    """ Read data and estimate flux through a transect
        The 3-D datacube contains Point #N, Time #N, {ETA/U/V}
        :param list_of_paths: a sequence of path pointing to HDF5 detector files
            (the first one is chosen as base case).
        :param fluxtype: 'Q' for flow discharge, 'E' for energy flux
        :param transect: name of transect (str), 'A', 'B' or 'C'
        :param filename_datacube: output filename for datacube (*.npy), optional
        :param filename_flux: output filename for flux (*.npy), optional
        :return: datacube and flux series
    """
    print('Reading data...', flush=True)

    if fluxtype=='Q':
        flux_all = np.zeros((ntimes, N_CASES))
    if fluxtype=='E':
        flux_all = np.zeros((ntimes, N_CASES, 2))
    RHO = 1025  # seawater density
    GRAV = 9.81 # accel. gravitational

    for i_case, path in enumerate(list_of_paths):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            print('Keys in this file:', keys[:3], '...', keys[-3:])
            # ['detector0000', 'detector0001', 'detector0002', ... 'detector0899', 'time']
            
            # Sanity check on data
            assert npts == len(keys)-1, (
                    f'ERROR reading file {path_hdf5_PF1}, number of detectors {len(keys)-1} does not match npts {npts}.')
            assert keys[-1] == 'time', (
                    f'ERROR reading file {path_hdf5_PF1}, time column not detected!')            
            detector0 = keys[0]    
            data = np.array(f[detector0])
            assert data.shape[1] == 3, (
                    f'ERROR reading time series {detector0}: only {data.shape[1]} columns, expected 3.')
            
            fluxes = np.zeros(ntimes)
            datacube = np.zeros((npts, ntimes, 3))
            case_name = path.split('/')[-3]
            print('\nReading transects for case:', case_name, flush=True)
            for pt in range(npts):
                arr = np.array(f[f'detector{pt:04d}'])
                datacube[pt,:,:] = arr[:MAX_TLEN,:]

            print('Progress:', end=' ', flush=True)
            for idx, timestamp in enumerate(times):
                if idx > MAX_TLEN:
                    break
                if idx % step_output != 0:
                    continue
                t = T_START + timedelta(seconds=float(timestamp))
                if idx % 2000 == 0:
                    print(f'Aug {t.day} {t.hour:02d}:{t.minute:02d}', end='', flush=True)
                elif idx % 200 == 0:
                    print('.', end='', flush=True)
                etas = np.zeros(npts)
                us = np.zeros(npts)
                vs = np.zeros(npts)
                for pt in range(npts):
                    eta,u,v = datacube[pt,idx,0:3]
                    etas[pt] = eta
                    us[pt] = u
                    vs[pt] = v
                if fluxtype=='Q':
                    unit_flux = [np.dot((u,v), nvec) * (depth + eta) 
                                    for depth,eta,u,v in zip(profile_B, etas,us,vs)]
                    fluxes[idx] = delta_L * np.sum(unit_flux)
                elif fluxtype[0]=='E':
                    if transect=='A':
                        profile = profile_A
                        deltaL = deltaL_A
                    elif transect=='C':
                        profile = profile_C
                        deltaL = deltaL_C
                    # a tuple of Epot, Ekin
                    unit_flux = [RHO * (GRAV * eta) * np.dot((u,v), nvec) * (depth + eta)
                                + RHO * (0.5 * np.sqrt(u*u + v*v)) * np.dot((u,v), nvec) * (depth + eta)
                                for depth,eta,u,v in zip(profile, etas,us,vs)]
                    fluxes[idx] = deltaL * np.sum(unit_flux)
                flux_all[idx, i_case] = fluxes[idx]
        
        # Save data for later use
        if filename_datacube:
            np.save(filename_datacube, datacube, allow_pickle=False)
        if filename_flux:
            np.save(filename_flux, flux_all, allow_pickle=False)

    return datacube, flux_all


def process_energy(list_of_paths, filename_inlet_outlet_datacube=None, filename_energy_flux=None):
    """ Read data and estimate energy flux through the 
        The 3-D datacube contains Point #N, Time #N, {ETA/U/V}
        :param list_of_paths: a sequence of path pointing to HDF5 detector files
            (the first one is chosen as base case).
        :param filename_datacube: output filename for datacube (*.npy), optional
        :param filename_discharge: output filename for discharge (*.npy), optional
        :return: datacube and discharge series
    """
    print('Reading data...', flush=True)

    energy_flux_all = np.zeros((ntimes, N_CASES, 2))
    exit(0)
    for i_case, path in enumerate(list_of_paths):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            print('Keys in this file:', keys[:3], '...', keys[-3:])
            # ['detector0000', 'detector0001', 'detector0002', ... 'detector0899', 'time']
            
            # Sanity check on data
            assert npts == len(keys)-1, (
                    f'ERROR reading file {path_hdf5_PF1}, number of detectors {len(keys)-1} does not match npts {npts}.')
            assert keys[-1] == 'time', (
                    f'ERROR reading file {path_hdf5_PF1}, time column not detected!')            
            detector0 = keys[0]    
            data = np.array(f[detector0])
            assert data.shape[1] == 3, (
                    f'ERROR reading time series {detector0}: only {data.shape[1]} columns, expected 3.')
            
            discharges = np.zeros(ntimes)
            datacube_E = np.zeros((npts, ntimes, 3))
            case_name = path.split('/')[-3]
            print('\nReading transects for case:', case_name, flush=True)
            for pt in range(npts):
                arr = np.array(f[f'detector{pt:04d}'])
                datacube_E[pt,:,:] = arr

            print('Progress:', end=' ', flush=True)
            for idx, timestamp in enumerate(times):
                if idx % step_output != 0:
                    continue
                t = T_START + timedelta(seconds=float(timestamp))
                if idx % 2000 == 0:
                    print(f'Aug {t.day} {t.hour:02d}:{t.minute:02d}', end='', flush=True)
                elif idx % 200 == 0:
                    print('.', end='', flush=True)
                # t_check_plot = t + timedelta(seconds=float(step_output * dt/2))
                etas = np.zeros(npts)
                us = np.zeros(npts)
                vs = np.zeros(npts)
                for pt in range(npts):
                    eta,u,v = datacube_E[pt,idx,0:3]
                    etas[pt] = eta
                    us[pt] = u
                    vs[pt] = v
                unit_discharge = [np.dot((u,v), nvec) * (depth + eta) 
                                for depth,eta,u,v in zip(profile_B, etas,us,vs)]
                discharges[idx] = delta_L * np.sum(unit_discharge)
                # if t_check_plot >= times_plot_transect[0]: # high time to plot result
                #     ## This part moved outside - TODO: check why this is not called
                #     times_plot_transect.pop(0)
                #     plt.plot(ds, discharge_part/deltaL, label=f'Aug {t.day} {t.hour}:{t.minute}')
                #     plt.xlabel('Distance along transect, m')
                #     plt.ylabel(r'Unit discharge, m$^3$/s/m')
                #     plt.legend()
                #     plt.grid(True)
                #     plt.show()
                Q_all[idx, i_case] = discharges[idx]
        
        # Save data for later use
        if filename_energy_flux:
            np.save(filename_energy_flux, energy_flux_all, allow_pickle=False)

    return datacube_E, energy_flux_all



def avg_cycle(T, V, alt_sign=True, idx_split=None, printout=False):
    """ Calculate average of values within a half-tidal cycle
        A change in tidal phase can be identified as change in sign of V or when V reaches minimum
        :param T: Time sequence
        :param V: Value sequence
        :param alt_sign: (bool) Use sign changes to split cycles
        :param idx_split: (list) List index to split between ebb and flood
        :param printout: Print out the average values and corresponding times at the end of each segment (cycle)
        :return: Average of values (list)
    """
    counter = 0
    cnt_start = 0
    idx_split_list = []
    Tmean_list = []
    Vmean_list = []
    if idx_split is not None:
        while len(idx_split) > 0:
            idx = idx_split.pop(0)
            if idx > MAX_TLEN:
                break
            Vmean = np.mean(V[cnt_start:idx])
            Tmean = T[cnt_start] + 0.5 * (T[idx] - T[cnt_start])
            Vmean_list.append(Vmean)
            Tmean_list.append(Tmean)
            cnt_start = idx
        return Vmean_list, Tmean_list
    for counter in range(len(T)-1):
        T_curr = T[counter]
        T_next = T[counter+1]
        if alt_sign:
            cond = (V[counter] * V[counter+1] < 0)
        else:
            cond = (V[counter-1] > V[counter] < V[counter+1] if counter > 0
                    else False)  # not effective because of multiple local minima
        if cond:  # change phase
            idx_split_list.append(counter+1)
            Vmean = np.mean(V[cnt_start:counter+1])
            Tmean = T[cnt_start] + 0.5 * (T_next - T[cnt_start])
            Vmean_list.append(Vmean)
            Tmean_list.append(Tmean)
            if printout:
                print(f"{T_curr.day}D {T_curr.hour}H {Vmean:.2f}(unit)", end="   ")
            cnt_start = counter+1
    # np.savetxt('idx_split.txt', idx_split_list)
    return Vmean_list, Tmean_list



def flux_transect(datacube, profile, time_indices, plot=True):
    """ Calculate the distribution of flux (unit discharge, q [m3/s/m]) over a transect 
        :param datacube: The 3-D datacube contains Point #N, Time #N, {ETA/U/V}
        :param profile: series of seabed elevation (zb [m])
        :param time_indices: list of (int, st) time indices and tags, when to calculate q
        :param plot: (bool) produce plot, optionally
        :return: (list of list) unit discharge over transect at selected times
    """
    mpl.rcParams['font.size'] = 12

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex='col')

    dt_plot = [tp - T_START for tp in times_plot_transect]
    tot_sec = [dt.days * 86400 + dt.seconds for dt in dt_plot]
    indices = [t // dtsim for t in tot_sec]

    q_all = []
    # Export discharge at key time steps
    with h5py.File(path_hdf5_PF1, 'r') as f:
        pass

    # ntimes = len(times)
    # dt = float(times[1] - times[0])
    # discharges = np.zeros(ntimes)
    for (it, label) in time_indices:
        t = times[it]
        t_date = T_START + timedelta(seconds=float(t))
        print('Time:', t, 'date:', t_date)
        etas = np.zeros(npts)
        us = np.zeros(npts)
        vs = np.zeros(npts)
        for seg in range(npts):
            arr = np.array(f[f'detector{seg:04d}'])
            eta,u,v = datacube[seg,it,0:3]
            eta = arr[it,0]
            u, v = arr[it,1], arr[it,2]
            # vmag = (arr[it,1]**2 + arr[it,2]**2)**0.5
            etas[seg] = eta
            us[seg] = u
            vs[seg] = v
        discharge_unit = [np.dot((u,v), nvec) * (depth + eta) 
                            for depth,eta,u,v in zip(profile, etas, us, vs)]
        # discharge_part = [q * deltaL for q in discharge_unit]
        # discharges[it] = np.sum(discharge_part)
        a0.plot(ds, discharge_unit, label=label)
        q_all.append(discharge_unit)
        

    a0.set_ylabel(r'Unit discharge, m$^3$/s/m')
    a0.set_xlim((0, 4000))
    a0.fill_between(ds, q_all[0], q_all[2], alpha=0.5)  # new in `fill_between` method: 'color' is now keyword argument
    a0.fill_between(ds, q_all[1], q_all[3], alpha=0.5)
    a0.legend(loc='best', ncol=1, prop={'size':10})
    a0.grid(True)

    zb = -profile
    a1.plot(ds, zb, 'k-')
    a1.set_xlim((0, 4000))
    a1.set_xlabel('Distance along transect, m')
    a1.set_ylabel('Elevation, m')
    a1.fill(ds, zb, "cyan")
    a1.grid(True)

    plt.show()

    exit(0)



def plot_diff(flux_data, list_of_paths, fluxtype='Q', smooth_window=None, cutoff=None, webplot=False):
    """ Plot time series of a flux quantity, e.g. flow discharge 
        and show two groups of difference against the base case,
        i.e. the first case listed in (list of paths).
        This function adjust the colour scheme.
        :param flux_data: data cube of flux quantity with size [N_times, N_cases]
        :param list_of_paths: a sequence of path pointing to HDF5 detector files
            (the first one is chosen as base case).
        :param fluxtype: (str) flux type to show 
        :param smooth_window: (float) window size for moving average diff time series
        :param cutoff: (float) threshold where smaller data values are assigned NaN
        :param webplot: (bool) show HTML plot using PyPlot
        :return: a list of series of percentage differences
    """
    diffs = []
    nt = min(ntimes, MAX_TLEN)
    out_idx = np.arange(0, nt, step_output)

    QMIN_LIM, QMAX_LIM = -3E5, 3E5
    DMIN_LIM, DMAX_LIM = -18, 18

    mpl.rcParams['font.size'] = 16  # bigger font
    f, (a0, av, a1, a2) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 0.6, 1, 1]}, sharex='col')

    if webplot:
        diffs = []
    
    for i_case, path in enumerate(list_of_paths):
        case_name = path.split('/')[-3]
        T_ser = [T_START+timedelta(seconds=float(t)) for t in times[out_idx]]
        Q_ser = flux_data[:nt, i_case]  # Q is a quantity, not necessarily flow discharge
        alt_sign = (fluxtype=='Q')  # discharge-series has alternative signs which can be used for splitting cycles
        idx_split = np.loadtxt('idx_split.txt')
        idx_split = [int(i) for i in idx_split]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        a0.annotate(rf'$\uparrow$ flood', (mdates.date2num(T_PLOT_BEGIN + timedelta(hours=8)),  2.2E5), 
                    ha='left', va='center', size=12)
        a0.annotate('↓  ebb ', (mdates.date2num(T_PLOT_BEGIN + timedelta(hours=8)), -2.2E5),
                    ha='left', va='center', size=12)
        if i_case == 0:
            a0.plot(T_ser, Q_ser[out_idx], 'b', linewidth=1, alpha=0.99)
            difference = np.zeros(nt)
            Qmean_base, Tmean_base = avg_cycle(T_ser, Q_ser, alt_sign=alt_sign, idx_split=idx_split)
            if fluxtype=='Q':
                Tfld = (np.array(T_ser))[Q_ser > 0]
                # split this time series into contiguous chunks
                # credit Haleemur Ali on StackOverflow/a/74521405
                cut_points = np.where(np.diff(Tfld) > 2*timedelta(seconds=dt))[0] + 1
                Tfld = np.split(Tfld, cut_points)
                for Tfld_seg in Tfld:
                    a0.fill_between(Tfld_seg, QMIN_LIM, QMAX_LIM, color='gray', alpha=0.2)
                    a1.fill_between(Tfld_seg, DMIN_LIM, DMAX_LIM, color='gray', alpha=0.2)
                    a2.fill_between(Tfld_seg, DMIN_LIM, DMAX_LIM, color='gray', alpha=0.2)
        else:
            Qmean_comp, Tmean_comp = avg_cycle(T_ser, Q_ser, alt_sign=alt_sign, idx_split=idx_split)
            if fluxtype=='Q':
                if Qmean_base[0] * Qmean_comp[0] < 0:  # opposite phases
                    Qmean_comp = np.roll(Qmean_comp, -1)
            difference = [100 * (Q1 - Q0) / Q0 for Q0, Q1 in zip(Qmean_base, Qmean_comp)]
            # max_time_diff = max([abs(t1 - t0) for t0, t1 in zip(Tmean_base, Tmean_comp)])
            
            # if max_time_diff < timedelta(hours=2): 
            #     raise ValueError()"Too much difference in phase!")
            if smooth_window is not None:
                difference = np.convolve(difference, np.ones(smooth_window)/smooth_window, mode='same')  # moving average
            if cutoff is not None:
                difference[abs(difference) > DIFF_CUTOFF] = np.nan
            diffs.append(difference)

            av.step(Tmean_base, np.abs(Qmean_base), color='blue', where='mid')
            av.fill_between(Tmean_base, np.abs(Qmean_base), color='cyan', alpha=0.3, step='mid')
            if i_case == 1 or i_case == 2:
                a1.step(Tmean_base, difference, label=case_name[5:10], where='mid') 
            elif i_case == 3 or i_case == 4:
                a2.step(Tmean_base, difference, color=color_cycle[i_case-1], 
                        label=case_name[5:10], where='mid') 

    if webplot:
        fig = go.Figure()
        for i, path in enumerate(list_of_paths):
            if i == 0:
                continue  # no difference for base case
            case_name = path.split('/')[-3]
            assert len(Tmean_base)==len(diffs[i]), f"Unequal length: T[{len(Tmean_base)}], diff[{len(diffs[i])}]"
            fig.add_trace(go.Scatter(x=Tmean_base, y=diffs[i-1], name=case_name[5:10]))
        fig.update_layout(xaxis_title='Datetime', yaxis_title=f'$\Delta \bar {fluxtype} / \bar {fluxtype} %$')
        fig.show()
    
    a0.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    av.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    if fluxtype=='Q':
        flux_unit = r'm$^3$/s'
        a0.set_ylabel(rf'$Q$, {flux_unit}')
    elif fluxtype=='E':
        flux_unit = r'J/m/s'
        a0.set_ylabel(rf'$E$, J/m/s')
    av.set_ylabel(rf'$ | \bar {fluxtype} | $, ' + flux_unit)
    a1.set_ylabel(rf'$\Delta \bar {fluxtype} / \bar {fluxtype}$, %')
    a2.set_ylabel(rf'$\Delta \bar {fluxtype} / \bar {fluxtype}$, %')
    # T_LAST = T_START + timedelta(days=N_DAYS)  # otherwise, use default T_LAST
    a0.set_xlim(T_PLOT_BEGIN, T_PLOT_END)
    av.set_xlim(T_PLOT_BEGIN, T_PLOT_END)
    a1.set_xlim(T_PLOT_BEGIN, T_PLOT_END)
    a2.set_xlim(T_PLOT_BEGIN, T_PLOT_END)
    myFmt = mdates.DateFormatter('%d/%m')
    a2.xaxis.set_major_formatter(myFmt)

    if fluxtype == 'Q':
        a0.set_ylim(QMIN_LIM, QMAX_LIM)
        a1.set_ylim(DMIN_LIM, DMAX_LIM)
        a2.set_ylim(DMIN_LIM, DMAX_LIM)
    elif fluxtype == 'E':
        pass
        # a0.set_ylim(0.8*flux_min, 1.2*flux_max)
        # a0.set_ylim(-3E5, 3E5)
        # a1.set_ylim(-15, 10)
        # a2.set_ylim(-15, 10)
    a1.legend(loc='best', ncol=2, prop={'size':10})
    a2.legend(loc='best', ncol=2, prop={'size':10})
    a0.grid(True)
    av.grid(True)
    a1.grid(True)
    a2.grid(True)

    # Indicate selected representative time instances ebb/flood
    if fluxtype=='Q':
        annot = ['ES', 'FS', 'EN', 'FN', 'ES2', 'FS2']
        align = ['right', 'left', 'right', 'left', 'right', 'left']
        for k, (it, _) in enumerate(time_idx_out):
            t = times[it]
            t_date = T_START + timedelta(seconds=float(t))
            a0.plot((t_date, t_date), (-2.2E5, 2.5E5), '--', color='gray', linewidth=0.7)
            a0.annotate(annot[k], (mdates.date2num(t_date), -2.5E5),
                        ha=align[k], va='center', size=12)

    plt.tight_layout()
    plt.show()
    return diffs, Tmean_base


## Main 
# To be soon wrapped in if __name__ == '__main__'
# debugging is now straightforward.

## STEP 1:  RUN and STORE 
# do once to obtain the *.npy files
# datacube, Q_all = process_flux(all_paths, fluxtype='Q', transect='B',
#                             filename_datacube='raw_flux_output/sthuv_24.npy', 
#                             filename_flux='raw_flux_output/Q_all_24.npy')
# exit(0)
## STEP 2: 
# Now you already have the *.npy data file stored
print('Reading data... ')
datacube = np.load('raw_flux_output/sthuv_24.npy')  # sthuv_24.npy
Q_all = np.load('raw_flux_output/Q_all_24.npy')  # Q_all_24.npy
# q_series = flux_transect(datacube, profile_B, time_idx_out)  # flux distribution across the transect
step_output = 1
diffs, Tmean_base, = plot_diff(Q_all, all_paths, fluxtype='Q', webplot=False)
exit(0)


# Energy flux
# must change list of paths here
# STEP 1
# datacube_A, E_all_A = process_flux(all_paths, fluxtype='E', transect='A',
#                                     filename_datacube='sthuv_A_24.npy', 
#                                     filename_flux='Epk_A_all_24.npy')
# datacube_C, E_all_C = process_flux(all_paths, fluxtype='E', transect='C',
#                                     filename_datacube='sthuv_C_24.npy', 
#                                     filename_flux='Epk_C_all_24.npy')
# exit(0)

# # STEP 2
# E_all_A = np.load('E_A_all_24.npy')
# E_all_C = np.load('E_C_all_24.npy')

# step_output = 1
# diffs, Tmean_base = plot_diff(E_all_A, all_paths, fluxtype='E', webplot=False)
# diffs, Tmean_base = plot_diff(E_all_C, all_paths, fluxtype='E', webplot=False)
