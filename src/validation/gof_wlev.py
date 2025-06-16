#!/usr/bin/env python
''' Goodness of fit (GoF) between the observed and simulated tidal data. 
    New version: reading from a cleaned version of BODC text file (observed)
    and HDF5 file (simulated).
'''
import os
import sys
import math
import uptide
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Swapping blue and orange in the default color cycle and assign back to matplotlib
# https://stackoverflow.com/a/9398214
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
        color=['#ff7f0e', '#1f77b4', '#2ca02c',
               '#d62728', '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])


def plot(TZobs, *TZsim, labels=None, webplot=False):
    """ 
    Plot two time series to compare the simulation results against measured data. 
    :param TZobs: tuple of series (time, Z) observed (datetime, float)
    :param *TZsim: optionally, one or more tuple of series (time, Z) simulated (datetime, float)
    :param labels: optional list of labels for the simulated time series
    :param webplot: if True, plot on web browser

    :returns: None
    :side effects: plot to screen and (optionally) on web browser

    Note: TZsim might include TZideal (e.g. harmonic reconstruction using Uptide)
    """
    import plotly.graph_objs as go
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    (Tobs, Zobs) = TZobs
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(Tobs, Zobs, marker='o', markersize=2, linestyle='None', label='Obs.')
    for (i, (Tsim, Zsim)) in enumerate(TZsim):
        ax.plot(Tsim, Zsim, linewidth=0.75, label=labels[i])
    
    myFmt = mdates.DateFormatter('%d/%m/%Y')
    ax.xaxis.set_major_formatter(myFmt)
    plt.xlabel('Datetime')
    plt.ylabel(r'Tidal level η (m)')
    plt.grid()
    plt.legend(ncol=2+len(TZsim), loc='upper left')
    plt.ylim([-2, 2.5])
    plt.tight_layout()
    plt.show()

    if webplot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Tobs, y=Zobs, name='Obs.', mode='markers'))
        for (i, (Tsim, Zsim)) in enumerate(TZsim):
            fig.add_trace(go.Scatter(x=Tsim, y=Zsim, name=labels[i]))
        fig.show()


def calc_goodness_of_fit(Tobs, Zobs, Tsim, Zsim, idx_trim_start=0):
    """ 
    Calculate the goodness of fit between `Zmeas` and `Zsim`
    not necessarily in the same time instances 
    represented through RMSE and R^2 criteria.

    :param Tmeas: time series, measured (list, numpy vector, or pandas series of datetime)
    :param Zmeas: water level series, measured (list, numpy vector, or pandas series of float)
    :param Tsim: time series, simulated (list, numpy vector, or pandas series of datetime)
    :param Zsim: water level series, simulated (list, numpy vector, or pandas series of float)
    :param idx_trim_start: index of the first trimmed point (not necessarily 0)
    :returns: (RMSE, R2)  (float, float)
    """
    assert pd.Series(Tobs).is_monotonic_increasing, \
            "Measured Time Series not monotonic increasing!"
    assert pd.Series(Tsim).is_monotonic_increasing, \
            "Simulated Time Series not monotonic increasing!"
    assert len(Tobs)==len(Zobs) and len(Tsim)==len(Zsim), \
            "Incompatible length of data series!"
    N = len(Tsim)
    
    # interpolate Zmeas to equidistant Tsim 
    Zinterp = [0]*N
    pos = 0
    for i in range(N):
        while True:
            if Tobs[pos] <= Tsim[i] <= Tobs[pos+1]:
                DeltaZ = Zobs[pos+1] - Zobs[pos]
                DeltaT = Tobs[pos+1] - Tobs[pos]
                Zinterp[i] = Zobs[pos] + DeltaZ * (Tsim[i] - Tobs[pos])/DeltaT
                break
            else:
                pos += 1
            
    SE = 0    # accumulation squared errors
    var = 0   # accumulation variance
    Zmean = np.mean(Zinterp)
    # The following is very important: if the type of `Zsim` is pandas Series,
    # we will have to offset the index by `idx_trim_start`. But this is not
    # the case if `Zsim` is a numpy array.
    offset = idx_trim_start if isinstance(Zsim, pd.Series) else 0
    
    for i in range(N):
        SE += (Zinterp[i] - Zsim[i+offset])**2
        var += (Zinterp[i] - Zmean)**2

    RMSE = math.sqrt(SE/N)
    R2 = 1 - SE/var
    
    return RMSE, R2


def ideal_tide(date_time_start, duration, interval):
    """ 
    Generate the ideal tidal water level for WICK using `uptide` package.
    :param date_time_start: Start datetime to generate the tidal series
    :param duration: (in seconds) duration of the tidal series
    :param interval: (in seconds) sampling interval of the tidal series
    :returns: (T, Z)  (list of datetime, list of float)
    """
    # data from WICK record in file 'inputs/uesful_gauges.csv'
    tide = uptide.Tides(['Q1', 'O1', 'P1', 'S1', 'K1', '2N2', 'MU2', 'N2', 'NU2',
                         'M2', 'L2', 'T2', 'S2', 'K2', 'M4'])
    amp = [0.039, 0.114, 0.033, 0.009, 0.11, 0.029, 0.02, 0.204, 0.042, 
           1.016, 0.034, 0.016, 0.348, 0.099, 0.036]
    pha = [336.7, 27.7, 163.2, 98, 175.9, 282.3, 305.7, 302, 304,
            322.3, 328.4, 348.6, 0.3, 357.7, 317.5]
    pha = [math.radians(p) for p in pha]  # phases (in radians!) of the constituents
    
    assert len(amp) == len(pha), "Incompatible length of amplitude and phase lists!"
    
    tide.set_initial_time(date_time_start)
    t = np.arange(0, duration, interval)
    eta = tide.from_amplitude_phase(amp, pha, t)
    times = [date_time_start + timedelta(seconds=j*interval) for j in range(len(t))]
    return times, eta


def trim_series(Tobs, T, *Z, shift=0):
    """ 
    Trim a number of time series to remove the initial and the final
    few data points that are not properly generated.
    (Return value index of first trimmed point is necessary
    for the goodness of fit calculation, which is based on 
    pandas series index (not starting from 0).)

    :param Tobs: time series, observed (sequence of datetime)
    :param T: time series, simulated (sequence of datetime)
    :param *Z: one or more series, simulated (sequence of float)
    :param shift: (in hours) shift the time series by this amount 
        (forward if shift > 0, backward if shift < 0)
    :returns: a tuple of (new T series, new Z series, 
        index of the first trimmed point)
    """
    Tshift = [t + timedelta(hours=shift) for t in T]
    # print(f"For shifted simulation time: {shift} hours.")
    # Check and ensure that the simulated time period as subset of measured time period,
    # otherwise, trim the simulated time series on either end, or both.
    idx_trim_start = 0
    idx_trim_end = len(Tshift) - 1
    while Tshift[idx_trim_start] < Tobs[0]:
        idx_trim_start += 1
    while Tshift[idx_trim_end] > Tobs[-1]:
        idx_trim_end -= 1

    Tcomp = Tshift[idx_trim_start:idx_trim_end]
    Zcomp = []   # Zcomp = Z[idx_trim_start:idx_trim_end]
    for series in Z:
        Zcomp.append(series[idx_trim_start:idx_trim_end])
    
    return Tcomp, Zcomp, idx_trim_start


# Replicate tools/export.hdf5_detector_to_TS() here
# Attempt to do (relative) import failed
def hdf5_detector_to_TS(filename, colidx):
    """ 
    Export time series from detector HDF5 output file.
    :param filename: path to HDF5 file (str)
    :param colidx: index column to export (int). Default=all columns
    :return: array of data in which 1st column is time (2D numpy array)
    """
    import h5py
    import numpy as np
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        # print('List of keys in this file: ', keys)
        assert keys[-1] == 'time', (
                f'ERROR reading file {filename}, time column not detected!')
        print('Reading column ', colidx, ' item: ', keys[colidx])
        group_data = keys[colidx]
        group_time = keys[-1]
        data = np.array(f[group_data])
        assert data.shape[1] == 3, (
                f'ERROR reading time series {group_data}: only {data.shape[1]} columns, expected 3.')
        time = np.array(f[group_time])
        # error to write return here: the HDF5 file is not closed!
        
    return np.hstack((time, data))


if __name__ == "__main__":
    Tobs_all = []   # to aggregate
    Zobs_all = []  # to aggregate
    Tcomp_all = []   # to aggregate
    Zcomp_all = []  # to aggregate

    cases = [
        {
            'tag': 'PF0x1.1Wick',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2017WIC_clean_Aug.csv'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputsMCx1.1/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2, # Wick gauge is always next to last
            't0obs': datetime(2017, 8, 1, 0, 0, 0),
            't0sim': datetime(2017, 8, 1, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF0x1Wick',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2017WIC_clean_Aug.csv'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputs_MCx1/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2017, 8, 1, 0, 0, 0),
            't0sim': datetime(2017, 8, 1, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'Uptide_Wick_2017',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2017WIC_clean_Aug.csv'),
            'file_sim': os.path.expanduser('Uptide'),
            'idx_col': -2,
            't0obs': datetime(2017, 8, 1, 0, 0, 0),
            't0sim': datetime(2017, 8, 1, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF1_Wick_2013',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2013WIC_clean.csv'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputs_MCx1.1_2013/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2013, 1, 1, 0, 0, 0),
            't0sim': datetime(2013, 2,17, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF0_Wick_2013',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2013WIC_clean.csv'),
            # 'file_sim': os.path.expanduser('~/GitHub-repo/TSSM/PF1_interm/outputs_2024_x1MC/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM/PF1_interm/outputs_MCx1.1_2013/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2013, 1, 1, 0, 0, 0),
            't0sim': datetime(2013, 2,17, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF2_Wick_2013',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2013WIC_clean.csv'),
            'file_sim': os.path.expanduser('~/Documents/diagnostic_detectors-adcp_20130217.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2013, 1, 1, 0, 0, 0),
            't0sim': datetime(2013, 2,17, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'Uptide_Wick_2013',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2013WIC_clean.csv'),
            'file_sim': os.path.expanduser('Uptide'),
            'idx_col': None,
            't0obs': datetime(2013, 1, 1, 0, 0, 0),
            't0sim': datetime(2013, 2,17, 0, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF0_Wick_2024',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2024WIC_JanFeb.csv'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputs_MCx1.1_2024/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2024, 1, 1, 0, 0, 0),
            't0sim': datetime(2024, 1,26,12, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF1_Wick_2024',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2024WIC_JanFeb.csv'),
            # 'file_sim': os.path.expanduser('~/GitHub-repo/TSSM/PF1_interm/outputs_2024_x1MC/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM/PF1_interm/out_2024/outputs_run/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2024, 1, 1, 0, 0, 0),
            't0sim': datetime(2024, 1,26,12, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'PF2_Wick_2024',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2024WIC_JanFeb.csv'),
            'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF2_design/src/outputs_MCx1.1_2024/outputs_run__Sucess_No_Delete/diagnostic_detectors-adcp.hdf5'),
            'idx_col': -2,
            't0obs': datetime(2024, 1, 1, 0, 0, 0),
            't0sim': datetime(2024, 1,26,18, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
        {
            'tag': 'Uptide_Wick_2024',
            'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/validation/2024WIC_JanFeb.csv'),
            'file_sim': os.path.expanduser('Uptide'),
            'idx_col': None,
            't0obs': datetime(2024, 1, 1, 0, 0, 0),
            't0sim': datetime(2024, 1,26,18, 0, 0),
            'dtobs': timedelta(seconds=900),
            'dtsim': timedelta(seconds=100),
            'shift': 0,
        },
    ]

    for info in cases:
        tag = info['tag']
        file_obs = info['file_obs']
        file_sim = info['file_sim']
        t0obs = info['t0obs']
        t0sim = info['t0sim']
        dtobs = info['dtobs']
        dtsim = info['dtsim']
        shift = info['shift']
        print(f"\nProcessing {tag}... starting at {t0sim}... shifted by {shift} hours")

        # Ensure correct data file path when running on VS Code debug mode
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is None:
            # discard the path (consider current folder)
            path, file_obs = os.path.split(file_obs)
            path, file_sim = os.path.split(file_sim)
        
        # Read cleaned BODC file *.csv

        # DEV: using pandas
        # table_obs = pd.read_csv(file_obs, skiprows=0, delim_whitespace=False, header=None)
        # Zobs = table_obs[2].dropna()
        # ALT: using numpy (equivalent)
        Zobs = np.genfromtxt(file_obs, delimiter=',', skip_header=0, usecols=(2,))
        # TODO: read also the time column, as there are bad data in the original BODC file
        # and after removing those, the time series will not be equidistant.
        # For now we use clean and trimmed dataset to the validation period.

        avg = Zobs.mean()
        Zobs = Zobs - avg  # "normalise" water level
        Tobs = [t0obs + i*dtobs for i in range(len(Zobs))]  # autogen
        
        if file_sim.lower() != 'uptide':
            table_sim = hdf5_detector_to_TS(file_sim, info['idx_col'])
            # Col 0: time, Col 1: Z
            Zsim = table_sim[:,1]
            Tsim = [t0sim + i*dtsim for i in range(len(Zsim))]  # autogen
        else:  # generate series using uptide
            Tsim, Zsim = ideal_tide(t0sim, 30*24*3600, dtsim.total_seconds())

        # Calculate goodness-of-fit for simulated data series
        Tcomp, (Zcomp,), idx_trim_start = trim_series(Tobs, Tsim, Zsim, shift=shift)
        # print(f"Trim index {idx_trim_start}/{len(Tcomp)} for velocity series.")

        # print("Case\t\tRMSE\tR2")
        RMSE, R2 = calc_goodness_of_fit(Tobs, Zobs,
                                        Tcomp, Zcomp, idx_trim_start)
        print(f"{tag}:\tRMSE={RMSE:.3f}m    R²={R2:.3f}")
        Tobs_all.append(Tobs)
        Zobs_all.append(Zobs)
        Tcomp_all.append(Tcomp)
        Zcomp_all.append(Zcomp)

        # Harmonic analysis
        tide_Wick = {'M2': (1.016, 322.3), 'S2': (0.348, 0.3)}  # published tidal constituents (amplitude, phase)
        constituents = list(tide_Wick.keys())
        tide_PF = uptide.Tides(constituents)
        tide_PF.set_initial_time(t0sim)
        Tsec = [(t - t0sim).total_seconds() for t in Tcomp]
        amp, pha = uptide.harmonic_analysis(tide_PF, Zcomp, Tsec)
        pha = np.degrees(pha)
        for cons, amp_i, pha_i in zip(constituents, amp, pha):
            diff_amp = amp_i - tide_Wick[cons][0]
            diff_pha = (pha_i - tide_Wick[cons][1]) % 360
            print(f"{cons}: Δα={diff_amp:.3f}m, Δφ={diff_pha:.3f}°")


    # Example plotting
    # plot((Tobs_all[0], Zobs_all[0]),  # one line for observation (o)
    #     (Tcomp_all[0], Zcomp_all[0]),
    #     # (Tcomp_all[2], Zcomp_all[2]), # simulation result, 1st (-)
    #     (Tcomp_all[1], Zcomp_all[1]), # simulation result, 2nd (-)
    #                                   # simulation can be Uptide result
    #     labels=('x1.1', 'x1.0'),     # labels (number of simulation results)
    #     )

    # Examine why Cases PF0, PF1, PF2 for 2024 are problematic
    plot((Tobs_all[-1], Zobs_all[-1]),  # one line for observation (o)
        (Tcomp_all[-1], Zcomp_all[-1]),
        # (Tcomp_all[2], Zcomp_all[2]), # simulation result, 1st (-)
        (Tcomp_all[-3], Zcomp_all[-3]), # simulation result, 2nd (-)
                                      # simulation can be Uptide result
        labels=('Uptide', 'PF1'),     # labels (number of simulation results)
        )
