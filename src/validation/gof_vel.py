#!/usr/bin/env python
# coding: utf-8
"""
Calculates the goodness of fit (GoF) 
between the observed and simulated tidal data.

Input: 
    - HDF5 detectors file for each simulation case
    - ADCP files e.g. `U2.csv`, `V2.csv` and `vel2.csv`
      these are generated earlier with `process_adcp.py`

Note: This version does not intermediate CSV files,
thus avoids redundancy.

"""
import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# from ..tools import export  # ImportError: attempted relative import with no known parent package
from scipy.signal import find_peaks
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import chisquare, kstest

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go


def plot(TSobs, *TSsim, plottype, webplot=False, labels=None):
    """ 
    Plot multiple simulated time series against measured data.
    
    :param TSobs: tuple of series (time, X) observed (datetime, float)
    :param *TSsim: optionally, one or more tuple of series 
        (time, X) simulated (datetime, float)
    :param plottype: quantity to be plotted, either 'mag' or 'dir'
    :param webplot: if True, plot on web browser, otherwise,
    :returns: None
    :side effects: plot to screen and on web browser
    """
    
    (Tobs, Xobs) = TSobs
    fig, ax = plt.subplots(figsize=(12,3))
    # ax.plot(Tobs[0], Xobs[0], linewidth=0.75, label=f'')
    ax.plot(Tobs, Xobs, marker='o', markersize=2, linestyle='None', label='Obs.')
    for (i, (Tsim, Xsim)) in enumerate(TSsim):
        ax.plot(Tsim, Xsim, linewidth=0.75, label=labels[i])

    if plottype =='mag':
        ylabel = 'Vel. mag. (m/s)'
        legend_loc = 'upper left'
    elif plottype == 'dir':
        ylabel = 'Flow dir. (deg)'
        legend_loc = 'lower left'
    else:
        raise ValueError(f"Invalid plottype: {plottype}. Must be either 'mag' or 'dir'.")

    myFmt = mdates.DateFormatter('%d/%m/%Y')
    ax.xaxis.set_major_formatter(myFmt)
    plt.xlabel('Datetime')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(ncol=2+len(TSsim), loc=legend_loc)
    plt.xlim(Tsim[0], Tsim[-1])
    plt.tight_layout()
    plt.show()

    if webplot:
        fig = go.Figure()
        for (i, (Tsim, Xsim)) in enumerate(TSsim):
            fig.add_trace(go.Scatter(x=Tsim, y=Xsim, name=f'{labels[i]}'))
        fig.add_trace(go.Scatter(x=Tobs, y=Xobs, name='Obs.', mode='markers'))
        fig.update_layout(xaxis_title='Datetime', yaxis_title=ylabel)
        fig.show()


def calc_goodness_of_fit(Tobs, vel_obs, dir_obs, Tsim, vel_sim, dir_sim,
                         idx_trim_start=0, sep_flood_ebb=True):
    """ 
    Calculate the goodness of fit between observed (`obs`) and simulated (`sim`)
    vector quantities. Comparison is made for both magnitude and direction, and
    represented through RMSE and R^2 criteria.

    Optionally, the series are split into flood and ebb phases.
    These time series are not necessarily in the same time instances 
    
    :param Tobs: time series, observed (sequence of datetime)
    :param vel_obs: data series, observed (sequence of float)
    :param dir_obs: angular data series, observed (sequence of float)
    :param Tsim: time series, simulated (sequence of datetime)
    :param vel_sim: data series, simulated (sequence of float)
    :param dir_sim: angular data series, simulated (sequence of float)
    :param idx_trim_start: index of the first trimmed point (not necessarily 0)
    :param sep_flood_ebb: if True, then perform comparison in flood-ebb direction
    :returns: (RMSE, NRMSE, R2)  (float, float)
    """
    assert pd.Series(Tobs).is_monotonic_increasing, \
            "Observed Time Series not monotonic increasing!"
    assert pd.Series(Tsim).is_monotonic_increasing, \
            "Simulated Time Series not monotonic increasing!"
    assert len(Tobs)==len(vel_obs) and len(Tsim)==len(vel_sim), \
            "Incompatible length of data series!"
    N = len(Tsim)
    
    # interpolate Zmeas to equidistant Tsim 
    vel_intp = [0]*N
    dir_intp = [0]*N
    pos = 0
    for i in range(N):
        while True:
            if Tobs[pos] <= Tsim[i] <= Tobs[pos+1]:
                Delta_vel = vel_obs[pos+1] - vel_obs[pos]
                Delta_dir = dir_obs[pos+1] - dir_obs[pos]
                DeltaT = Tobs[pos+1] - Tobs[pos]
                vel_intp[i] = vel_obs[pos] + Delta_vel * (Tsim[i] - Tobs[pos])/DeltaT
                dir_intp[i] = dir_obs[pos] + Delta_dir * (Tsim[i] - Tobs[pos])/DeltaT
                break
            else:
                pos += 1
            
    Er_vel = 0    # accumulation errors
    AE_vel = 0    # accumulation absolute errors
    SE_vel = 0    # accumulation squared errors
    var_vel = 0   # accumulation variance
    mean_vel = 0
    
    Er_dir = 0    # accumulation errors
    AE_dir = 0    # accumulation absolute errors
    SE_dir = 0    # accumulation squared errors
    var_dir = 0   # accumulation variance
    mean_dir = 0

    Er_vel_flood = 0    # accumulation errors
    AE_vel_flood = 0    # accumulation absolute errors
    SE_vel_flood = 0    # accumulation squared errors
    var_vel_flood = 0   # accumulation variance
    mean_vel_flood = 0

    Er_dir_flood = 0    # accumulation errors
    AE_dir_flood = 0    # accumulation absolute errors
    SE_dir_flood = 0    # accumulation squared errors
    var_dir_flood = 0   # accumulation variance
    mean_dir_flood = 0

    Er_vel_ebb = 0    # accumulation errors
    AE_vel_ebb = 0    # accumulation absolute errors
    SE_vel_ebb = 0    # accumulation squared errors
    var_vel_ebb = 0   # accumulation variance
    mean_vel_ebb = 0

    Er_dir_ebb = 0    # accumulation errors
    AE_dir_ebb = 0    # accumulation absolute errors
    SE_dir_ebb = 0    # accumulation squared errors
    var_dir_ebb = 0   # accumulation variance
    mean_dir_ebb = 0

    count_flood = 0
    count_ebb = 0
    count_slack = 0
    
    # The following is very important: if the type of `Zsim` is pandas Series,
    # we will have to offset the index by `idx_trim_start`. But this is not
    # the case if `Zsim` is a numpy array.
    offset = idx_trim_start if isinstance(vel_sim, pd.Series) else 0

    SCALE_DIR = 1   # to reduce the running total of sum squares (TSS) angle (dir)
    DIR_MIN_FLOOD = 300
    DIR_MAX_FLOOD = 360
    DIR_MIN_EBB = 150
    DIR_MAX_EBB = 210

    if sep_flood_ebb:
        for i in range(N):
            try:
                vel_sim_item = vel_sim[i+offset]
                dir_sim_item = dir_sim[i+offset]
                if MIN_VEL < vel_sim_item < MAX_VEL:
                    diff = dir_intp[i] - dir_sim_item
                    diff = (diff + 180) % 360 - 180  # compare angles https://stackoverflow.com/a/7869457 
                    if DIR_MIN_FLOOD < dir_sim_item < DIR_MAX_FLOOD:
                        count_flood += 1
                        Er_vel_flood += (vel_intp[i] - vel_sim_item)
                        AE_vel_flood += abs(vel_intp[i] - vel_sim_item)
                        SE_vel_flood += (vel_intp[i] - vel_sim_item)**2
                        mean_vel_flood += vel_intp[i]
                        var_vel_flood += (vel_intp[i])**2

                        Er_dir_flood += diff
                        AE_dir_flood += abs(diff)
                        SE_dir_flood += diff**2
                        mean_dir_flood += dir_intp[i]
                        var_dir_flood += SCALE_DIR * (dir_intp[i])**2
                    
                    elif DIR_MIN_EBB < dir_sim[i+offset] < DIR_MAX_EBB:
                        count_ebb += 1
                        Er_vel_ebb += (vel_intp[i] - vel_sim_item)
                        AE_vel_ebb += abs(vel_intp[i] - vel_sim_item)
                        SE_vel_ebb += (vel_intp[i] - vel_sim_item)**2
                        mean_vel_ebb += vel_intp[i]
                        var_vel_ebb += (vel_intp[i])**2

                        Er_dir_ebb += diff
                        AE_dir_ebb += abs(diff)
                        SE_dir_ebb += diff**2
                        mean_dir_ebb += dir_intp[i]
                        var_dir_ebb += SCALE_DIR * (dir_intp[i])**2
                    
                    else:  # slack
                        count_slack += 1
                        Er_vel += (vel_intp[i] - vel_sim_item)
                        AE_vel += abs(vel_intp[i] - vel_sim_item)
                        SE_vel += (vel_intp[i] - vel_sim_item)**2
                        mean_vel += vel_intp[i]
                        var_vel += (vel_intp[i])**2
                    
                        Er_dir += diff
                        AE_dir += abs(diff)
                        SE_dir += diff**2
                        mean_dir += dir_intp[i]
                        var_dir += SCALE_DIR * (dir_intp[i])**2
                
            except IndexError:
                print("Terminated at index", i)
                break

        mean_vel_flood /= count_flood
        mean_dir_flood /= count_flood
        mean_vel_ebb /= count_ebb
        mean_dir_ebb /= count_ebb

        var_vel_flood -= (mean_vel_flood**2)*count_flood
        var_dir_flood -= (mean_dir_flood**2)*count_flood
        var_vel_ebb -= (mean_vel_ebb**2)*count_ebb
        var_dir_ebb -= (mean_dir_ebb**2)*count_ebb

        RMSE_vel_flood = math.sqrt(SE_vel_flood/count_flood)
        RMSE_dir_flood = math.sqrt(SE_dir_flood/count_flood)
        NRMSE_vel_flood = RMSE_vel_flood/mean_vel_flood
        NRMSE_dir_flood = RMSE_dir_flood/mean_dir_flood
        MAE_vel_flood = AE_vel_flood/count_flood
        MAE_dir_flood = AE_dir_flood/count_flood
        R2_vel_flood = 1 - SE_vel_flood/var_vel_flood
        R2_dir_flood = 1 - SE_dir_flood/var_dir_flood
        Bias_vel_flood = Er_vel_flood/count_flood
        Bias_dir_flood = Er_dir_flood/count_flood

        RMSE_vel_ebb = math.sqrt(SE_vel_ebb/count_ebb)
        RMSE_dir_ebb = math.sqrt(SE_dir_ebb/count_ebb)
        NRMSE_vel_ebb = RMSE_vel_ebb/mean_vel_ebb
        NRMSE_dir_ebb = RMSE_dir_ebb/mean_dir_ebb
        MAE_vel_ebb = AE_vel_ebb/count_ebb
        MAE_dir_ebb = AE_dir_ebb/count_ebb
        R2_vel_ebb = 1 - SE_vel_ebb/var_vel_ebb
        R2_dir_ebb = 1 - SE_dir_ebb/var_dir_ebb
        Bias_vel_ebb = Er_vel_ebb/count_ebb
        Bias_dir_ebb = Er_dir_ebb/count_ebb

        if count_slack > 0:
            mean_vel /= count_slack
            mean_dir /= count_slack
            var_vel -= (mean_vel**2)*count_slack
            var_dir -= (mean_dir**2)*count_slack
            RMSE_vel = math.sqrt(SE_vel/count_slack)
            RMSE_dir = math.sqrt(SE_dir/count_slack)
            NRMSE_vel = RMSE_vel/mean_vel
            NRMSE_dir = RMSE_dir/mean_dir
            MAE_vel = AE_vel/count_slack
            MAE_dir = AE_dir/count_slack
            R2_vel = 1 - SE_vel/var_vel
            R2_dir = 1 - SE_dir/var_dir
            Bias_vel = Er_vel/count_slack
            Bias_dir = Er_dir/count_slack
        else:
            RMSE_vel = np.nan
            RMSE_dir = np.nan
            NRMSE_vel = np.nan
            NRMSE_dir = np.nan
            MAE_vel = np.nan
            MAE_dir = np.nan
            R2_vel = np.nan
            R2_dir = np.nan
            Bias_vel = np.nan
            Bias_dir = np.nan

    else:
        mean_vel = np.mean(vel_intp)
        mean_dir = np.mean(dir_intp)
        for i in range(N):
            try:
                SE_vel += (vel_intp[i] - vel_sim[i+offset])**2
                var_vel += (vel_intp[i] - mean_vel)**2
                # shortest circular distance https://stackoverflow.com/a/7869457 
                diff = dir_intp[i] - dir_sim[i+offset]
                diff = (diff + 180) % 360 - 180
                SE_dir += diff**2
                var_dir += (dir_intp[i] - mean_dir)**2
            except IndexError:
                # print("Terminated at index", i)
                break
    
        RMSE_vel = math.sqrt(SE_vel/N)
        NRMSE_vel = RMSE_vel/mean_vel
        R2_vel = 1 - SE_vel/var_vel
        RMSE_dir = math.sqrt(SE_dir/N)
        NRMSE_dir = RMSE_dir/mean_dir
        R2_dir = 1 - SE_dir/var_dir
    
    gof = dict(
        flood_mag = dict(
            RMSE=RMSE_vel_flood,
            NRMSE=NRMSE_vel_flood,
            R2=R2_vel_flood,
        ),
        flood_dir = dict(
            RMSE=RMSE_dir_flood,
            NRMSE=NRMSE_dir_flood,
            MAE=MAE_dir_flood,
            R2=R2_dir_flood,
            Bias=Bias_dir_flood,
        ),
        ebb_mag = dict(
            RMSE=RMSE_vel_ebb,
            NRMSE=NRMSE_vel_ebb,
            R2=R2_vel_ebb,
        ),
        ebb_dir = dict(
            RMSE=RMSE_dir_ebb,
            NRMSE=NRMSE_dir_ebb,
            MAE=MAE_dir_ebb,
            R2=R2_dir_ebb,
            Bias=Bias_dir_ebb,
        ),
        mag = dict(
            RMSE=RMSE_vel,
            NRMSE=NRMSE_vel,
            R2=R2_vel,
        ),
        dir = dict(
            RMSE=RMSE_dir,
            NRMSE=NRMSE_dir,
            MAE=MAE_dir,
            R2=R2_dir,
            Bias=Bias_dir,
        )
    )
    return gof


def comp_peaks(T1, Vel1, T2, Vel2, tag='', plot=True, show_title=True,
               series=['Vel1', 'Vel2'], auto_trim=True):
    """
    Compare two time series of peaks.
    :param T1: Time sequence 1
    :param Vel1: Velocity sequence 1
    :param T2: Time sequence 2
    :param Vel2: Velocity sequence 2
    :param tag: Tag for the figure (displayed in the title)
    :param plot: If True, plot the time series and correlation of velocity peaks
    :param show_title: If True, show the title of the figure
    :param series: Used for the legend
    :param auto_trim: Trim the longer peak series to the shorter one.
    :return: (r2, r2_flood, r2_ebb) Coefficient of determination for the two time series.
    :sideeffect: Plots the time series and the peaks.

    NB: The two time series must have the same length. However, measured signals can
    be noisy and does not coincide with the simulated signal. Therefore, an option 
    to automatically trim the longer series to the shorter one is implemented.
    """
    peaks1, _ = find_peaks(Vel1, prominence=1.0)
    peaks2, _ = find_peaks(Vel2, prominence=1.0)
    T1 = np.array(T1)
    Vel1 = Vel1.array  # List -> array conversion
    T2 = np.array(T2)

    if len(peaks1) != len(peaks2):
        if auto_trim:
            if len(peaks1) > len(peaks2):
                peaks1 = peaks1[:len(peaks2)]
            else:
                peaks2 = peaks2[:len(peaks1)]
        else:
            raise IndexError(f"""Cannot compare -- different number of peaks
                             {len(peaks1)} vs {len(peaks2)}
                             between the two time series!""")

    # splitting flood-ebb peak flows by picking alternative values in the series
    T1_flood = T1[peaks1[::2]]
    T2_flood = T2[peaks2[::2]]
    Upeak1_fld = Vel1[peaks1[::2]]
    Upeak2_fld = Vel2[peaks2[::2]]
    T1_ebb = T1[peaks1[1::2]]
    T2_ebb = T2[peaks2[1::2]]
    Upeak1_ebb = Vel1[peaks1[1::2]]
    Upeak2_ebb = Vel2[peaks2[1::2]]
    COLOR_NAVY = '#1f77b4'
    COLOR_ORANGE = '#ff7f0e'

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 14
    if plot:
        fig1 = plt.figure(figsize=(13, 6))
        plt.plot(T1_flood, Upeak1_fld, ">", color=COLOR_NAVY, markerfacecolor='none', label=f"Pk Fld {series[0]}")
        plt.plot(T2_flood, Upeak2_fld, ">", color=COLOR_NAVY, label=f"Pk Fld {series[1]}")
        plt.plot(T1_ebb, Upeak1_ebb, "<", color=COLOR_ORANGE, markerfacecolor='none', label=f"Pk Ebb {series[0]}")
        plt.plot(T2_ebb, Upeak2_ebb, "<", color=COLOR_ORANGE, label=f"Pk Ebb {series[1]}")
        plt.plot(T1, Vel1, label=series[0], color='green', linewidth=0.5)
        plt.plot(T2, Vel2, label=series[1], color='red', linewidth=0.5)
        plt.legend(ncols=2)
        myFmt = mdates.DateFormatter('%d/%m/%Y')
        fig1.gca().xaxis.set_major_formatter(myFmt)

        if show_title:
            plt.title(f"Flow vel. time series {tag}")
        plt.ylabel("Vel. (m/s)")
        plt.grid()

        fig2 = plt.figure(figsize=(7, 6))
        plt.xlabel(f"Vel. {series[0]}, m/s")
        plt.ylabel(f"Vel. {series[1]}, m/s")
        fig2.gca().set_xlim(2, 4.5)
        fig2.gca().set_ylim(2, 4.5)
        if show_title:
            plt.title(f"Scatter plot of peak vel. {tag}")

    Upeak1 = Vel1[peaks1]  # obs
    Upeak2 = Vel2[peaks2]  # sim
    # Average of observed peaks, for normalisation
    Upeak_avg = np.mean(Upeak1)
    Upeak_flood_avg = np.mean(Upeak1_fld)
    Upeak_ebb_avg = np.mean(Upeak1_ebb)
    nrmse = mean_squared_error(Upeak1, Upeak2, squared=False) / Upeak_avg  # RMSE
    nrmse_flood = mean_squared_error(Upeak1_fld, Upeak2_fld, squared=False) / Upeak_flood_avg
    nrmse_ebb = mean_squared_error(Upeak1_ebb, Upeak2_ebb, squared=False) / Upeak_ebb_avg
    r2 = r2_score(Vel1[peaks1], Vel2[peaks2])
    r2_flood = r2_score(Upeak1_fld, Upeak2_fld)
    r2_ebb = r2_score(Upeak1_ebb, Upeak2_ebb)
    if plot:
        plt.scatter(Upeak1_fld, Upeak2_fld, label="flood")
        plt.scatter(Upeak1_ebb, Upeak2_ebb, label="ebb")
        # plotting the best fit line https://stackoverflow.com/a/31800660/4956603 
        plt.plot(np.unique(Upeak1_fld), 
                np.poly1d(np.polyfit(Upeak1_fld, Upeak2_fld, 1))(np.unique(Upeak1_fld)), 'k:')
        plt.plot(np.unique(Upeak1_ebb), 
                np.poly1d(np.polyfit(Upeak1_ebb, Upeak2_ebb, 1))(np.unique(Upeak1_ebb)), 'k:')
        # plt.plot(np.unique(Vel1[peaks1]), 
        #          np.poly1d(np.polyfit(Vel1[peaks1], Vel2[peaks2], 1))(np.unique(Vel1[peaks1])), 'k--')
        plt.plot([2,5], [2,5], 'green', linewidth=0.5)
        
        plt.annotate(f"$R^2$ = {r2:.3f} ALL", (0.05, 0.90), xycoords="axes fraction")
        plt.annotate(f"$R^2$ = {r2_flood:.3f} Flood", (0.05, 0.85), xycoords="axes fraction")
        plt.annotate(f"$R^2$ = {r2_ebb:.3f} Ebb", (0.05, 0.80), xycoords="axes fraction")
        plt.legend(loc="lower right")
        plt.gca().set_aspect('equal', 'box')
        plt.grid()
        plt.show()
    return (r2, r2_flood, r2_ebb), (nrmse, nrmse_flood, nrmse_ebb)


def chi_sq(seq1, seq2, tag='', show_plot=True, series=['Vel1', 'Vel2']):
    """
    Compare histograms of the two time series.
    :param seq1: Data sequence 1 (observed)
    :param seq2: Data sequence 2 (expected)
    :param tag: Tag for the figure (displayed in the title)
    :param show_plot: Show the histogram plot
    :param series: Used for the legend
    :return: None
    :sideeffect: Plots the two histograms, overlapped.
    """
    mpl.rcParams['font.size'] = 12
    fig3 = plt.figure(figsize=(7, 6))
    bin_seq = np.linspace(0, max(max(seq1), max(seq2)), 25)
    hist1, *rest = plt.hist(seq1, bins=bin_seq, density=True, edgecolor='black', alpha=0.5, label=f"{series[0]}")
    hist2, *rest = plt.hist(seq2, bins=bin_seq, density=True, alpha=0.5, label=f"{series[1]}")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Normalised Frequency")
    if show_plot:
        plt.title(f"Histogram of vel. {tag}")
        plt.gca().set_ylim(0, 0.4)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.expanduser(f"~/Documents/hist/hist_{tag}.pdf"))
        plt.show()
    # Discard the bins in seq_hist2 = 0.
    kmin, kmax = 0, len(hist2) - 1
    while hist2[kmin] == 0:
        kmin += 1
    while hist2[kmax] == 0:
        kmax -= 1
    seq_trim1 = hist1[kmin:kmax]
    seq_trim2 = hist2[kmin:kmax]
    return chisquare(seq_trim1, seq_trim2)
    

def ks(seq1, seq2, tag='', show_plot=True, series=['Vel1', 'Vel2']):
    """
    Compare histograms of the two time series based on the Kolmogorov-Smirnov test.
    :param seq1: Data sequence 1 (observed)
    :param seq2: Data sequence 2 (simulated)
    :param tag: Tag for the figure (displayed in the title)
    :param show_plot: Show the histogram plot
    :param series: Used for the legend
    :return: None
    :sideeffect: Plots the two histograms, overlapped.
    """
    if show_plot:
        mpl.rcParams['font.size'] = 12
        plt.figure(figsize=(7, 6))
        ax = plt.gca()
        n_bins = 25
        bin_seq = np.linspace(0, max(max(seq1), max(seq2)), n_bins)
        plt.hist(seq1, bins=bin_seq, density=True, histtype='step', cumulative=True, label=f"{series[0]}")
        plt.hist(seq2, bins=bin_seq, density=True, histtype='step', cumulative=True, label=f"{series[1]}")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Probability")
        plt.title(f"Histogram of vel. {tag}")
        ax.grid(True)
        ax.legend(loc='best')
        plt.savefig(os.path.expanduser(f"~/Documents/hist/cdf_{tag}.pdf"))
        plt.show()
    return kstest(seq1, seq2)


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
    cases = [
            # the tagnames PF0 and PF1 are swapped (as folders cannot be renamed)
            {
                'tag': 'PF1 (base Manning) ADCP2',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputs_MCx1/outputs_run/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 14,
                't0obs': datetime(2017, 8, 3, 11, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            {
                'tag': 'PF1 (base Manning) ADCP3',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputs_MCx1/outputs_run/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 15,
                't0obs': datetime(2017, 8, 2, 12, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                't_end': datetime(2017,8,29, 22,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            {
                'tag': 'PF1 (Manning x 1.1) ADCP2',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G/outputs/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 14,
                't0obs': datetime(2017, 8, 3, 11, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            {
                'tag': 'PF1 (Manning x 1.1) ADCP3',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G/outputs/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 15,
                't0obs': datetime(2017, 8, 2, 12, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                't_end': datetime(2017,8,29, 22,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            # {
            #     'tag': 'PF1 (Manning x 1.2) ADCP2',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
            #     'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputsMCx1.2/outputs_run/diagnostic_detectors-adcp.hdf5'),
            #     'idx_col': 14,
            #     't0obs': datetime(2017, 8, 3, 11, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            # {
            #     'tag': 'PF1 (Manning x 1.2) ADCP3',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
            #     'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/outputsMCx1.2/outputs_run/diagnostic_detectors-adcp.hdf5'),
            #     'idx_col': 15,
            #     't0obs': datetime(2017, 8, 2, 12, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     't_end': datetime(2017,8,29, 22,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            # {
            #     'tag': 'Baseline', # 'PF0 (base Manning) ADCP2',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
            #     'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/outputs_MCx1/outputs_run/diagnostic_detectors-adcp.hdf5'),
            #     'idx_col': 14,
            #     't0obs': datetime(2017, 8, 3, 11, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            # {
            #     'tag': 'PF0 (base Manning) ADCP3',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
            #     'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/outputs_MCx1/outputs_run/diagnostic_detectors-adcp.hdf5'),
            #     'idx_col': 15,
            #     't0obs': datetime(2017, 8, 2, 12, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     't_end': datetime(2017,8,29, 22,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            {
                'tag': 'PF0 x 1.1', # 'PF1 (Manning x 1.1) ADCP2',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF0-N/outputs/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 14,
                't0obs': datetime(2017, 8, 3, 11, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            {
                'tag': 'PF0 (Manning x 1.1) ADCP3',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF0-N/outputs/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 15,
                't0obs': datetime(2017, 8, 2, 12, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                't_end': datetime(2017,8,29, 22,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            # {
            #     'tag': 'PF0 x 1.2', # 'PF1 (Manning x 1.2) ADCP2',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
            #     'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/outputsMCx1.2/outputs_run/diag_detectors.hdf5'),
            #     'idx_col': 14,
            #     't0obs': datetime(2017, 8, 3, 11, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            # {
            #     'tag': 'PF0 (Manning x 1.2) ADCP3',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
            #     'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_PF1_assessment/src/outputsMCx1.2/outputs_run/diag_detectors.hdf5'),
            #     'idx_col': 15,
            #     't0obs': datetime(2017, 8, 2, 12, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     't_end': datetime(2017,8,29, 22,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            # {
            #     'tag': 'PF2 (base Manning) ADCP2',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
            #     'file_sim': os.path.expanduser('~/Documents/diagnostic_detectors-adcp_20170801.hdf5'),
            #     'idx_col': 12,
            #     't0obs': datetime(2017, 8, 2, 12, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     'dtobs': timedelta(seconds=100),
            #     'dtsim': timedelta(seconds=100),
            #     'shift': 0,
            # },
            # {
            #     'tag': 'PF2 (base Manning) ADCP3',
            #     'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
            #     'file_sim': os.path.expanduser('~/Documents/diagnostic_detectors-adcp_20170801.hdf5'),
            #     'idx_col': 13,
            #     't0obs': datetime(2017, 8, 2, 12, 0, 50),
            #     't0sim': datetime(2017,8,1,0,0,0),
            #     't_end': datetime(2017,8,29, 22,0,0),
            # },
            {
                'tag': 'PF2 (Manning x 1.1) ADCP2',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_2.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF2-N_wrong_manning_map/outputs/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 12,
                't0obs': datetime(2017, 8, 2, 12, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                'dtobs': timedelta(seconds=100),
                'dtsim': timedelta(seconds=100),
                'shift': 0,
            },
            {
                'tag': 'PF2 (Manning x 1.1) ADCP3',
                'file_obs': os.path.expanduser('~/GitHub-repo/TSSM_PF0_demo/src/validation/vel_ADCP_3.csv'),
                'file_sim': os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF2-N_wrong_manning_map/outputs/diagnostic_detectors-adcp.hdf5'),
                'idx_col': 13,
                't0obs': datetime(2017, 8, 2, 12, 0, 50),
                't0sim': datetime(2017,8,1,0,0,0),
                't_end': datetime(2017,8,29, 22,0,0),
            },
            ]
    Tobs_all = []   # to aggregate
    velobs_all = []  # to aggregate
    Tcomp_all = []   # to aggregate
    velcomp_all = []  # to aggregate
    dirobs_all = []  # to aggregate
    dircomp_all = []  # to aggregate
    histobs_all = []  # to aggregate
    histcomp_all = []  # to aggregate
    MIN_VEL = 1  # cut-in speed
    MAX_VEL = 999  # cut-out speed, if relevant

    for info in cases:
        tag = info['tag']
        file_obs = info['file_obs']
        file_sim = info['file_sim']
        t0obs = info['t0obs']
        t0sim = info['t0sim']
        t_end = info.get('t_end', None)
        dtobs = info.get('dtobs', timedelta(seconds=100))
        dtsim = info.get('dtsim', timedelta(seconds=100))
        shift = info.get('shift', 0)
        print(f"\nProcessing {tag}... starting at {t0sim}... shifted by {shift} hours")

        # Ensure correct data file path when running on VS Code debug mode
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is None:
            # discard the path (consider current folder)
            path, file_obs = os.path.split(file_obs)
            path, file_sim = os.path.split(file_sim)
        
        table_obs = pd.read_csv(file_obs, skiprows=1, header=None)
        # Data tables now excluding time column, which can be generated by Python
        velobs = table_obs[0].dropna()
        Uobs = table_obs[1].dropna()
        Vobs = table_obs[2].dropna()

        table_sim = hdf5_detector_to_TS(file_sim, info['idx_col'])
        # Col 0: time, Col 1: Z
        Usim = table_sim[:,2]
        Vsim = table_sim[:,3]
        
        # Trim the series based on the end time
        if t_end is not None:
            nt_obs = (t_end - t0obs) // dtobs
            nt_sim = (t_end - t0sim) // dtsim
            Uobs = Uobs[:(nt_obs + 1)]
            Vobs = Vobs[:(nt_obs + 1)]
            velobs = velobs[:(nt_obs + 1)]
            Usim = Usim[:(nt_sim + 1)]
            Vsim = Vsim[:(nt_sim + 1)]

        # Since the time series are not properly generated 
        # (round-off errors in spreadsheet, improper formatting, etc.)
        # we will manually generate the simulation time series.

        # auto generated time series with a constant interval = timestep
        Tobs = [t0obs + i*dtobs for i in range(len(Uobs))]
        Tsim = [t0sim + i*dtsim for i in range(len(Usim))]
        
        velsim = np.sqrt(Usim**2 + Vsim**2)  # velocity magnitude (m/s)
        # Flow direction follows the Cartesian convention
        dirsim = np.degrees(np.arctan2(Vsim, Usim)) % 360  # velocity direction (deg.)
        dirobs = np.degrees(np.arctan2(Vobs, Uobs)) % 360  # velocity direction (deg.)

        # Calculate goodness-of-fit for simulated data series
        Tcomp, (velcomp, dircomp), idx_trim_start = trim_series(Tobs, Tsim, velsim, 
                                                                dirsim, shift=shift)
        # print(f"Trim index {idx_trim_start}/{len(Tcomp)} for velocity series.")

        # Split flood and ebb phases
        # Unused? Such separation is in `calc_goodness_of_fit()`
        # idx_flood = np.where((dirobs > 310) & (dirobs < 340))[0]
        # idx_ebb = np.where((dirobs > 145) & (dirobs < 205))[0]
        # idx_slack = np.where( ((dirobs >= 205) & (dirobs <= 310)) | (dirobs >= 340) | (dirobs <= 145)) [0]

        print("Quantity\tRMS/MAE\tNRMSE\tR2\tBias")
        gof = calc_goodness_of_fit(Tobs, velobs, dirobs,
                                    Tcomp, velcomp, dircomp, idx_trim_start)
        
        print(f"Vel. flood.:\t{gof['flood_mag']['RMSE']:.3f}  {gof['flood_mag']['NRMSE']:.3f}  {gof['flood_mag']['R2']:.3f}")
        print(f"Vel. ebb.:\t{gof['ebb_mag']['RMSE']:.3f}  {gof['ebb_mag']['NRMSE']:.3f}  {gof['ebb_mag']['R2']:.3f}")
        print(f"Vel. slack.:\t{gof['mag']['RMSE']:.3f}  {gof['mag']['NRMSE']:.3f}  {gof['mag']['R2']:.3f}")
        print(f"Dir. flood.:\t{gof['flood_dir']['MAE']:.3f}  {gof['flood_dir']['NRMSE']:.3f}  -----  {gof['flood_dir']['Bias']:.3f}")
        print(f"Dir. ebb.:\t{gof['ebb_dir']['MAE']:.3f}  {gof['ebb_dir']['NRMSE']:.3f}  -----  {gof['ebb_dir']['Bias']:.3f}")
        print(f"Dir. slack.:\t{gof['dir']['MAE']:.3f}  {gof['dir']['NRMSE']:.3f}  -----  {gof['dir']['Bias']:.3f}")
        Tobs_all.append(Tobs)
        velobs_all.append(velobs)
        Tcomp_all.append(Tcomp)
        velcomp_all.append(velcomp)
        dirobs_all.append(dirobs)
        dircomp_all.append(dircomp)

        if tag in ['PF0 x 1.1', 'PF1 (Manning x 1.1) ADCP2', 'PF2 (Manning x 1.1) ADCP2']:
            metrics = comp_peaks(Tobs, velobs, Tcomp, velcomp,
                                 plot=False, show_title=False, tag=tag, series=['Obs', tag[:3]])
            (R2, R2flood, R2ebb), (NRMSE, NRMSE_flood, NRMSE_ebb) = metrics
            print("Peak velocity comparison:")
            print("Quantity\tR2\tRMSE")
            print(f"Peak:\t{R2:.3f}\t{NRMSE:.3f}")
            print(f"Flood:\t{R2flood:.3f}\t{NRMSE_flood:.3f}")
            print(f"Ebb:\t{R2ebb:.3f}\t{NRMSE_ebb:.3f}")
            # chisq, p = chi_sq(velcomp, velobs, # mind the order!
            #                   show_plot=False, tag=tag, series=['Obs', tag[:3]])
            # print(f"Chi-squared: {chisq:.3f}, p = {p:.3f}")
            d, p, *rest = ks(velcomp, velobs, # mind the order!
                            show_plot=False, tag=tag, series=['Obs', tag[:3]])
            print(f"K-S stat: {d:.3f}, p = {p:.3f}")

    
        if tag in ['PF0 (Manning x 1.1) ADCP3', 'PF1 (Manning x 1.1) ADCP3', 'PF2 (Manning x 1.1) ADCP3']:
            velcomp_all.append(velcomp)
            velobs_all.append(velobs)

    show_KS = False
    if show_KS:
        mpl.rcParams['font.size'] = 16
        plt.figure(figsize=(7, 6))
        ax = plt.gca()
        n_bins = 25
        bin_seq = np.linspace(0, max(velobs_all[0]), n_bins)
        plt.hist(velobs_all[0], bins=bin_seq,
                    density=True, histtype='step', cumulative=True, 
                    linewidth=2, linestyle='dashed', label=f"Obs ADCP3")
        for seq, tag in zip(velcomp_all, ('PF0-N', 'PF1-G', 'PF2-N')):
            plt.hist(seq, bins=bin_seq,
                        density=True, histtype='step', cumulative=True, label=tag)
        # plt.hist(seq2, bins=bin_seq, density=True, histtype='step', cumulative=True, label=f"{series[1]}")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Cumulative Probability")
        # plt.title(f"Histogram of vel. {tag}")
        ax.grid(True)
        ax.legend(loc='best')
        # plt.savefig(os.path.expanduser(f"~/Documents/hist/cdf_{tag}.pdf"))
        plt.show()

    # comp_peaks(Tobs_all[0], velobs_all[0], Tcomp_all[0], velcomp_all[0], series=['Obs', 'PF0'])
    # comp_peaks(Tobs_all[6], velobs_all[6], Tcomp_all[6], velcomp_all[6], series=['Obs', 'PF1'])

    # Only for time-series plots
    # Swapping blue and orange in the default color cycle and assign back to matplotlib
    # https://stackoverflow.com/a/9398214
    # import matplotlib as mpl
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    #     color=['#ff7f0e', '#1f77b4', '#2ca02c',
    #             '#d62728', '#9467bd', '#8c564b',
    #             '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    # mpl.rcParams['font.size'] = 12

    # plot((Tobs_all[0], velobs_all[0]),
    #     (Tcomp_all[0], velcomp_all[0]),
    #     (Tcomp_all[1], velcomp_all[1]),
    #     (Tcomp_all[2], velcomp_all[2]),
    #     labels = ['Baseline', 'x 1.1', 'x 1.2'],
    #     plottype='mag'
    #     )

    # plot((Tobs_all[0], dirobs_all[0]),
    #     (Tcomp_all[0], dircomp_all[0]),
    #     (Tcomp_all[1], dircomp_all[1]),
    #     (Tcomp_all[2], dircomp_all[2]),
    #     labels = ['Baseline', 'x 1.1', 'x 1.2'],
    #     plottype='dir'
    #     )
