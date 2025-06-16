# Filename: 'val_tools.py'
# Date: 14/08/2023
# Author: Connor Jordan
# Institution: University of Edinburgh (IIE)
# Contains plotting and analysis tools to simplify the main validation scripts.

import h5py
import pandas as pd
import numpy as np
import datetime as datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mhkit import tidal
import uptide


def load_thetis_data(detectors_file, start_date, key='ADCP2022', elev_only=False, vel_only=False):
    """
    Load Thetis data from a HDF5 file and return a DataFrame.

    Parameters:
    detectors_file (str): Path to the HDF5 file containing Thetis data.
    start_date (str): Start date of the data in the format 'YYYY-MM-DDTHH:MM:SS'.
    key (str): Key to access the data within the HDF5 file.
    elev_only (bool): If True, returns only the elevation.
    vel_only (bool): If True, returns only the velocity.

    Returns:
    df_data (pd.DataFrame): DataFrame containing Thetis data.
    """
    df = h5py.File(detectors_file, 'r')

    model_eta = df[key][:, 0]
    model_u = df[key][:, 1]
    model_v = df[key][:, 2]
    t_array = np.array(df['time'][:, -1])  # should be in terms of timestep t = 100s

    uv_mag = np.sqrt(model_u ** 2 + model_v ** 2)
    model_theta = (np.degrees(np.arctan2(model_u, model_v)) + 360) % 360
    # check with simple data - arctan2 does not do what the documentation says...

    time_array = start_date + t_array.astype('timedelta64[s]')

    if elev_only:
        data = {"t": time_array, "eta": model_eta, "theta": model_theta}
    elif vel_only:
        data = {"t": time_array, "u": model_u, "v": model_v, "uv": uv_mag, "theta": model_theta}
    else:
        data = {"t": time_array, "eta": model_eta, "u": model_u, "v": model_v, "uv": uv_mag, "theta": model_theta}

    df_data = pd.DataFrame(data)

    return df_data


def apply_velocity_corrections(vel_dir, t_dt, correction_before, correction_after, correction_date):
    """
    Apply velocity corrections to data based on a given date.

    Parameters:
    vel_dir (np.ndarray): Array of burst-averaged unsigned current magnitude [burst_time x bin_number].
    t_dt (list): List of datetime.datetime objects corresponding to the times in vel_dir.
    correction_before (float): Correction to be applied before the correction_date.
    correction_after (float): Correction to be applied after the correction_date.
    correction_date (datetime.datetime): Date to apply the corrections.

    Returns:
    None. Modifies vel_dir array in place.
    """
    for i in range(vel_dir.shape[1]):
        if t_dt[i] < correction_date:
            mask = ~np.isnan(vel_dir[:, i])
            vel_dir[:, i][mask] += correction_before
        else:
            mask = ~np.isnan(vel_dir[:, i])
            vel_dir[:, i][mask] += correction_after

    return vel_dir


def load_ADCP_2017_data(ADCP_file, data_start_time, start_cut_date=None, end_cut_date=None):
    """
    Load ADCP 2017 data and process it into a dataframe.

    Parameters:
    ADCP_file (str): Path to the ADCP data file.
    data_start_time (np.datetime64): Start time for the data.
    start_cut_date (np.datetime64, optional): Starting cutoff date.
    end_cut_date (np.datetime64, optional): Finishing cutoff date.

    Returns:
    meas_data (pd.DataFrame): Dataframe containing processed ADCP data.
    """
    SMADCP_data = h5py.File(ADCP_file, 'r')

    t = np.array(SMADCP_data['mTime'][:][0], dtype=np.float64)  # Time of each sample on Matlab Time format
    t_dt = np.array([data_start_time + datetime.timedelta(days=days) for days in t - t[0]])

    # vel_mag = SMADCP_data['Speed'][:]  # Speed based on east, north and up velocity vectors
    # vel_dir = SMADCP_data['Direction'][:]  # Based on east, north and up velocities vectors
    #
    # depth_averaged_vel = np.nanmean(vel_mag, axis=0)
    # depth_averaged_vel_dir = np.nanmean(vel_dir, axis=0)
    # vel_dir_radians = np.radians(depth_averaged_vel_dir)

    u_meas = SMADCP_data['Velocity_E'][:]
    v_meas = SMADCP_data['Velocity_N'][:]
    w_meas = SMADCP_data['Velocity_U'][:]

    # u_meas = depth_averaged_vel * np.sin(vel_dir_radians)
    # v_meas = depth_averaged_vel * np.cos(vel_dir_radians)

    u_meas = np.nanmean(u_meas, axis=0)
    v_meas = np.nanmean(v_meas, axis=0)
    w_meas = np.nanmean(w_meas, axis=0)

    nan_u = np.isnan(u_meas)
    nan_v = np.isnan(v_meas)
    nan_w = np.isnan(w_meas)
    u_meas[nan_u] = np.interp(np.flatnonzero(nan_u), np.flatnonzero(~nan_u), u_meas[~nan_u])
    v_meas[nan_v] = np.interp(np.flatnonzero(nan_v), np.flatnonzero(~nan_v), v_meas[~nan_v])
    w_meas[nan_w] = np.interp(np.flatnonzero(nan_w), np.flatnonzero(~nan_w), v_meas[~nan_w])
    dir_meas = (np.degrees(np.arctan2(u_meas, v_meas)) + 360) % 360
    uv_meas = np.sqrt(u_meas**2 + v_meas**2 + w_meas**2)

    meas_data = pd.DataFrame({"t": t_dt, "u": u_meas, "v": v_meas, "uv": uv_meas, "theta": dir_meas})

    # Apply cutoff dates if provided
    if start_cut_date is not None:
        meas_data = meas_data[meas_data['t'] >= start_cut_date]
    if end_cut_date is not None:
        meas_data = meas_data[meas_data['t'] <= end_cut_date]

    return meas_data


def load_ADCP_2022_data(ADCP_file, data_start_time, correction_before, correction_after, correction_date,
                        start_cut_date=None, end_cut_date=None):
    """
    Load ADCP 2022 data and process it into a dataframe.

    Parameters:
    ADCP_file (str): Path to the ADCP data file.
    data_start_time (np.datetime64): Start time for the data.
    correction_before (float): Correction to be applied before the correction_date.
    correction_after (float): Correction to be applied after the correction_date.
    correction_date (np.datetime64): Date to apply the corrections.
    start_cut_date (np.datetime64, optional): Starting cutoff date.
    end_cut_date (np.datetime64, optional): Finishing cutoff date.

    Returns:
    meas_data (pd.DataFrame): Dataframe containing processed ADCP data.
    """
    SMADCP_data = h5py.File(ADCP_file, 'r')
    data = SMADCP_data['CURRENT_DO']

    t = np.array(data['mean_mtime'][:][0], dtype=np.float64)  # Burst centre time in Matlab time (UTC) [burst_time]
    t_dt = np.array([data_start_time + datetime.timedelta(days=days) for days in t - t[0]])

    vel_mag = data['mean_mag'][:]  # Burst-averaged current direction [burst_time x bin_number]
    vel_dir = data['mean_dir'][:]  # Burst-averaged unsigned current magnitude [burst_time x bin_number]
    eta = np.squeeze(data['mean_depth'][:])  # Burst-averaged depth [burst_time]

    # make corrections to the velocity direction as required
    apply_velocity_corrections(vel_dir, t_dt, correction_before, correction_after, correction_date)

    depth_averaged_vel = np.nanmean(vel_mag, axis=0)
    depth_averaged_vel_dir = np.nanmean(vel_dir, axis=0)
    vel_dir_radians = np.radians(depth_averaged_vel_dir)

    u_meas = depth_averaged_vel * np.sin(vel_dir_radians)
    v_meas = depth_averaged_vel * np.cos(vel_dir_radians)

    # Replace NaN values with interpolated values
    nan_u = np.isnan(u_meas)
    nan_v = np.isnan(v_meas)
    u_meas[nan_u] = np.interp(np.flatnonzero(nan_u), np.flatnonzero(~nan_u), u_meas[~nan_u])
    v_meas[nan_v] = np.interp(np.flatnonzero(nan_v), np.flatnonzero(~nan_v), v_meas[~nan_v])
    dir_meas = (np.degrees(np.arctan2(u_meas, v_meas)) + 360) % 360
    uv_meas = np.sqrt(u_meas**2 + v_meas**2)

    meas_data = pd.DataFrame({"t": t_dt, "eta": eta, "u": u_meas, "v": v_meas, "uv": uv_meas, "theta": dir_meas})

    # Apply cutoff dates if provided
    if start_cut_date is not None:
        meas_data = meas_data[meas_data['t'] >= start_cut_date]
    if end_cut_date is not None:
        meas_data = meas_data[meas_data['t'] <= end_cut_date]

    return meas_data


def plot_xy(x_arrays, y_arrays, labels=None, **kwargs):
    """
    Plot XY plot for multiple datasets.

    Parameters:
    x_arrays (list of list): List of arrays x-axis values for each dataset.
    y_arrays (list of list): List of arrays y-axis values for each dataset.
    labels (list or None, optional): List of labels for each dataset.
    **kwargs: Additional keyword arguments for customization.
        title (str, optional): Title for the plot.
        x_label (str, optional): X-label for the plot.
        y_label (str, optional): Y-label for the plot.
        xlim (tuple, optional): Tuple containing x-limits (e.g., (xmin, xmax)).
        ylim (tuple, optional): Tuple containing y-limits (e.g., (ymin, ymax)).
        xlabel_size (int, optional): Font size for x-label.
        ylabel_size (int, optional): Font size for y-label.
        tick_label_size (int, optional): Font size for x- and y-ticks.
        legend (bool, optional): Whether to show legend.
        legend_size (int, optional): Font size for legend.
        x_is_time (Bool, optional): Changes datetime formatting to DD/MM if x-data is time.
        split_intervals (Bool, optional): Removes lines between no data periods.
        max_intervals (list of float, optional): Maximum intervals to split into segments for
                                                 each dataset.

    Returns:
    None
    """
    plt.figure(figsize=kwargs.get('figsize', (8, 6)))

    if kwargs.get('x_is_time', True):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    xlim_mins, xlim_maxs, ylim_mins, ylim_maxs = [], [], [], []

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(x_arrays)):
        color = str(color_cycle[i % len(color_cycle)])
        if kwargs.get('split_intervals', True):
            max_intervals = kwargs.get('max_intervals', [100] * len(x_arrays))
            x_data = x_arrays[i]
            y_data = y_arrays[i]

            # Split data into segments based on x intervals
            segments_x = []
            segments_y = []
            current_segment_x = [x_data[0]]
            current_segment_y = [y_data[0]]
            for j in range(1, len(x_data)):
                interval = (x_data[j] - x_data[j - 1])
                if interval > max_intervals[i]:
                    segments_x.append(current_segment_x)
                    segments_y.append(current_segment_y)
                    current_segment_x = []
                    current_segment_y = []
                current_segment_x.append(x_data[j])
                current_segment_y.append(y_data[j])
            if current_segment_x:  # Append the last segment
                segments_x.append(current_segment_x)
                segments_y.append(current_segment_y)

            # Plot each segment separately
            for k, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
                if k == 0:
                    plt.plot(seg_x, seg_y, label=labels[i] if labels else None, color=color)
                else:
                    plt.plot(seg_x, seg_y, color=color)
        else:
            plt.plot(x_arrays[i], y_arrays[i], label=labels[i] if labels else None)
        xlim_mins.append(np.min(x_arrays[i]))
        xlim_maxs.append(np.max(x_arrays[i]))
        ylim_mins.append(np.min(y_arrays[i]))
        ylim_maxs.append(np.max(y_arrays[i]))

    if 'xlim' in kwargs:
        xmin, xmax = kwargs['xlim']
    else:
        delta = (np.min(xlim_maxs) - np.max(xlim_mins)) / 40
        xmin, xmax = np.max(xlim_mins) - delta, np.min(xlim_maxs) + delta

    if 'ylim' in kwargs:
        ymin, ymax = kwargs['ylim']
    else:
        delta = (np.min(ylim_maxs) - np.max(ylim_mins)) / 40
        ymin, ymax = np.max(ylim_mins) - delta, np.min(ylim_maxs) + delta

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tick_params(axis='both', labelsize=kwargs.get('tick_label_size', plt.rcParams['xtick.labelsize']))

    plt.title(kwargs.get('title', ''))
    plt.xlabel(kwargs.get('x_label', 'X'), fontsize=kwargs.get('xlabel_size', plt.rcParams['axes.labelsize']))
    plt.ylabel(kwargs.get('y_label', 'Y'), fontsize=kwargs.get('ylabel_size', plt.rcParams['axes.labelsize']))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    legend_enabled = kwargs.get('legend', True)
    if legend_enabled:
        plt.legend(fontsize=kwargs.get('legend_size', plt.rcParams['legend.fontsize']))

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_vel_dir_variation(time_arrays, uv_arrays, theta_arrays, dataset_labels=None, **kwargs):
    """
    Plot variation of velocity components over time.

    Parameters:
    time_arrays (list of list): List of time arrays for each dataset.
    uv_arrays (list of list): List of velocity magnitude arrays for each dataset.
    theta_arrays (list of list): List of flow direction arrays for each dataset.
    dataset_labels (list or None, optional): List of labels for each dataset.
    **kwargs: Additional keyword arguments for customization.
        ylabel_uv (str, optional): Y-label for velocity magnitude plot.
        ylabel_theta (str, optional): Y-label for direction plot.
        xlabel (str, optional): X-label for the plot.
        title (str, optional): Title for the plot.
        xlim (tuple, optional): Tuple containing x-limits (e.g., (xmin, xmax)).
        ulim (tuple, optional): Tuple containing y-limits for the speed
                                (e.g., (umin, umax)).
        thetalim (tuple, optional): Tuple containing y-limits for direction (e.g.,
                                    (thetamin, thetamax)).
        xlabel_size (int, optional): Font size for x-label.
        ylabel_size (int, optional): Font size for y-label.
        tick_label_size (int, optional): Font size for x- and y-ticks.
        title_size (int, optional): Font size for title.
        legend (bool, optional): Whether to show legend.
        legend_size (int, optional): Font size for legend.

    Returns:
    None
    """
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(18, 10))

    xlim_mins, xlim_maxs, u_minmaxs, v_minmaxs, theta_minmaxs = [], [], [], [], []
    for i in range(len(time_arrays)):
        xlim_mins.append(np.min(time_arrays[i]))
        xlim_maxs.append(np.max(time_arrays[i]))
        u_minmaxs.append(np.array([np.min(uv_arrays[i]), np.max(uv_arrays[i])]))
        theta_minmaxs.append(np.array([np.min(theta_arrays[i]), np.max(theta_arrays[i])]))

    if dataset_labels:
        for i in range(len(time_arrays)):
            axs[0].plot(time_arrays[i], uv_arrays[i], label=dataset_labels[i])
            axs[1].plot(time_arrays[i], theta_arrays[i], label=dataset_labels[i])
    else:
        for i in range(len(time_arrays)):
            axs[0].plot(time_arrays[i], uv_arrays[i])
            axs[1].plot(time_arrays[i], theta_arrays[i])

    if 'xlim' in kwargs:
        xmin, xmax = kwargs['xlim']
    else:
        delta = (np.min(xlim_maxs) - np.max(xlim_mins)) / 40
        xmin, xmax = np.max(xlim_mins) - delta, np.min(xlim_maxs) + delta

    if 'ulim' in kwargs:
        u_min, u_max = kwargs['ulim']
    else:
        u_min, u_max = np.min(u_minmaxs), np.max(u_minmaxs)
        delta = (u_max - u_min) / 40
        u_min, u_max = u_min - delta, u_max + delta
    axs[0].set_ylim([u_min, u_max])

    if 'thetalim' in kwargs:
        theta_min, theta_max = kwargs['thetalim']
    else:
        theta_min, theta_max = np.min(theta_minmaxs), np.max(theta_minmaxs)
        delta = (theta_max - theta_min) / 40
        theta_min, theta_max = np.min(theta_min - delta, 0), np.max(theta_max + delta, 360)
    axs[1].set_ylim([theta_min, theta_max])

    for i in np.arange(0, 2, 1):
        axs[i].set_xlim([xmin, xmax])
        axs[i].tick_params(axis='both', labelsize=kwargs.get('tick_label_size', plt.rcParams['xtick.labelsize']))

    axs[0].set_ylabel(kwargs.get('ylabel_uv', 'Speed (m/s)'),
                      fontsize=kwargs.get('ylabel_size', plt.rcParams['axes.labelsize']))
    axs[1].set_ylabel(kwargs.get('ylabel_theta', 'Tidal Direction (degrees)'),
                      fontsize=kwargs.get('ylabel_size', plt.rcParams['axes.labelsize']))
    axs[1].set_xlabel(kwargs.get('xlabel', 'Time'), fontsize=kwargs.get('xlabel_size', plt.rcParams['axes.labelsize']))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    plt.xticks(rotation=0)

    title = kwargs.get('title', 'Velocity Variation')
    fig.suptitle(title, fontsize=kwargs.get('title_size', plt.rcParams['axes.titlesize']))

    legend_enabled = kwargs.get('legend', True)
    if dataset_labels and legend_enabled:
        plt.legend(loc='upper center', bbox_to_anchor=(0.77, -0.15), ncol=len(dataset_labels),
                   fontsize=kwargs.get('legend_size', plt.rcParams['legend.fontsize']))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    plt.close()


def plot_velocity_variation(time_arrays, u_arrays, v_arrays, theta_arrays, dataset_labels=None, **kwargs):
    """
    Plot variation of velocity components over time.

    Parameters:
    time_arrays (list of list): List of time arrays for each dataset.
    u_arrays (list of list): List of horizontal velocity component arrays for each dataset.
    v_arrays (list of list): List of vertical velocity component arrays for each dataset.
    theta_arrays (list of list): List of flow direction arrays for each dataset.
    dataset_labels (list or None, optional): List of labels for each dataset.
    **kwargs: Additional keyword arguments for customization.
        ylabel_u (str, optional): Y-label for horizontal velocity component plot.
        ylabel_v (str, optional): Y-label for vertical velocity component plot.
        ylabel_theta (str, optional): Y-label for direction plot.
        xlabel (str, optional): X-label for the plot.
        title (str, optional): Title for the plot.
        xlim (tuple, optional): Tuple containing x-limits (e.g., (xmin, xmax)).
        ulim (tuple, optional): Tuple containing y-limits for the x-direction of velocity
                                (e.g., (umin, umax)).
        vlim (tuple, optional): Tuple containing y-limits for the y-direction of velocity
                                (e.g., (vmin, vmax)).
        thetalim (tuple, optional): Tuple containing y-limits for direction (e.g.,
                                    (thetamin, thetamax)).
        xlabel_size (int, optional): Font size for x-label.
        ylabel_size (int, optional): Font size for y-label.
        tick_label_size (int, optional): Font size for x- and y-ticks.
        title_size (int, optional): Font size for title.
        legend (bool, optional): Whether to show legend.
        legend_size (int, optional): Font size for legend.

    Returns:
    None
    """
    fig, axs = plt.subplots(3, 1, sharex='all', figsize=(18, 10))

    xlim_mins, xlim_maxs, u_minmaxs, v_minmaxs, theta_minmaxs = [], [], [], [], []
    for i in range(len(time_arrays)):
        xlim_mins.append(np.min(time_arrays[i]))
        xlim_maxs.append(np.max(time_arrays[i]))
        u_minmaxs.append(np.array([np.min(u_arrays[i]), np.max(u_arrays[i])]))
        v_minmaxs.append(np.array([np.min(v_arrays[i]), np.max(v_arrays[i])]))
        theta_minmaxs.append(np.array([np.min(theta_arrays[i]), np.max(theta_arrays[i])]))

    if dataset_labels:
        for i in range(len(time_arrays)):
            axs[0].plot(time_arrays[i], u_arrays[i], label=dataset_labels[i])
            axs[1].plot(time_arrays[i], v_arrays[i], label=dataset_labels[i])
            axs[2].plot(time_arrays[i], theta_arrays[i], label=dataset_labels[i])
    else:
        for i in range(len(time_arrays)):
            axs[0].plot(time_arrays[i], u_arrays[i])
            axs[1].plot(time_arrays[i], v_arrays[i])
            axs[2].plot(time_arrays[i], theta_arrays[i])

    if 'xlim' in kwargs:
        xmin, xmax = kwargs['xlim']
    else:
        delta = (np.min(xlim_maxs) - np.max(xlim_mins)) / 40
        xmin, xmax = np.max(xlim_mins) - delta, np.min(xlim_maxs) + delta

    if 'ulim' in kwargs:
        u_min, u_max = kwargs['ulim']
    else:
        u_min, u_max = np.min(u_minmaxs), np.max(u_minmaxs)
        delta = (u_max - u_min) / 40
        u_min, u_max = u_min - delta, u_max + delta
    axs[0].set_ylim([u_min, u_max])

    if 'vlim' in kwargs:
        v_min, v_max = kwargs['vlim']
    else:
        v_min, v_max = np.min(v_minmaxs), np.max(v_minmaxs)
        delta = (v_max - v_min) / 40
        v_min, v_max = v_min - delta, v_max + delta
    axs[1].set_ylim([v_min, v_max])

    if 'thetalim' in kwargs:
        theta_min, theta_max = kwargs['thetalim']
    else:
        theta_min, theta_max = np.min(theta_minmaxs), np.max(theta_minmaxs)
        delta = (theta_max - theta_min) / 40
        theta_min, theta_max = np.min(theta_min - delta, 0), np.max(theta_max + delta, 360)
    axs[2].set_ylim([theta_min, theta_max])

    for i in np.arange(0, 3, 1):
        axs[i].set_xlim([xmin, xmax])
        axs[i].tick_params(axis='both', labelsize=kwargs.get('tick_label_size', plt.rcParams['xtick.labelsize']))

    axs[0].set_ylabel(kwargs.get('ylabel_u', 'x-\n component\n of velocity\n (m/s)'),
                      fontsize=kwargs.get('ylabel_size', plt.rcParams['axes.labelsize']))
    axs[1].set_ylabel(kwargs.get('ylabel_v', 'y-\n component\n of velocity\n (m/s)'),
                      fontsize=kwargs.get('ylabel_size', plt.rcParams['axes.labelsize']))
    axs[2].set_ylabel(kwargs.get('ylabel_theta', 'Tidal\n Direction\n (degrees)'),
                      fontsize=kwargs.get('ylabel_size', plt.rcParams['axes.labelsize']))
    axs[2].set_xlabel(kwargs.get('xlabel', 'Time'), fontsize=kwargs.get('xlabel_size', plt.rcParams['axes.labelsize']))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    plt.xticks(rotation=0)

    title = kwargs.get('title', 'Velocity Variation')
    fig.suptitle(title, fontsize=kwargs.get('title_size', plt.rcParams['axes.titlesize']))

    legend_enabled = kwargs.get('legend', True)
    if dataset_labels and legend_enabled:
        plt.legend(loc='upper center', bbox_to_anchor=(0.8, -0.2), ncol=len(dataset_labels),
                   fontsize=kwargs.get('legend_size', plt.rcParams['legend.fontsize']))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    plt.close()


def plot_velocity_magnitude(time_array, velocity_magnitude, **kwargs):
    """
    Plot velocity magnitude over time.

    Parameters:
    time_array (list): Time values for the x-axis.
    velocity_magnitude (list): Velocity magnitude data.
    **kwargs: Additional keyword arguments for customization.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        title (str, optional): Title for the plot.

    Returns:
    None
    """
    fig, axs = plt.subplots(1, 1, figsize=(20, 8))
    axs.plot(time_array, velocity_magnitude)
    axs.set_xlabel(kwargs.get('xlabel', 'Time'))
    axs.set_ylabel(kwargs.get('ylabel', r'Velocity magnitude, $\left|\sqrt{u^2 + v^2}\right|$ (m/s)'))

    title = kwargs.get('title', 'Velocity')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_velocity_in_direction(theta_data, velocity_data, direction, **kwargs):
    """
    Plot velocity in specified direction over time.

    Parameters:
    theta_data (list): Tidal direction angle data.
    velocity_data (list): Velocity magnitude data.
    direction (float): Direction in degrees for which to plot the velocity.
    **kwargs: Additional keyword arguments for customization.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        title (str, optional): Title for the plot.

    Returns:
    None
    """
    ax = tidal.graphics.plot_current_timeseries(theta_data, velocity_data, direction)
    ax.set_xlabel(kwargs.get('xlabel', 'Time'))
    ax.set_ylabel(kwargs.get('ylabel', 'Velocity'))
    title = kwargs.get('title', f'Velocity in direction: {np.round(direction, 2)}°')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def joint_probability_distribution(theta, uv, dir_bin_width, vel_bin_width, flood=None, ebb=None, **kwargs):
    """
    Plot joint probability distribution rose.

    Parameters:
    theta (list): Tidal direction angle data.
    uv (list): Velocity magnitude data.
    dir_bin_width (float): Width of directional bins for histogram in degrees.
    vel_bin_width (float): Width of velocity bins for histogram in m/s.
    flood (float, optional): Flood direction in degrees.
    ebb (float, optional): Ebb direction in degrees.
    **kwargs: Additional keyword arguments for customization.
        title (str, optional): Title for the plot.
        label_size (int, optional): Font size for x- and y-labels.
        tick_label_size (int, optional): Font size for x- and y-ticks.
        title_size (int, optional): Font size for title.
        colorbar_size (int, optional): Font size for colorbar.

    Returns:
    None
    """
    # note I have modified tidal for this to work... you can't access colorbar properties from ax
    ax = tidal.graphics.plot_joint_probability_distribution(theta, uv, dir_bin_width, vel_bin_width,
                                                            flood=flood, ebb=ebb,
                                                            colorbar_size=kwargs.get('colorbar_size', None))
    ax.tick_params(axis='both', labelsize=kwargs.get('tick_label_size', plt.rcParams['xtick.labelsize']))
    y_lim = kwargs.get('y_upper_lim', np.max(uv))
    ax.set_ylim(0, y_lim)
    y_ticks = ax.get_yticks()
    y_ticks = [f'{y:.1f} $m/s$' for y in y_ticks]
    ax.set_yticklabels(y_ticks, fontsize=kwargs.get('label_size', plt.rcParams['axes.labelsize']))
    title = kwargs.get('title', 'Joint probability distribution rose')
    plt.title(title, fontsize=kwargs.get('title_size', plt.rcParams['axes.titlesize']))
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_comparison_subplots(flood_vels_list, flood_eps, ebb_vels_list, ebb_eps, flood_r2, ebb_r2, flood_labels,
                             ebb_labels, **kwargs):
    """
    Plot comparison subplots for modelled and measured data.

    Parameters:
    flood_vels_list (list): List of modelled and measured flood velocity data series (final entry is measured).
    flood_eps (array-like): Modelled and measured flood exceedance probability data.
    ebb_vels_list (list): List of modelled and measured ebb velocity data series (final entry is measured).
    ebb_eps (array-like): Modelled and measured ebb exceedance probability data.
    flood_r2 (float): R-squared value for flood data.
    ebb_r2 (float): R-squared value for ebb data.
    flood_labels (list): Labels for flood data.
    ebb_labels (list): Labels for ebb data.
    **kwargs: Additional keyword arguments for customization.
        - colors (list): List of colors for each dataset.
        - linestyles (list): List of linestyles for each dataset.
        - ylabel_dist (float, optional): Location of y label for second column.

    Returns:
    None
    """
    colors = kwargs.get('colors', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyles = kwargs.get('linestyles', ['-']*len(colors))

    vel_lim = max(np.max(ebb_vels_list), np.max(flood_vels_list))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Plot data on subplots for flood data
    for i, flood_vels in enumerate(flood_vels_list):
        axs[0, 0].plot(flood_vels, flood_eps[i], label=flood_labels[i], color=colors[i], linestyle=linestyles[i])
    axs[0, 0].legend(frameon=False)
    axs[0, 0].set_title(kwargs.get('column0title', 'Velocity exceedance probability'))
    axs[0, 0].set_xlim([-0.1, vel_lim + 0.1])
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot data on subplots for flood velocity comparison
    for i in range(len(flood_vels_list)-1):
        axs[0, 1].plot(flood_vels_list[i], flood_vels_list[-1], label=r'$R^2 = {:.3f}$'.format(flood_r2[i]),
                       color=colors[i],  linestyle=linestyles[i])
    axs[0, 1].plot([0, vel_lim], [0, vel_lim], c='black', ls='dashdot', lw=0.5)
    axs[0, 1].legend(frameon=False)
    axs[0, 1].set_title(kwargs.get('column1title', 'Velocity magnitude comparison'))
    axs[0, 1].set_xlim([-0.1, vel_lim + 0.1])
    axs[0, 1].set_ylim([-0.1, vel_lim + 0.1])
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot data on subplots for ebb data
    for i, ebb_vels in enumerate(ebb_vels_list):
        axs[1, 0].plot(ebb_vels, ebb_eps[i], label=ebb_labels[i], color=colors[i], linestyle=linestyles[i])
    axs[1, 0].legend(frameon=False)
    axs[1, 0].set_xlim([-0.1, vel_lim + 0.1])
    axs[1, 0].set_xlabel(kwargs.get('xlabel0', r'Velocity magnitude, $|\textbf{u}|$ [m/s]'))

    # Plot data on subplots for ebb velocity comparison
    for i in range(len(ebb_vels_list)-1):
        axs[1, 1].plot(ebb_vels_list[i], ebb_vels_list[-1], label=r'$R^2 = {:.3f}$'.format(ebb_r2[i]), color=colors[i],
                       linestyle=linestyles[i])
    axs[1, 1].plot([0, vel_lim], [0, vel_lim], c='black', ls='dashdot', lw=0.5)
    axs[1, 1].legend(frameon=False)
    axs[1, 1].set_xlabel(kwargs.get('xlabel1', r'Modelled velocity magnitude values, $|\textbf{u}|$ [m/s]'))
    axs[1, 1].set_xlim([-0.1, vel_lim + 0.1])
    axs[1, 1].set_ylim([-0.1, vel_lim + 0.1])

    fig.subplots_adjust(hspace=0, wspace=0.2)

    if plt.rcParams['text.usetex']:
        fig.text(0.04, 0.5, r'Exceedance probability (\%)', va='center', rotation='vertical',
                 fontsize=plt.rcParams['axes.labelsize'])
    else:
        fig.text(0.04, 0.5, 'Exceedance probability (%)', va='center', rotation='vertical',
                 fontsize=plt.rcParams['axes.labelsize'])
    fig.text(kwargs.get('ylabel_dist', 0.49), 0.5, kwargs.get('ylabel1', r'Measured velocity magnitude values, $|\textbf{u}|$ [m/s]'), va='center',
             rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
    plt.show()
    plt.close()


def calculate_principal_flow_directions(theta_data, bin_width, expected_flood_direction, expected_ebb_direction):
    """
    Calculate and print principal flow directions, flood, and ebb directions. North = 0 degrees.

    Parameters:
    theta_data (list): Tidal direction angle data.
    bin_width (float): Histogram bin width for directions to calculate the principal flow directions in degrees.
    expected_flood_direction (float): Expected flood direction in degrees [0-360].
    expected_ebb_direction (float): Expected ebb direction in degrees [0-360].

    Returns:
    ebb (float): Principal ebb direction in degrees.
    flood (float): Principal flood direction in degrees.
    """
    # Calculate principal flow directions
    direction1, direction2 = tidal.resource.principal_flow_directions(theta_data, bin_width)

    # Set flood and ebb directions based on site knowledge
    dir_diff = abs(expected_ebb_direction - expected_flood_direction)
    if expected_flood_direction > expected_ebb_direction:
        transition_direction = expected_ebb_direction + dir_diff/2
        if (direction1 % 360) < transition_direction:
            ebb = (direction1 + 360) % 360  # (dir + 360) % 360 makes sure we stay between 0 -> 360
            flood = (direction2 + 360) % 360
        else:
            ebb = (direction2 + 360) % 360
            flood = (direction1 + 360) % 360
    else:
        transition_direction = expected_flood_direction + dir_diff/2
        if (direction1 % 360) < transition_direction:
            flood = (direction1 + 360) % 360
            ebb = (direction2 + 360) % 360
        else:
            flood = (direction2 + 360) % 360
            ebb = (direction1 + 360) % 360

    return ebb, flood


def split_flood_ebb(df_data, ebb, flood):
    """
    Split a dataset into flood and ebb based on specified ebb and flood directions.

    Parameters:
    df_data (pandas.DataFrame): Input dataset with 'theta' column containing direction data.
    ebb (float): Ebb direction in degrees [0-360].
    flood (float): Flood direction in degrees [0-360].

    Returns:
    df_flood (pandas.DataFrame): DataFrame containing data points classified as flood.
    df_ebb (pandas.DataFrame): DataFrame containing data points classified as ebb.
    """
    if (ebb + 0.5 * (flood - ebb)) > 170:
        high_split = ebb + 0.5 * (flood - ebb)
        low_split = high_split - 180
    else:
        low_split = ebb + 0.5 * (flood - ebb)
        high_split = low_split + 180

    df_flood = df_data[(df_data['theta'] > low_split) & (df_data['theta'] < high_split)]
    df_ebb = df_data[(df_data['theta'] < low_split) | (df_data['theta'] > high_split)]

    return df_flood, df_ebb


def calculate_ep(dataframe):
    """
    Calculate exceedance probability, sort the values, and return the sorted DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing velocity and time data.

    Returns:
    df_ep_data (pandas.DataFrame): Sorted DataFrame with calculated exceedance probability.
    """
    ep_data = {"uv": dataframe.uv}
    df_ep_data = pd.DataFrame(ep_data)
    df_ep_data.index = dataframe.t
    df_ep_data['EP'] = tidal.resource.exceedance_probability(df_ep_data)
    df_ep_data.sort_values('EP', ascending=True, kind='mergesort', inplace=True)

    return df_ep_data


def calculate_ep_CF(dataframe, time_column, power_column):
    """
    Calculate exceedance probability, sort the values, and return the sorted DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing velocity and time data.

    Returns:
    df_ep_data (pandas.DataFrame): Sorted DataFrame with calculated exceedance probability.
    """
    ep_data = {"CF": dataframe[power_column]}
    df_ep_data = pd.DataFrame(ep_data)
    df_ep_data.index = dataframe[time_column]
    df_ep_data['EP'] = tidal.resource.exceedance_probability(df_ep_data)
    df_ep_data.sort_values('EP', ascending=True, kind='mergesort', inplace=True)

    return df_ep_data


def print_exceedance_probabilities(uv_values, velocities, exceedance_probs):
    """
    Print exceedance probabilities for given velocity values using an interpolation function.

    Parameters:
    uv_values (list): List of velocity values to compute exceedance probabilities for.
    interp_func (callable): Interpolation function for velocity and exceedance probability.

    Returns:
    None
    """
    interp_func = interp1d(velocities, exceedance_probs, kind='linear', fill_value='extrapolate')
    ep_values = interp_func(uv_values)

    for i, uv in enumerate(uv_values):
        print(f'EP at u={uv:.2f}m/s: {ep_values[i]:.2f}%')
    print(f'u_mean: {velocities.mean():.3f} m/s\n')


def print_exceedance_probabilities_CF(CF_values, CFs, exceedance_probs):
    """
    Print exceedance probabilities for given velocity values using an interpolation function.

    Parameters:
    CF_values (list): List of CF values to compute exceedance probabilities for.
    interp_func (callable): Interpolation function for velocity and exceedance probability.

    Returns:
    None
    """
    interp_func = interp1d(CFs, exceedance_probs, kind='linear', fill_value='extrapolate')
    ep_values = interp_func(CF_values)

    for i, uv in enumerate(CF_values):
        print(f'EP at CF={uv:.2f}%: {ep_values[i]:.2f}%')
    print(f'CF_mean: {CFs.mean():.3f} %\n')


def trim_dataframes_to_common_time_range(df1, df2, time_column):
    """
    Trims two dataframes to have the same time range based on a specified time column
    and adds a new column representing time since the new start date.

    Parameters:
    df1 (pd.DataFrame): First dataframe.
    df2 (pd.DataFrame): Second dataframe.
    time_column (str): Name of the time column in both dataframes.

    Returns:
    pd.DataFrame: Trimmed and modified df1.
    pd.DataFrame: Trimmed and modified df2.
    """

    # Find the latest start date
    initial_time = max(df1[time_column].min(), df2[time_column].min())

    # Find the earliest end date
    final_time = min(df1[time_column].max(), df2[time_column].max())

    # Trim both dataframes to the same time range
    trimmed_df1 = df1[(df1[time_column] >= initial_time) & (df1[time_column] <= final_time)].copy()
    trimmed_df2 = df2[(df2[time_column] >= initial_time) & (df2[time_column] <= final_time)].copy()

    # Calculate time since the new start date
    trimmed_df1["time_since_start"] = (trimmed_df1[time_column] - initial_time).dt.total_seconds()
    trimmed_df2["time_since_start"] = (trimmed_df2[time_column] - initial_time).dt.total_seconds()

    return trimmed_df1, trimmed_df2


def analyze_tidal_constituents(dataframe, tide, analysis_parameter):
    """
    Analyzes tidal constituents using the provided dataframe and tide object.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the analysis parameter and 'time_since_start' columns.
    tide: Uptide tide object for harmonic analysis.
    analysis_parameter (str): The parameter for analysis, as named in the pandas dataframe.

    Returns:
    np.ndarray: Tidal constituent amplitudes.
    np.ndarray: Tidal constituent phases.
    """

    model_parameter = np.array(dataframe[analysis_parameter])
    model_time = np.array(dataframe["time_since_start"])

    amplitudes, phases = uptide.analysis.harmonic_analysis(tide, model_parameter, model_time)
    phases = np.remainder(phases, 2 * np.pi) * 360 / (2 * np.pi)

    return amplitudes, phases


def calculate_circle_slice_areas(radius_, num_slices_):
    """
    Calculate the areas of slices in a circle divided into equal depths.

    Parameters:
        radius_ (float): The radius of the circle.
        num_slices_ (int): The number of equal slices to divide the circle into.

    Returns:
        list of float: A list containing the areas of each slice in the circle,
        with elements representing the areas of the slices from top to bottom.

    Note:
        - See https://mathworld.wolfram.com/CircularSegment.html for formulas.

    """
    segment_areas_ = []
    slice_depth_ = 2 * radius_ / num_slices_
    for i_ in range(int(num_slices_ // 2)):
        r_ = radius_ - (i_ + 1) * slice_depth_  # distance to chord from circle centre
        theta = 2 * np.arccos(r_ / radius_)  # angle between chord edges
        area_segment = 0.5 * (radius_ ** 2) * (theta - np.sin(theta))  # segment area
        if i_ == 0:
            segment_areas_.append(area_segment)
        else:
            segment_areas_.append(area_segment - np.sum(segment_areas_))  # split into slices

    if num_slices_ % 2 == 0:
        full_circle_areas = segment_areas_ + segment_areas_[::-1]
    else:
        center_area = np.pi * (radius_ ** 2) - 2 * np.sum(segment_areas_)  # Area of the central slice
        full_circle_areas = segment_areas_ + [center_area] + segment_areas_[::-1]

    return full_circle_areas


def find_index(value, list_of_values):
    return np.digitize(value, list_of_values, right=True) - 1


def retrieve_alpha_beta(u, theta, low_split_, high_split_):
    ebb_vels = np.array([0, 1, 2, 3])
    flood_vels = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    alpha_ebb = np.array([4.399, 4.554, 4.613, 4.641])
    beta_ebb = np.array([0.383, 0.383, 0.383, 0.383])
    alpha_flood = np.array([9, 3.667, 3.451, 3.828, 4.029, 4.091, 3.793, 3.565, 3.542])
    beta_flood = np.array([0.382, 0.385, 0.386, 0.384, 0.383, 0.383, 0.384, 0.386, 0.385])

    if len(alpha_ebb) < len(alpha_flood):
        alpha_ebb = np.pad(alpha_ebb, (0, len(alpha_flood) - len(alpha_ebb)), 'constant')
        beta_ebb = np.pad(beta_ebb, (0, len(beta_flood) - len(beta_ebb)), 'constant')
    elif len(alpha_flood) < len(alpha_ebb):
        alpha_flood = np.pad(alpha_flood, (0, len(alpha_ebb) - len(alpha_flood)), 'constant')
        beta_flood = np.pad(beta_flood, (0, len(beta_ebb) - len(beta_flood)), 'constant')

    flood_condition = np.logical_and(low_split_ < theta, theta < high_split_)

    index_ = np.where(flood_condition, find_index(u, flood_vels), find_index(u, ebb_vels))

    alpha = alpha_flood[index_] * flood_condition + alpha_ebb[index_] * ~flood_condition
    beta = beta_flood[index_] * flood_condition + beta_ebb[index_] * ~flood_condition

    return alpha, beta


def power_law(z, h, U_bar, theta, low_split, high_split):
    alpha, beta = retrieve_alpha_beta(U_bar, theta, low_split, high_split)

    profile = (z[:, np.newaxis] / (beta * h)) ** (1/alpha) * U_bar

    return profile.T


def da_vel_to_rotor_avg(params, turb_area, seg_depths, seg_areas):
    h, U_bar, theta, low_split, high_split = params

    vel_3D = power_law(seg_depths, h, U_bar, theta, low_split, high_split)

    rotor_avg = np.cbrt(np.sum((1 / turb_area) * vel_3D ** 3 * seg_areas, axis=1))

    condition = vel_3D[:, 5] > 4.8
    rotor_avg[condition] = 0

    return rotor_avg


def split_directions(ebb, flood):
    """
    Determine flood-ebb transition hyperplane.
    Flood sits within low_split < theta < high_split regardless of ebb/flood directions.

    Parameters:
    ebb (float): Ebb direction in degrees [0-360].
    flood (float): Flood direction in degrees [0-360].

    Returns:
    low_split (float): Low transition direction in degrees [0-360].
    high_split (float): High transition direction in degrees [0-360].
    """
    if (ebb + 0.5 * (flood - ebb)) > 180:
        high_split = ebb + 0.5 * (flood - ebb)
        low_split = high_split - 180
    else:
        low_split = ebb + 0.5 * (flood - ebb)
        high_split = low_split + 180

    return low_split, high_split


def get_thrust_coefficient(speed, speeds, thrust_coefficients):
    """
    Get the thrust coefficient for the given speed by interpolation.

    Parameters:
        speed (float or array-like): The speed for which you want to find the thrust coefficient.
        speeds (array-like): An array of speeds at which thrust coefficients are given.
        thrust_coefficients (array-like): An array of thrust coefficients corresponding to the speeds.

    Returns:
        float: The interpolated thrust coefficient for the given speed.
    """
    return np.interp(speed, speeds, thrust_coefficients)
