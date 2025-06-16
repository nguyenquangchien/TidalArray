import os
import numpy as np
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
# Not newest version, refer to TSSM main branch

WIN_SMOOTH = 36
P_C = 1570  # kW, installed capacity
RHO = 1026  # kg/m3, seawater density  ## TODO: multiply this....
# Not the newest results - check the cases from _powr_cap
path_to_file_sim0 = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF0-G_nM_10/out_DT/powers.csv')
path_to_file_sim1 = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G_nM_11_Emils/out_DT/powers.csv')
path_to_file_sim2 = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF2-G_nM_10_hotstart/out_DT/powers.csv')
path_to_file_sim0 = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF0-G_visc_5/out_DT/powers.csv')
# path_to_file_sim1 = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G_debug_pow_cap/out_DT/powers.csv')
path_to_file_sim2 = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF2-G_visc_5/out_DT/powers.csv')
P_sim0 = np.genfromtxt(path_to_file_sim0, delimiter=',')
P_sim1 = np.genfromtxt(path_to_file_sim1, delimiter=',')
P_sim2 = np.genfromtxt(path_to_file_sim2, delimiter=',')
length_sim0 = P_sim0.shape[0]
length_sim1 = P_sim1.shape[0]
length_sim2 = P_sim2.shape[0]

path_to_file_obs = os.path.expanduser('~/Downloads/turbine_power_Meygen/turb_power_Aug2017.npy')
P_obs = np.load(path_to_file_obs, allow_pickle=True)

length_obs = P_obs.shape[0]
dt_sim = timedelta(seconds=100)
T_sim0 = [datetime(2017,8,1,0,0,0) + i * dt_sim for i in range(length_sim0)]
T_sim1 = [datetime(2017,8,1,0,0,0) + i * dt_sim for i in range(length_sim1)]
T_sim2 = [datetime(2017,8,1,0,0,0) + i * dt_sim for i in range(length_sim2)]
T_obs = P_obs[::WIN_SMOOTH,0]

# Reversing and right align to make the time series effectively left-aligned
# Credit: bluenote10 on Stack Overflow 2021
P2_smoothed = pd.Series(P_obs[:,2])[::-1].rolling(WIN_SMOOTH).mean()[::-1]
P3_smoothed = pd.Series(P_obs[:,3])[::-1].rolling(WIN_SMOOTH).mean()[::-1]
P2_smoothed[P2_smoothed == 0] = np.nan
P3_smoothed[P3_smoothed == 0] = np.nan

T_shift = [t - timedelta(hours=1) for t in T_obs]  # accounting for time saver UK

Pow_total_PF0 = P_sim0[:,0]
Pow_total_PF1 = P_sim1[:,0] + P_sim1[:,1] + P_sim1[:,2] + P_sim1[:,3]
Pow_total_PF2 = P_sim2[:,0] + P_sim2[:,1] + P_sim2[:,2] + P_sim2[:,3]

# plt.figure()
# plt.plot(T_sim, P_sim[:,2], label='T2_sim')
# plt.plot(T_obs, P2_smoothed[::WIN_SMOOTH], label='T2_obs', marker='o', markersize=4,
#                      markerfacecolor='none', alpha=0.5, linestyle='none')
# plt.gca().set_xlim(datetime(2017,8,17,0,0,0), datetime(2017,8,24,0,0,0))
# plt.grid(True); plt.legend(loc='best'); plt.ylabel('Power, kW'); plt.show()

# plt.figure()
# plt.plot(T_sim, P_sim[:,3], label='T3_sim')
# plt.plot(T_obs, P3_smoothed[::WIN_SMOOTH], label='T3_obs', marker='o', markersize=4,
#                      markerfacecolor='none', alpha=0.5, linestyle='none')
# plt.gca().set_xlim(datetime(2017,8,17,0,0,0), datetime(2017,8,24,0,0,0))
# plt.grid(True); plt.legend(loc='best'); plt.ylabel('Power, kW'); plt.show()

# Try subplots
mpl.rcParams['font.size'] = 12
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex='col', figsize=(12, 8))

ax1.set_title('(b) Turbine T2', loc='left', fontsize=12)
ax1.plot(T_sim1, P_sim1[:,1]/P_C * 100, label='PF1')
ax1.plot(T_sim2, P_sim2[:,1]/P_C * 100, label='PF2')
ax1.plot(T_shift, P2_smoothed[::WIN_SMOOTH]/P_C * 100, label='Obs', 
         marker='o', markersize=3, markerfacecolor='none', alpha=0.5, linestyle='none')
ax1.grid(True); # ax1.legend(loc=(0.1, 1.0))
ax1.get_xaxis().set_ticklabels([])  # hide labels for upper subplot

ax2.set_title('(c) Turbine T3', loc='left', fontsize=12)
ax2.plot(T_sim1, P_sim1[:,2]/P_C * 100, label='PF1')
ax2.plot(T_sim2, P_sim2[:,2]/P_C * 100, label='PF2')
ax2.plot(T_shift, P3_smoothed[::WIN_SMOOTH]/P_C * 100, label='Obs',
         marker='o', markersize=3, markerfacecolor='none', alpha=0.5, linestyle='none')
ax2.grid(True); ax2.legend(ncol=3, loc=(0.66, 1.01))
ax2.get_xaxis().set_ticklabels([])  # hide labels for upper subplot

for a in (ax1, ax2):
    # a.set_xlim(datetime(2017,8,17,0,0,0), datetime(2017,8,24,0,0,0))
    # a.set_xlim(datetime(2017,8,17,0,0,0), datetime(2017,8,20,0,0,0))  # make clearer
    a.set_xlim(datetime(2017,8,4,0,0,0), datetime(2017,8,7,0,0,0))  # make clearer
    a.set_ylabel(r'$C_F$, %')

ax0.set_title('(a) Turbine group of 4, with monthly yield', loc='left', fontsize=12)
ax0.plot(T_sim0, Pow_total_PF0 / 1000, label='PF0:1930 MWh', color='red')
ax0.plot(T_sim1, Pow_total_PF1 / 1000, label='PF1:1652 MWh')
ax0.plot(T_sim2, Pow_total_PF2 / 1000, label='PF2:1806 MWh')
ax0.grid(True); ax0.legend(ncol=3, loc=(0.40, 1.01))
ax0.set_ylabel(r'$P$, MW')
# ax0.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
myFmt = mdates.DateFormatter('%d/%m %H:%M')
ax0.xaxis.set_major_formatter(myFmt)

plt.show()

# Estimate power yield - convert to MWh
print('Total Yield PF0:', np.trapz(Pow_total_PF0) / 36 / 1000)  # 36: 1 hour has 36 timesteps
print('Total Yield PF1:', np.trapz(Pow_total_PF1) / 36 / 1000)
print('Total Yield PF2:', np.trapz(Pow_total_PF2) / 36 / 1000)

# exit(0)

# plt.figure()
# # plt.plot(range(length_sim), P_sim[:,3], label='T4_sim') 
# plt.plot(range(length_obs), P_obs[:,3], label='T4_obs')
# plt.legend(loc='best'); plt.show()
