
# Plot the difference in field variables
from thetis import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Load common mesh
# mesh2d = Mesh(os.path.expanduser("inputs/PF1G.msh"))  # will encounter error AttributeError: 'MeshTopology' object has no attribute 'sfXC'
# or load from a checkpoint file
dir_ramp = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G_visc_5/preproc/ramp')
with CheckpointFile(os.path.join(dir_ramp, "ramp.h5"), 'r') as h5f_ramp:
    mesh2d = h5f_ramp.load_mesh()

# Compare at 2017-08-05T00:00, i.e. t = 5 days, expect ebb tide
# Load base case
dir_res_base = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G_nM_10/outputs/hdf5')
with CheckpointFile(os.path.join(dir_res_base, "Elevation2d_00005.h5"), 'r') as h5f_base_elev:
    elev_base = h5f_base_elev.load_function(mesh2d, name="elev_2d")

# Load modified case
dir_res_modf = os.path.expanduser('~/GitHub-repo/TSSM_unify_main/TSSM/src/case_PF1-G_visc_5/outputs/hdf5')
with CheckpointFile(os.path.join(dir_res_modf, "Elevation2d_00005.h5"), 'r') as h5f_modf_elev:
    elev_modf = h5f_modf_elev.load_function(mesh2d, name="elev_2d")

# Calculate difference
# diff_elev = elev_modf - elev_base
fs = get_functionspace(mesh2d, "DG", 1)
diff_elev = Function(fs, name="diff_elev").assign(elev_modf - elev_base)


# Plot config
mpl.rcParams['font.size'] = 14
X_LL, Y_LL = 489500, 6500800
X_EXT, Y_EXT = 5200, 2600

# Plot base
fig = plt.figure(figsize=(6,3))
# levels = np.linspace(0, 1.5E-4, 11) # np.linspace(0, 1E-2, 11)  # np.linspace(0, 1.5E-4, 13)
axes = fig.gca()
contours = tricontourf(elev_base, axes=axes, # levels=levels, 
                       cmap="Blues")  # afmhot_r
triplot(mesh2d, axes=axes,
        interior_kw={"linewidths":0.05,"edgecolors":'0.1','alpha':0.5})
axes.set_xlim([X_LL, X_LL + X_EXT])
axes.set_ylim([Y_LL, Y_LL + Y_EXT])
axes.set_aspect("equal")
cbar = fig.colorbar(contours, label='Base. Elev., m', ax=axes, extend='max')
# cbar.formatter.set_powerlimits((0, 0))  # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.ScalarFormatter.set_powerlimits 
# Scaling axes, https://stackoverflow.com/a/70435827/
offset_x = lambda x, _: f'{(x - X_LL)/1000:g}'
offset_y = lambda y, _: f'{(y - Y_LL)/1000:g}'
axes.xaxis.set_major_formatter(offset_x)
axes.yaxis.set_major_formatter(offset_y)
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Documents/elev_base.png"))
plt.show()


# Plot modf
fig = plt.figure(figsize=(6,4))
# levels = np.linspace(0, 1.5E-4, 11) # np.linspace(0, 1E-2, 11)  # np.linspace(0, 1.5E-4, 13)
axes = fig.gca()
contours = tricontourf(elev_modf, axes=axes, # levels=levels, 
                       cmap="Blues")  # afmhot_r
triplot(mesh2d, axes=axes,
        interior_kw={"linewidths":0.05,"edgecolors":'0.1','alpha':0.5})
axes.set_xlim([X_LL, X_LL + X_EXT])
axes.set_ylim([Y_LL, Y_LL + Y_EXT])
axes.set_aspect("equal")
cbar = fig.colorbar(contours, label='Modf. Elev., m', ax=axes, extend='max')
# cbar.formatter.set_powerlimits((0, 0))  # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.ScalarFormatter.set_powerlimits 
# Scaling axes, https://stackoverflow.com/a/70435827/
offset_x = lambda x, _: f'{(x - X_LL)/1000:g}'
offset_y = lambda y, _: f'{(y - Y_LL)/1000:g}'
axes.xaxis.set_major_formatter(offset_x)
axes.yaxis.set_major_formatter(offset_y)
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Documents/elev_visc5.png"))
plt.show()


# Plot difference
fig = plt.figure(figsize=(6,4))
axes = fig.gca()
contours = tricontourf(diff_elev, axes=axes, 
                       levels = np.linspace(-0.1, 0.1, 21),
                       norm=colors.Normalize(vmin=-0.1, vmax=0.1),
                       cmap="PiYG")
triplot(mesh2d, axes=axes,
        interior_kw={"linewidths":0.05,"edgecolors":'0.1','alpha':0.5})
axes.set_xlim([X_LL, X_LL + X_EXT])
axes.set_ylim([Y_LL, Y_LL + Y_EXT])
axes.set_aspect("equal")
cbar = fig.colorbar(contours, label='Diff. Elev., m', ax=axes, extend='max')
# cbar.formatter.set_powerlimits((0, 0))  # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.ScalarFormatter.set_powerlimits 
# Scaling axes, https://stackoverflow.com/a/70435827/
offset_x = lambda x, _: f'{(x - X_LL)/1000:g}'
offset_y = lambda y, _: f'{(y - Y_LL)/1000:g}'
axes.xaxis.set_major_formatter(offset_x)
axes.yaxis.set_major_formatter(offset_y)
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Documents/elev_diff_visc5.png"))
plt.show()
