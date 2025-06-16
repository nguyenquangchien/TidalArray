import numpy as np
from thetis import *

import warnings
warnings.simplefilter("ignore")

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

mpl.rcParams['font.size'] = 12
X_LL, Y_LL = 489500, 6500800
X_EXT, Y_EXT = 5200, 2600

DX, DY = 100, 100
NX = X_EXT // DX
NY = Y_EXT // DY

# Rectangle mesh for interpolate quiver plot
mesh_rect = RectangleMesh(NX, NY, X_LL+NX*DX, Y_LL+NY*DY, X_LL, Y_LL)
vfs_rect = VectorFunctionSpace(mesh_rect, 'CG', 1)


def map_show_compact(t, field, mesh_, show_mesh=False, hatch_outlier=True,
             colormap="PiYG", clabel='', 
             title_="", savefile=None, **options):
    mpl.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(4,2))
    plt.subplots_adjust(left=0.05, right=0.8)
    axes = fig.gca()

    if hatch_outlier:
        cmap = mpl.colormaps[colormap]
        COLOR_MAX = cmap(0.95)
        axes.add_patch(Polygon([(X_LL+3200, Y_LL+2000), (X_LL+3200, Y_LL+2500), (X_LL+4000, Y_LL+2500), (X_LL+4300, Y_LL+2600), 
                                (X_LL+5200, Y_LL+2600), (X_LL+5200, Y_LL+0), (X_LL+1000, Y_LL+0), (X_LL+1000, Y_LL+2600), (X_LL+1800, Y_LL+2600), (X_LL+2500, Y_LL+2000), 
                               ],
                                     facecolor=COLOR_MAX))

    contours = tricontourf(field, axes=axes, cmap=colormap, **options)
    tricontour(field, axes=axes, colors='gray', linewidths=0.1, **options)
    #contours.cmap.set_over('r')
    #contours.set_clim(-80,80)  # temp
    if title_ is not None:
        plt.annotate(title_, xy=(X_LL + 200, Y_LL + 200), ha='center', va='center')
    if show_mesh:
        triplot(mesh_, axes=axes,
                interior_kw={"linewidths":0.05,"edgecolors":'0.1','alpha':0.1})
    axes.set_xlim([X_LL, X_LL + X_EXT])
    axes.set_ylim([Y_LL, Y_LL + Y_EXT])
    axes.set_aspect("equal")
    axes.set_xticks([X_LL + offset for offset in [1000, 2000, 3000, 4000, 5000] ])
    axes.set_yticks([Y_LL + offset for offset in [1000, 2000] ])
    cax = fig.add_axes([axes.get_position().x1+0.01,axes.get_position().y0,0.02,axes.get_position().height]) # https://stackoverflow.com/a/56900830
    # cax = fig.add_axes([axes.get_position().x1+0.05,axes.get_position().y0,0.02,axes.get_position().height])
    cbar = fig.colorbar(contours, label=clabel, cax=cax)
    offset_x = lambda x, _: f'{(x - X_LL)/1000:g}'
    offset_y = lambda y, _: f'{(y - Y_LL)/1000:g}'
    axes.xaxis.set_major_formatter(offset_x)
    axes.yaxis.set_major_formatter(offset_y)
    axes.tick_params(axis="y",direction="in", pad=-12)  # https://stackoverflow.com/a/47874059
    axes.tick_params(axis="x",direction="in", pad=-15)
    
    if type(savefile) is str:
        plt.savefig(savefile)
    plt.show()
    
    
# Compact plots for making matrices, quiver plot from field_and_vector
def quiver_compact(t, field, vector_field, mesh_, show_mesh=False, hatch_outlier=True,
             colormap="PiYG", clabel='', clims=(-80,80), format_num='%.1E', contourline=False,
             refveclen=1, veclegend=r'$10^6 J m/s$', arrowwidth=0.05,
             vecposx=0.5, vecposy=0.9, coordinates='figure',
             title_="", savefile=None, **options):
    mpl.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(4,2))
    plt.subplots_adjust(left=0.05, right=0.8)
    axes = fig.gca()
    if hatch_outlier:
        cmap = matplotlib.cm.get_cmap('RdBu_r'); COLOR_MAX = cmap(0.95)
        axes.add_patch(Polygon([(X_LL+3200, Y_LL+2000), (X_LL+3200, Y_LL+2500), (X_LL+4000, Y_LL+2500), (X_LL+4300, Y_LL+2600), 
                                (X_LL+5200, Y_LL+2600), (X_LL+5200, Y_LL+0), (X_LL+1000, Y_LL+0), (X_LL+1000, Y_LL+2600), (X_LL+1800, Y_LL+2600), (X_LL+2500, Y_LL+2000), 
                               ],
                                     facecolor=COLOR_MAX))
    contours = tricontourf(field, axes=axes, cmap=colormap, **options)
    if contourline:
        tricontour(field, axes=axes, colors='gray', linewidths=0.05, **options)
    vecs = firedrake.pyplot.quiver(vector_field, axes=axes, color='k', linewidths=arrowwidth)  # cannot do  **options here
    axes.quiverkey(vecs, vecposx, vecposy, 
                   refveclen, veclegend, labelpos='E',)

    if title_ is not None:
        plt.annotate(title_, xy=(X_LL + 200, Y_LL + 200), ha='center', va='center')
    if show_mesh:
        triplot(mesh_, axes=axes,
                interior_kw={"linewidths":0.05,"edgecolors":'0.1','alpha':0.1})
    axes.set_xlim([X_LL, X_LL + X_EXT])
    axes.set_ylim([Y_LL, Y_LL + Y_EXT])
    axes.set_aspect("equal")
    axes.set_xticks([X_LL + offset for offset in [1000, 2000, 3000, 4000, 5000] ])
    axes.set_yticks([Y_LL + offset for offset in [1000, 2000] ])
    cax = fig.add_axes([axes.get_position().x1+0.01,axes.get_position().y0,0.02,axes.get_position().height]) # https://stackoverflow.com/a/56900830
    # cax = fig.add_axes([axes.get_position().x1+0.05,axes.get_position().y0,0.02,axes.get_position().height])
    cbar = fig.colorbar(contours, label=clabel, cax=cax)
    offset_x = lambda x, _: f'{(x - X_LL)/1000:g}'
    offset_y = lambda y, _: f'{(y - Y_LL)/1000:g}'
    axes.xaxis.set_major_formatter(offset_x)
    axes.yaxis.set_major_formatter(offset_y)
    axes.tick_params(axis="y",direction="in", pad=-12)  # https://stackoverflow.com/a/47874059
    axes.tick_params(axis="x",direction="in", pad=-15)
    # contours.set_clim(clims)
    if type(savefile) is str:
        plt.savefig(savefile)
    plt.show()


def compare_and_plot(baseline, modified, 
                     folder='figout',
                     quantity='U',
                     scenario='pilot',
                     subfig='(a)',
                     fformat='pdf',
                     levels=np.linspace(-6, 6, 13),
                    ):
    """ Plot and save files (*.png) separately for field difference between
        baseline and modified scenarios, both flood and ebb stages.

        :param baseline: name of baseline case (also name of folder) (`str`)
        :param modified: name of modified case (also name of folder) (`str`)
        :param folder: name of the output folder to store image (`str`)
        :param quantity: name of quantity (to include in file name) (`str`)
        :param scenario: name of scenario (to include in file name) (`str`)
        :param subfig: e.g. (a), (b), (c), (d), to annotate in figures (`str`)
        :param fformat: export file format (`'png'`, `'pdf'`) (`str`)
        :param levels: list of values for color bar (`list int`)
    """ 
    # Flood phase
    start_fld, stop_fld = 1, 19  # layer indices
    for step in range(start_fld, stop_fld):
        with CheckpointFile(f'{baseline}/outputs/hdf5/Velocity2d_{step:05d}.h5', 'r') as h5f_amb_velo:
            if step == start_fld:
                mesh2d = h5f_amb_velo.load_mesh()
                fs = get_functionspace(mesh2d, "CG", 1)
                # mesh_show(mesh2d)
                # return
                
            velo_amb = h5f_amb_velo.load_function(mesh2d, name="velocity")
            uu_amb = velo_amb.sub(0)
            vv_amb = velo_amb.sub(1)
            if step == start_fld:
                vel_mag_max = (uu_amb**2 + vv_amb**2)**0.5
                vel_mag_avg = (uu_amb**2 + vv_amb**2)**0.5
            else:
                vel_mag = (uu_amb**2 + vv_amb**2)**0.5
                vel_mag_max = conditional(gt(vel_mag, vel_mag_max), vel_mag, vel_mag_max)
                vel_mag_avg = vel_mag_avg + vel_mag
    
    # vel_mag_max = conditional(gt(vel_mag, vel_mag_max), vel_mag, vel_mag_max)
    vel_mag_peak_amb = Function(fs, name="vel_peak_fld_base").interpolate(vel_mag_max)
    vel_mag_mean_amb = Function(fs, name="vel_mean_fld_base").interpolate(vel_mag_avg / (stop_fld - start_fld + 1))
    
    for step in range(start_fld, stop_fld):
        with CheckpointFile(f'{modified}/outputs/hdf5/Velocity2d_{step:05d}.h5', 'r') as h5f_grp_velo:                
            velo_grp = h5f_grp_velo.load_function(mesh2d, name="velocity")
            uu_grp = velo_grp.sub(0)
            vv_grp = velo_grp.sub(1)
            if step == start_fld:
                vel_mag_max = (uu_grp**2 + vv_grp**2)**0.5
                vel_mag_avg = (uu_grp**2 + vv_grp**2)**0.5
            else:
                vel_mag = (uu_grp**2 + vv_grp**2)**0.5
                vel_mag_max = conditional(gt(vel_mag, vel_mag_max), vel_mag, vel_mag_max)
                vel_mag_avg = vel_mag_avg + vel_mag
    
    vel_mag_peak_grp = Function(fs, name="vel_peak_fld_modf").interpolate(vel_mag_max)
    vel_mag_mean_grp = Function(fs, name="vel_mean_fld_modf").interpolate(vel_mag_avg / (stop_fld - start_fld + 1))
    
    vmag_peak_reldiff = (vel_mag_peak_grp - vel_mag_peak_amb)/ vel_mag_peak_amb * 100
    # v_clamped = conditional(gt(vmag_peak_reldiff, 80), 80, vmag_peak_reldiff)
    velmag_peak_reldiff = Function(fs, name="vel_peak_fld_diff").interpolate(vmag_peak_reldiff)
    velmag_mean_reldiff = Function(fs, name="vel_mean_fld_diff").interpolate((vel_mag_mean_grp - vel_mag_mean_amb)/ vel_mag_mean_amb * 100)
    # velmag_peak_reldiff = conditional(gt(velmag_peak_reldiff, 80), velmag_peak_reldiff, 80)

    map_show_compact(0, velmag_peak_reldiff, mesh2d, colormap="RdBu_r", 
                     # clabel=r"$\Delta {|\mathbf{u}|_\max} \, / \, {|\mathbf{u}|_\max}$, %", 
                     title_=subfig, levels=levels,
                     savefile=os.path.join(folder, f'{quantity}_max_diff_{scenario}_fld.{fformat}')
            )
    map_show_compact(0, velmag_mean_reldiff, mesh2d, colormap="RdBu_r", 
                     # clabel=r"$\Delta \overline{|\mathbf{u}|} \, / \, \overline{|\mathbf{u}|}$, %", 
                     title_=subfig, levels=levels,
                     savefile=os.path.join(folder, f'{quantity}_mean_diff_{scenario}_fld.{fformat}')
            )

    # Ebb phase
    start_ebb, stop_ebb = 19,37
    for step in range(start_ebb, stop_ebb):
        with CheckpointFile(f'{baseline}/outputs/hdf5/Velocity2d_{step:05d}.h5', 'r') as h5f_amb_velo:
            if step == start_ebb:
                mesh2d = h5f_amb_velo.load_mesh()
                fs = get_functionspace(mesh2d, "CG", 1)
                
            velo_amb = h5f_amb_velo.load_function(mesh2d, name="velocity")
            uu_amb = velo_amb.sub(0)
            vv_amb = velo_amb.sub(1)
            if step == start_ebb:
                vel_mag_max = (uu_amb**2 + vv_amb**2)**0.5
                vel_mag_avg = (uu_amb**2 + vv_amb**2)**0.5
            else:
                vel_mag = (uu_amb**2 + vv_amb**2)**0.5
                vel_mag_max = conditional(gt(vel_mag, vel_mag_max), vel_mag, vel_mag_max)
                vel_mag_avg = vel_mag_avg + vel_mag
    
    vel_mag_peak_amb = Function(fs, name="vel_peak_ebb_base").interpolate(vel_mag_max)
    vel_mag_mean_amb = Function(fs, name="vel_mean_ebb_base").interpolate(vel_mag_avg / (stop_ebb - start_ebb + 1))

    for step in range(start_ebb, stop_ebb):
        with CheckpointFile(f'{modified}/outputs/hdf5/Velocity2d_{step:05d}.h5', 'r') as h5f_grp_velo:
            # if step == start_ebb:
            #     mesh2d = h5f_grp_velo.load_mesh()
            #     fs = get_functionspace(mesh2d, "CG", 1)
                
            velo_grp = h5f_grp_velo.load_function(mesh2d, name="velocity")
            uu_grp = velo_grp.sub(0)
            vv_grp = velo_grp.sub(1)
            if step == start_ebb:
                vel_mag_max = (uu_grp**2 + vv_grp**2)**0.5
                vel_mag_avg = (uu_grp**2 + vv_grp**2)**0.5
            else:
                vel_mag = (uu_grp**2 + vv_grp**2)**0.5
                vel_mag_max = conditional(gt(vel_mag, vel_mag_max), vel_mag, vel_mag_max)
                vel_mag_avg = vel_mag_avg + vel_mag
    
    vel_mag_peak_grp = Function(fs, name="vel_peak_ebb_modf").interpolate(vel_mag_max)
    vel_mag_mean_grp = Function(fs, name="vel_peak_ebb_modf").interpolate(vel_mag_avg / (stop_ebb - start_ebb + 1))

    velmag_peak_reldiff_ebb = Function(fs, name="vel_peak_ebb_diff").interpolate((vel_mag_peak_grp - vel_mag_peak_amb)/ vel_mag_peak_amb * 100)
    velmag_mean_reldiff_ebb = Function(fs, name="vel_mean_ebb_diff").interpolate((vel_mag_mean_grp - vel_mag_mean_amb)/ vel_mag_mean_amb * 100)

    map_show_compact(0, velmag_peak_reldiff_ebb, mesh2d, colormap="RdBu_r", 
                     # clabel=r"$\Delta {|\mathbf{u}|_\max} \, / \, {|\mathbf{u}|_\max}$, %", 
                     title_=subfig, levels=levels,
                     savefile=os.path.join(folder, f'{quantity}_max_diff_{scenario}_ebb.{fformat}')
            )
    map_show_compact(0, velmag_mean_reldiff_ebb, mesh2d, colormap="RdBu_r", 
                     # clabel=r"$\Delta \overline{|\mathbf{u}|} \, / \, \overline{|\mathbf{u}|}$, %", 
                     title_=subfig, levels=levels,
                     savefile=os.path.join(folder, f'{quantity}_mean_diff_{scenario}_ebb.{fformat}')
            )


def compare_and_plot_flux(baseline, modified, 
                     folder='figout',
                     quantity='E',
                     scenario='pilot',
                     subfig='(a)',
                     fformat='pdf', contourline=False,
                     levels=np.linspace(-6, 6, 13),
                    ):
    """ Plot and save files (*.png) separately for field difference between
        baseline and modified scenarios, both flood and ebb stages.

        :param baseline: name of baseline case (also name of folder) (`str`)
        :param modified: name of modified case (also name of folder) (`str`)
        :param folder: name of the output folder to store image (`str`)
        :param quantity: name of quantity (to include in file name) (`str`)
        :param scenario: name of scenario (to include in file name) (`str`)
        :param subfig: e.g. (a), (b), (c), (d), to annotate in figures (`str`)
        :param fformat: export file format (`'png'`, `'pdf'`) (`str`)
        :param levels: list of values for color bar (`list int`)
    """
    from copy import copy, deepcopy

    RHO = 1025
    GRAV = 9.81
    phases = [
                ('fld', 1, 19, 10), 
                ('ebb', 19, 37, 28),
             ]
    for phase_name, phase_start, phase_end, phase_rep in phases:
        # (fixed) bathymetry
        with CheckpointFile(f"{baseline}/outputs/end_of_run_export.h5", 'r') as h5f_end:
            mesh2d = h5f_end.load_mesh()
            fs = get_functionspace(mesh2d, "CG", 1)
            bathy = h5f_end.load_function(mesh2d, name="bathymetry")
        
        for step in range(phase_start, phase_end):
            with CheckpointFile(f'{baseline}/outputs/hdf5/Velocity2d_{step:05d}.h5', 'r') as h5f_base_velo:
                velo_base = h5f_base_velo.load_function(mesh2d, name="velocity")
                uu_base = velo_base.sub(0)
                vv_base = velo_base.sub(1)
                vmag_base = (uu_base**2 + vv_base**2)**0.5
            
            with CheckpointFile(f"{baseline}/outputs/hdf5/Elevation2d_{step:05d}.h5", 'r') as h5f_base_elev:
                # mesh2d = h5f_base_elev.load_mesh()
                # fs = get_functionspace(mesh2d, "CG", 1)
                elev_base = h5f_base_elev.load_function(mesh2d, name="elevation")
    
            KE = 0.5 * RHO * vmag_base**2 * (bathy + elev_base)
            PE = RHO * GRAV * elev_base * (bathy + elev_base)
            TE = KE + PE
            Eflux = TE * vmag_base

            if step == phase_start:
                Eflux_max = Eflux
                Eflux_avg = Eflux
                TE_max = TE
                TE_avg = TE
            else:
                Eflux_max = conditional(gt(Eflux, Eflux_max), Eflux, Eflux_max)
                Eflux_avg = Eflux_avg + Eflux
                TE_max = conditional(gt(TE, TE_max), TE, TE_max)
                TE_avg = TE_avg + TE

            if step == phase_rep:
                print('Phase name', phase_name, 'representative', phase_rep)
                velo_rep_base = deepcopy(velo_base)
                TE_rep_base = copy(TE)
        
        Eflux_max_base = Function(fs, name=f"Eflux_peak_{phase_name}_base").interpolate(Eflux_max)
        Eflux_mean_base = Function(fs, name=f"Eflux_mean_{phase_name}_base").interpolate(Eflux_avg / (phase_end - phase_start + 1))
        
        for step in range(phase_start, phase_end):
            with CheckpointFile(f'{modified}/outputs/hdf5/Velocity2d_{step:05d}.h5', 'r') as h5f_modf_velo:
                velo_modf = h5f_modf_velo.load_function(mesh2d, name="velocity")
                uu_modf = velo_modf.sub(0)
                vv_modf = velo_modf.sub(1)
                vmag_modf = (uu_modf**2 + vv_modf**2)**0.5
    
            with CheckpointFile(f"{baseline}/outputs/hdf5/Elevation2d_{step:05d}.h5", 'r') as h5f_modf_elev:
                # mesh2d = h5f_modf_elev.load_mesh()
                # fs = get_functionspace(mesh2d, "CG", 1)
                elev_modf = h5f_modf_elev.load_function(mesh2d, name="elevation")
    
            KE = 0.5 * RHO * vmag_modf**2 * (bathy + elev_modf)
            PE = RHO * GRAV * elev_modf * (bathy + elev_modf)
            TE = KE + PE
            Eflux = TE * vmag_modf

            
            if step == phase_start:
                Eflux_max = Eflux
                Eflux_avg = Eflux
                TE_max = TE
                TE_avg = TE
            else:
                Eflux_max = conditional(gt(Eflux, Eflux_max), Eflux, Eflux_max)
                Eflux_avg = Eflux_avg + Eflux
                TE_max = conditional(gt(TE, TE_max), TE, TE_max)
                TE_avg = TE_avg + TE
        
        Eflux_max_modf = Function(fs, name=f"Eflux_max_{phase_name}_modf").interpolate(Eflux_max)
        Eflux_mean_modf = Function(fs, name=f"Eflux_mean_{phase_name}_modf").interpolate(Eflux_avg / (phase_end - phase_start + 1))
        
        Eflux_max_reldiff = Function(fs, name=f"Eflux_max_{phase_name}_diff").interpolate((Eflux_max_modf - Eflux_max_base)/ Eflux_max_base * 100)
        Eflux_mean_reldiff = Function(fs, name=f"Eflux_mean_{phase_name}_diff").interpolate((Eflux_mean_modf - Eflux_mean_base)/ Eflux_mean_base * 100)
    
        Eflux_base_rect = firedrake.interpolate(velo_rep_base * abs(TE_rep_base)/1E6, vfs_rect, allow_missing_dofs=True)  # velo_rep_base * TE_rep_base/1E6

        
        quiver_compact(0, Eflux_max_reldiff, Eflux_base_rect, mesh2d, colormap="RdBu_r", clims=(-80,80),
                         # clabel=r"$\Delta {|\mathbf{u}|_\max} \, / \, {|\mathbf{u}|_\max}$, %",
                         arrowwidth=0.02, # line of quiver vectors
                         refveclen=1, veclegend=r'$10^6$ W/m', labelsep=-0.1, vecposx=0.46, vecposy=0.92,
                         title_=subfig, levels=levels, contourline=contourline,
                         savefile=os.path.join(folder, f'{quantity}_max_diff_{scenario}_{phase_name}.{fformat}')
                )
        quiver_compact(0, Eflux_mean_reldiff, Eflux_base_rect, mesh2d, colormap="RdBu_r", clims=(-80,80),
                         # clabel=r"$\Delta \overline{|\mathbf{u}|} \, / \, \overline{|\mathbf{u}|}$, %", 
                         arrowwidth=0.02, # line of quiver vectors
                         refveclen=1, veclegend=r'$10^6$ W/m', labelsep=-0.1, vecposx=0.46, vecposy=0.92,
                         title_=subfig, levels=levels, contourline=contourline,
                         savefile=os.path.join(folder, f'{quantity}_mean_diff_{scenario}_{phase_name}.{fformat}')
                )


if __name__ == '__main__':
    compare_and_plot('case_PF2-N_visc_5_2c', 'case_PF2-A_visc_5_2c', 
                        folder='fig_out',
                        quantity='U_',
                        scenario='2A',
                        subfig='(f)',
                        levels=np.linspace(-80, 80, 17),
                        )
    # Recommendation for levels used for the difference plots:
    # G -- (-6,6,13)
    # A -- (-80, 80, 17)

    compare_and_plot_flux('case_PF2-N_visc_5_2c', 'case_PF2-G_visc_5_2c', 
                        folder='fig_out',
                        quantity='E',
                        scenario='2G',
                        subfig='(e)',
                        fformat='png',
                        levels=np.linspace(-8, 8, 17),
                        )  

    compare_and_plot_flux('case_PF2-N_visc_5_2c', 'case_PF2-A_visc_5_2c', 
                        folder='fig_out',
                        quantity='E',
                        scenario='2A',
                        subfig='(f)',
                        fformat='png', contourline=False,
                        levels=np.linspace(-100, 100, 11),
                        )
