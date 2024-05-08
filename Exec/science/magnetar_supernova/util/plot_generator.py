#!/usr/bin/env python3

import yt
import matplotlib.pyplot as plt
import unyt as u

try:
    import analysis_util as au
except ImportError:
    print("Warning: failed to import analysis_util module; cannot use MPI.")

import os
import sys
import glob
import argparse
import warnings
import numpy as np

from datetime import datetime

#########################
# Argument help strings #
#########################

description = """Generates plots of datasets using a specified yt plot function. Works for slice and projection
        plots, but does not support particles. Slice plots are the default. Any argument that states it takes a
        dictionary takes colon-separated keyword:value pairs. Lists can be supplied as comma-separated values with
        no whitespace between them."""
datasets_help = "A list of datasets to be loaded by yt. Will be sorted by plot number by default."
list_fields_help = "List out the available fields in each data file and exit."
proj_help = "Make a ProjectionPlot instead of a SlicePlot."
out_help = "The desired output directory for the image files."
var_help = "The variable to plot. Set to 'temp' by default."
normal_help = "The normal vector to the plot. Can either be an axis name or a sequence of floats."
bounds_help = "The bounds for the colorbar."
cmap_help = "The colormap for the variable to plot."
log_help = "If provided, sets the plot to a logarithmic scale."
time_help = """If provided, adds a timestamp to each plot with the given precision. Additional
        configuration options can be set using --time_opt."""
time_opt_help = """Dictionary of additional arguments for formatting the time displayed on the plot.
        The 'pos' and 'coord_system' keyword arguments to annotate_text are be valid, and the
        additional option 'sci' can be set to 1 to give the time in scientific notation, 'time_unit'
        and 'second_time_unit' can be used to set the units, and any remaining arguments are
        supplied using the text_args keyword argument."""
plot_args_help = """Dictionary of additional plot args to supply to the plot constructor
        (e.g. '--plot_args method:integrate' for a projection plot). Will be overridden by arguments
        supplied through other options. See script description for explanation of dictionary format."""
ext_help = "The extension of the file format to save to. PNG by default."
save_args_help = """Keyword arguments to supply to the save function. Should be valid keyword arguments for the
        matplotlib.pyplot.savefig function, in the form of a dictionary (see script description). Example: --save_args
        dpi:480 orientation:landscape."""
sort_help = """A floating point number specifying the digits to sort file names by. Digits preceding the decimal point
        give the starting index, digits following the decimal point give the number of characters. Make negative for
        descending order."""
quiver_help = """Overplots a vector field on each generated plot, taking the x and y components, the number of
        points to skip, and a scale factor for the arrows."""
contour_help = "Adds a contour map to the plot, with NCONT contours and limits specified by CLIM_LOW and CLIM_HI."
contour_opt_help = "Plot args to supply to the matplotlib contour function. Takes a color and a linewidth."
xlim_help = "The x-axis limits in cm."
ylim_help = "The y-axis limits in cm."
stream_help = """Adds streamlines to the plot showing the given vector field (x and y components are first two arguments).
        The third argument should be the streamline density factor (e.g. 16)."""
stream_opt_help = """Options for the streamlines -- will be ignored if streamlines themselves were not requested. Options
        should be supplied in the form of a dictionary (see script description). Valid options include
        display_threshold (to turn off the streamlines below a certain value for field color), outline (add a 1 px
        outline to the streamlines, value should be the color), and any keyword argument that can be supplied to
        matplotlib.pyplot.streamplot as a string, float, int, or list of any of the three."""
grid_help = """Add an overlay to the plot showing the hierarchical grid structure. May omit arguments or
        supply additional options to yt as a dictionary (e.g. --grid alpha:0.5 min_level:1 cmap:gray)."""
cell_edges_help = "Overplot the edges of the grid cells."
window_size_help = "Length of the plot window in inches in its maximum dimension. Can only be specified for on-axis plots."
aspect_help = "Aspect ratio to use for the plot axes (not the image). Can only be specified for on-axis plots."
even_dim_help = """Adjust the image size in pixels in each dimension to the nearest multiple of two.
        At the time of writing, using this option necessitates saving the figure directly with the
        matplotlib interface, as yt resets the figure size at some point during the plot.save() call."""
Pmag_help = "Magnetar period (in ms) to use for setting the magnetar timescale. Assumes B = 10^{15} Gauss."
use_mpi_help = """Whether to parallelize with MPI or not, assuming script was run with MPI. Requires
        analysis_util module to be present."""
plugin_help = """Provide this argument to have yt load a plugin file. Can supply a filename to load
        a specific plugin file, or omit the filename to have yt look for the plugin file in the
        default location."""
overwrite_image_help = """Overwrite the image with data from a uniform covering grid at the
        resolution of the specified AMR level. Requires the analysis_util module to be present."""

#########################
# Argument parser setup #
#########################

parser = argparse.ArgumentParser(description=description)
parser.add_argument('datasets', nargs='*', help=datasets_help)
parser.add_argument('-l', '--list_fields', action='store_true', help=list_fields_help)
parser.add_argument('--proj', action='store_true', help=proj_help)
parser.add_argument('-o', '--out', default='', help=out_help)
parser.add_argument('-v', '--var', default='density', help=var_help)
parser.add_argument('-n', '--normal', nargs='+', default='', help=normal_help)
parser.add_argument('-b', '--bounds', nargs=2, type=float, metavar=('LOWER', 'UPPER'),
        help=bounds_help)
parser.add_argument('-c', '--cmap', metavar=('NAME',), help=cmap_help)
parser.add_argument('--log', action='store_true', help=log_help)
parser.add_argument('-t', '--time', type=int, metavar=('PRECISION',), help=time_help)
parser.add_argument('-to', '--time_opt', nargs='+', help=time_opt_help)
parser.add_argument('--plot_args', nargs='+', default=None, help=plot_args_help)
parser.add_argument('-e', '--ext', default='png', help=ext_help)
parser.add_argument('--save_args', nargs='*', help=save_args_help)
parser.add_argument('-s', '--sort', type=float, default=0.0, help=sort_help)
parser.add_argument('-q', '--quiver', nargs=4, metavar=('XFIELD', 'YFIELD', 'FACTOR', 'SCALE'),
        help=quiver_help)
parser.add_argument('-C', '--contour', nargs=4, metavar=('FIELD', 'NCONT', 'CLIM_LOW', 'CLIM_HI'),
        help=contour_help)
parser.add_argument('-Co', '--contour_opt', nargs=2, metavar=('COLOR', 'LINEWIDTH'),
        help=contour_opt_help)
parser.add_argument('-x', '--xlim', nargs=2, type=float, metavar=('UPPER', 'LOWER'), help=xlim_help)
parser.add_argument('-y', '--ylim', nargs=2, type=float, metavar=('UPPER', 'LOWER'), help=ylim_help)
parser.add_argument('--xseq', nargs=3, type=float, metavar=('START', 'STOP', 'STEP'))
parser.add_argument('--yseq', nargs=3, type=float, metavar=('START', 'STOP', 'STEP'))
parser.add_argument('-S', '--stream', nargs=2, metavar=('XFIELD', 'YFIELD'))
parser.add_argument('-So', '--stream_opt', nargs='*', help=stream_opt_help)
parser.add_argument('--grid', nargs='*', default=None, help=grid_help)
parser.add_argument('--cell_edges', action='store_true', help=cell_edges_help)
parser.add_argument('--window_size', type=float, default=8.0, help=window_size_help)
parser.add_argument('--aspect', type=float, default=None, help=aspect_help)
parser.add_argument('--even_dim', action='store_true', help=even_dim_help)
parser.add_argument('--Pmag', type=float, default=1.0, help=Pmag_help)
parser.add_argument('--use_mpi', action='store_true', help=use_mpi_help)
parser.add_argument('--plugin', nargs='?', const="", help=plugin_help)
parser.add_argument('--overwrite_image', type=int, metavar=('LEVEL',), help=overwrite_image_help)

args = parser.parse_args()

##############################
# Process argparse arguments #
##############################

if args.plugin is not None:
    if args.plugin:
        yt.enable_plugins(args.plugin)
    else:
        yt.enable_plugins()

if args.use_mpi:
    MPI = au.mpi_importer()
is_main_proc = (not args.use_mpi) or (MPI.COMM_WORLD.Get_rank() == 0)

def safe_convert(string_val):
    """
    Convert string value to an int or float if possible, and return it if not.
    Can also convert comma-separated lists of these types.
    """

    values = string_val.split(',')

    for i in range(len(values)):

        try:
            values[i] = int(values[i])
        except ValueError:
            try:
                values[i] = float(values[i])
            except ValueError:
                pass

    if len(values) == 1:
        values = values[0]
    return values

def get_argdict(arglist):
    """
    Converts whitespace-delimited list of key-value pairs into dictionary. Can handle numeric
    and list values.
    """

    pairs = map(lambda s: s.split(':'), arglist)
    pairs = ((k, safe_convert(v)) for k, v in pairs)
    return dict(pairs)
    
def get_reasonable_time_unit(ds, time):
    """
    Convert to appropriate time unit (uses *ds.get_smallest_appropriate_unit* function with minutes
    added to the sequence).
    """
    
    if not isinstance(time, u.unyt_quantity):
        time = time * u.s
    if 60.0 <= time.in_cgs().d <= 3600:
        return u.min
    return ds.get_smallest_appropriate_unit(time, quantity="time")
    
if args.time is not None:
    
    if args.time_opt is not None:
        args.time_opt = get_argdict(args.time_opt)
    else:
        args.time_opt = dict()

if args.normal:

    if len(args.normal) == 1:
        args.normal = args.normal[0]
    else:
        args.normal = list(map(float, args.normal))

if args.quiver is not None:

    # Number of points to skip
    args.quiver[2] = int(args.quiver[2])
    # Scale factor
    args.quiver[3] = float(args.quiver[3])

if args.stream is not None:

    stream_opt = {}

    # Additional streamline options
    if args.stream_opt is not None:
        stream_opt = get_argdict(args.stream_opt)

if args.contour is not None:

    args.contour[1] = int(args.contour[1])
    args.contour[2:] = list(map(float, args.contour[2:]))
    contour_opt = {}

    if args.contour_opt is not None:
        plot_args = {'colors': args.contour_opt[0], 'linewidths': int(args.contour_opt[1])}
        contour_opt['plot_args'] = plot_args

#########################
# Make output directory #
#########################

if is_main_proc:
    if not args.out:
        args.out = os.getcwd()
    if not os.path.exists(args.out):
        os.makedirs(args.out)

#######################
# Sort and load files #
#######################

ts = args.datasets
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

desc = args.sort < 0
start = abs(int(args.sort))
nchars = int(str(args.sort).split('.')[1])

if nchars == 0:
    key = lambda fname: fname[start:]
else:
    key = lambda fname: fname[start:start + nchars]
ts.sort(key=key, reverse=desc)

if is_main_proc:
    print("Will load the following files: {}\n".format(ts))

if args.use_mpi:
    ts = au.FileLoader(ts, True)
    MPI.COMM_WORLD.Barrier()
else:
    tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')   
    ts = map(tf, ts)

#############################
# Prepare to generate plots #
#############################

def get_width(ds, xlim=None, ylim=None, zlim=None):
    """ Get the width of the plot. """

    xw, yw, zw = ds.domain_width.in_cgs()

    if xlim is not None:
        xw = min((xlim[1] - xlim[0]), xw) * u.cm

    if ylim is not None:
        yw = min((ylim[1] - ylim[0]), yw) * u.cm

    if zlim is not None:
        zw = min((zlim[1] - zlim[0]), zw) * u.cm

    return xw, yw, zw

def get_center(ds, xlim=None, ylim=None, zlim=None):
    """ Get the coordinates of the center of the plot. """

    xctr, yctr, zctr = ds.domain_center.in_cgs()

    if xlim is not None:
        xctr = 0.5 * (xlim[0] + xlim[1]) * u.cm

    if ylim is not None:
        yctr = 0.5 * (ylim[0] + ylim[1]) * u.cm

    if zlim is not None:
        zctr = 0.5 * (zlim[0] + zlim[1]) * u.cm

    return xctr, yctr, zctr

#####################
# Loop and generate #
#####################

def make_plot(ds, args, xlim, ylim, fname_pref=None):
    
    field = args.var
    
    if args.plot_args:
        settings = get_argdict(args.plot_args)
    else:
        settings = dict()
    
    settings['center'] = get_center(ds, xlim, ylim)
    settings['width'] = get_width(ds, xlim, ylim)

    if args.normal:
        settings['normal'] = args.normal

    if ds.geometry in {'cylindrical', 'spherical'}:
        settings.setdefault('normal', 'theta')
        settings['origin'] = 'native'
    else:
        settings.setdefault('normal', 'z')

    if isinstance(settings['normal'], str):
        settings['window_size'] = args.window_size
        settings['aspect'] = args.aspect

    if args.proj:
        plotfunc = yt.ProjectionPlot
        normal = settings.pop('normal')
        settings['axis'] = normal
    else:
        plotfunc = yt.SlicePlot
    
    # del settings['center']
    # del settings['width']
    plot = plotfunc(ds, fields=field, **settings)

    #######################################
    # Additional settings and annotations #
    #######################################

    if args.cmap:
        plot.set_cmap(field=field, cmap=args.cmap)

    if args.bounds is not None:
        plot.set_zlim(field, *args.bounds)

    plot.set_log(field, args.log)

    if args.time is not None:

        time = ds.current_time
        time_opt = args.time_opt.copy()
        
        if time_opt.pop('sci', 0):
            scistr = 'e'
        else:
            scistr = 'f'
        time_unit = time_opt.pop('time_unit', 's')
        second_time_unit = time_opt.pop('second_time_unit', None)
        if time_unit == "auto":
            time_unit = get_reasonable_time_unit(ds, time)
        if second_time_unit == "auto":
            second_time_unit = get_reasonable_time_unit(ds, time)

        def_pos = (0.03, 0.96) # upper left
        pos = time_opt.pop('pos', def_pos) 
        coord_sys = time_opt.pop('coord_system', 'axis')
        label_pref = time_opt.pop('prefix', 't = ')
        time_opt.setdefault('horizontalalignment', 'left')
        time_opt.setdefault('verticalalignment', 'top')

        reg = ds.current_time.units.registry
        reg.add('tmag', base_value=(args.Pmag**2 * 2047.49866274), dimensions=u.dimensions.time,
                tex_repr="\\rm{t_{mag}}")
        reg.add('teng', base_value=(args.Pmag**2 * 2047.49866274), dimensions=u.dimensions.time,
                tex_repr="\\rm{t_{eng}}")
                
        time_fmt = f"{{:.{args.time}{scistr}}}"
        time_text = label_pref + f"{time_fmt}$~{{}}$"
        time_u = time.to(time_unit)
        time_text = time_text.format(time_u.d, time_u.units.latex_repr)
        if second_time_unit is not None:
            second_time_text = f" ({time_fmt}$~{{}})$"
            time_su = time.to(second_time_unit)
            second_time_text = second_time_text.format(time_su.d, time_su.units.latex_repr)
            time_text += second_time_text
            
        plot.annotate_text(pos, time_text, coord_system=coord_sys, text_args=time_opt)
        
        # Previously: draw_inset_box=True, inset_box_args={'alpha': 0.0}
        # Previously: time_format = 't = {{time:.{}{}}}{{units}}'.format(args.time, scistr)
        # Previously used annotate_timestamp

    if args.quiver is not None:
        plot.annotate_quiver(*args.quiver)

    if args.contour is not None:
        plot.annotate_contour(args.contour[0], levels=args.contour[1], clim=args.contour[2:], **contour_opt)

    if args.stream is not None:

        plot_args = stream_opt.copy()

        # Non-matplotlib arguments
        outline = plot_args.pop('outline', None)
        cbar = plot_args.pop('cbar', False)

        # Defaults for some matplotlib arguments
        plot_args.setdefault('linewidth', 2)
        plot_args.setdefault('arrowstyle', '->')

        if outline is not None:
            plot_args_outline = plot_args.copy()
            plot_args_outline['color'] = outline
            plot_args_outline['linewidth'] =  plot_args['linewidth'] + 1
            plot.annotate_streamlines(*args.stream, **plot_args_outline)

        if cbar:
            plot_args['add_colorbar'] = True

        plot.annotate_streamlines(*args.stream, **plot_args)

    if args.grid is not None:

        opts = get_argdict(args.grid)
        plot.annotate_grids(**opts)

        # cmap = yt.make_colormap([('white', 64), ((188/255, 95/255, 211/255), 64),
        #         ((0, 0.627, 1.0), 64), ('purple', 64)],
        #         name='gridmap', interpolate=False)
        # opts = dict(zip(args.grid[::2], args.grid[1::2]))
        # plot.annotate_grids(**opts, max_level=2)

    if args.cell_edges:

        plot.annotate_cell_edges()
        
    if args.overwrite_image is not None:
        
        print("Overwriting image data...")
        ad = au.AMRData(ds, args.overwrite_image)
        data = ad.field_data(field, units=False)
        
        if xlim or ylim:
            if not xlim:
                xlim = ad.left_edge[0], ad.right_edge[0]
            if not ylim:
                ylim = ad.left_edge[1], ad.right_edge[1]
            data, _ = ad.select_region(data, *xlim, *ylim)    
                
        fig = plot.export_to_mpl_figure((1,1))
        subplot = plot.plots[field]    
        subplot.image.set_data(data.T[::-1])
        fig.canvas.draw_idle()

    #############
    # Save plot #
    #############

    fig = plot.plots[field].figure
    
    if args.save_args is None:
        save_args = {}
    else:
        save_args = get_argdict(args.save_args)
        
    if args.overwrite_image is not None:
        if 'bbox_inches' not in save_args:
            save_args['bbox_inches'] = 'tight'
            save_args['pad_inches'] = 0.25
    
    if fname_pref == None:
        fname_pref = f'{ds}'
    plot.save(os.path.join(args.out, fname_pref), suffix=args.ext,
            mpl_kwargs=save_args)

    if args.even_dim:

        dpi = save_args.setdefault('dpi', fig.get_dpi())
        fw, fh = dpi * fig.get_size_inches()
        fw = 2 * round(fw / 2)
        fh = 2 * round(fh / 2)
        if fw > (fw/dpi * dpi):
            fw += fw - (fw/dpi * dpi)
        if fh > (fh/dpi * dpi):
            fh += fh - (fh/dpi * dpi)
        fig.set_size_inches(fw/dpi, fh/dpi)

        if args.proj:
            plot_type = 'Projection'
            normkey = 'axis'
        else:
            plot_type = 'Slice'
            normkey = 'normal'

        if isinstance(settings[normkey], str):
            type_and_normal = f'{plot_type}_{settings[normkey]}'
        else:
            type_and_normal = f'OffAxis{plot_type}'

        fname = f'{fname_pref}_{type_and_normal}_{field}.{args.ext}'
        fpath = os.path.join(args.out, fname)
        print(f"Resaving using matplotlib...")
        fig.savefig(fpath, **save_args)

    print()

if not args.list_fields and is_main_proc:
    print("Generating...")

for ds in ts:

    ############################
    # List fields if requested #
    ############################

    if args.list_fields:
        print()
        print(f"Fields list for {ds}:")
        print(ds.field_list)
        print()
        continue

    ##################
    # Make base plot #
    ##################
    
    # If we are only doing one zoom level, generate that plot and continue
    if not (args.xseq or args.yseq):
        make_plot(ds, args, args.xlim, args.ylim)
        continue
        
    # Otherwise, generate zoom sequence with variable axis bounds
    if args.xseq:
        xseq = np.linspace(args.xseq[0], args.xseq[1], int((args.xseq[1] - args.xseq[0])/args.xseq[2]) + 1)
    else:
        xseq = 1.0
    if args.yseq:
        yseq = np.linspace(args.yseq[0], args.yseq[1], int((args.yseq[1] - args.yseq[0])/args.yseq[2]) + 1)
    else:
        yseq = 1.0
        
    xseq = np.outer(np.array(args.xlim), xseq).T
    yseq = np.outer(np.array(args.ylim), yseq).T
    
    n = max(len(xseq), len(yseq))
    for i in range(n):
        ndigits = len(str(n))
        base_pref = '{}_seq{:0%dd}' % ndigits
        make_plot(ds, args, xseq[i % len(xseq)], yseq[i % len(yseq)], base_pref.format(ds, i))

if args.use_mpi:
    MPI.COMM_WORLD.Barrier()
    
if is_main_proc:
    print("Task completed.")
