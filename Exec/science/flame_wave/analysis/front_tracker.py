#!/usr/bin/env python3

import sys
import argparse
import os
import yt
from yt.units import cm
from enum import Enum
from collections import defaultdict, deque
import matplotlib.pyplot as plt

################################
# set up parser and parse args #
################################

description = "Script for tracking the position of a flame or shock front as a function of time."

datasets_help = "Datasets to track the front position over."
out_help = "Output filename for the tracking information."
metrics_help = """A list of metrics to use for tracking. Should be fields followed by floating point
        numbers in the range (0, 1], indicating percent of maximum. For example, enuc 1 1e-3 Temp 1
        will track the position of max enuc and 1 / 1000th max enuc and the position of max
        temperature."""
xlim_help = "Limits on the first dimension."
ylim_help = "Limits on the second dimension."
zlim_help = "Limits on the third dimension."
res_help = "FRB resolution to use."
transform_help = """Operation to apply to each extra dimension. Can be of format
        ax_ind:transform or just a sequence of transforms, with ax_ind assumed to be in descending
        order. Transforms: 0 => slice, 1 => average."""
branch_help = """Whether to use the upper branch or lower branch when computing location. The upper
        branch is everything past the first instance of the local maximum, while the lower branch
        is everything before that. 0 => lower, 1 => upper. Default is upper."""
global_help = "If supplied, will use the global maximum across all snapshots instead of a local maximum."
units_help = "If supplied, will write out units as the second row in all data files."
plot_help = "A list of fields to plot profiles of."
plot_directory_help = "A directory to output the plots to. If it does not exist, it will be created."
wrl_help = """File to write data on local maxima to. Will write the times to the first column, along
with a column for each field used in a metric."""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('datasets', nargs='+', help=datasets_help)
parser.add_argument('-o', '--out', default='front_tracking.dat', help=out_help)
parser.add_argument('-m', '--metrics', nargs='*', default=['enuc', '1e-2', '1e-3'], help=metrics_help)
parser.add_argument('-x', '--xlim', nargs=2, type=float, metavar=('LOWER', 'UPPER'), help=xlim_help)
parser.add_argument('-y', '--ylim', nargs=2, type=float, metavar=('LOWER', 'UPPER'), help=ylim_help)
parser.add_argument('-z', '--zlim', nargs=2, type=float, metavar=('LOWER', 'UPPER'), help=zlim_help)
parser.add_argument('-r', '--res', nargs=2, type=int, help=res_help)
parser.add_argument('-t', '--transform', nargs='+', help=transform_help)
parser.add_argument('-b', '--branch', default=1, help=branch_help)
parser.add_argument('-gmax', '--use_global_max', action='store_true', help=global_help)
parser.add_argument('-u', '--units', action='store_true', help=units_help)
parser.add_argument('-p', '--plot', nargs='+', help=plot_help)
parser.add_argument('-pdir', '--plot_directory', help=plot_directory_help)
parser.add_argument('--write_local_maxima_to', help=wrl_help)

args = parser.parse_args(sys.argv[1:])

class Transform(Enum):
    
    SLC = 0
    AVG = 1

def process_args(args):
    
    # Load datasets
    tf = lambda fname: yt.load(fname.rstrip('/'))
    args.ts = list(map(tf, args.datasets))
    
    # Assume same domain for all datasets
    ds = args.ts[0]
    
    if args.xlim is None: args.xlim = ds.domain_left_edge[0], ds.domain_right_edge[0]
    if args.ylim is None: args.ylim = ds.domain_left_edge[1], ds.domain_right_edge[1]
    if args.zlim is None: args.zlim = ds.domain_left_edge[2], ds.domain_right_edge[2]
    
    metrics = dict()
    
    for item in args.metrics:
        
        try:
            ls.append(float(item))
        except:
            ls = metrics.setdefault(item, [])
            
    args.metrics = metrics
    
    if args.transform is None:
        
        args.transform = {Transform.SLC: [2], Transform.AVG: [1]}
        
    else:
        
        transform = {tr: [] for tr in Transform}
        
        for i, item in enumerate(args.transform):
            
            item = item.split(':')
            if len(item) == 1:
                item, = item
                transform[Transform(int(item))].append(3-i-1)
            elif len(item) == 2:
                ind, t = map(int, item)
                transform[Transform(t)].append(ind)
            else:
                raise ValueError("Invalid transform format.")
        
        args.transform = transform
        
    if args.res is None:
        args.res = ds.domain_dimensions
    if len(args.res) < 3:
        args.res += [1] * (3 - len(args.res))
        
    # Eventually may want to generalize this to allow multiple axes
    # Then we would just return a point in 2D or 3D space
        
    transformed = sum(args.transform.values(), [])
    args.axis, = filter(lambda ax: ax not in transformed, range(3))
    
    if ds.geometry == 'spherical': axnames = 'r',
    elif ds.geometry == 'cylindrical': axnames = 'r', 'z'
    else: axnames = 'x', 'y', 'z'
    
    args.axis_name = axnames[args.axis]
        
    assert len(args.res) == 3
    
process_args(args)

#################
# Metrics class #
#################

def get_window_parameters(ds, axis, width=None, center='c'):
    """ Some parameters controlling the frb window. """
    
    width = ds.coordinates.sanitize_width(axis, width, None)
    center, display_center = ds.coordinates.sanitize_center(center, axis)
    xax = ds.coordinates.x_axis[axis]
    yax = ds.coordinates.y_axis[axis]
    bounds = (display_center[xax]-width[0] / 2,
              display_center[xax]+width[0] / 2,
              display_center[yax]-width[1] / 2,
              display_center[yax]+width[1] / 2)
    return bounds, center, display_center
    
def get_width(ds, xlim=None, ylim=None, zlim=None):
    """ Get the width of the frb. """

    if xlim is None: xlim = ds.domain_left_edge[0], ds.domain_right_edge[0]
    else: xlim = xlim[0], xlim[1]

    if ylim is None: ylim = ds.domain_left_edge[1], ds.domain_right_edge[1]
    else: ylim = ylim[0], ylim[1]
    
    if zlim is None: zlim = ds.domain_left_edge[2], ds.domain_right_edge[2]
    else: zlim = zlim[0], zlim[1]

    xwidth = (xlim[1] - xlim[0]).in_cgs()
    ywidth = (ylim[1] - ylim[0]).in_cgs()
    zwidth = (zlim[1] - zlim[0]).in_cgs()

    return xwidth, ywidth, zwidth
    
def get_center(ds, xlim=None, ylim=None, zlim=None):
    """ Get the coordinates of the center of the frb. """

    if xlim is None: xlim = ds.domain_left_edge[0], ds.domain_right_edge[0]
    else: xlim = xlim[0], xlim[1]

    if ylim is None: ylim = ds.domain_left_edge[1], ds.domain_right_edge[1]
    else: ylim = ylim[0], ylim[1]
    
    if zlim is None: zlim = ds.domain_left_edge[2], ds.domain_right_edge[2]
    else: zlim = zlim[0], zlim[1]

    xctr = 0.5 * (xlim[0] + xlim[1]).in_cgs()
    yctr = 0.5 * (ylim[0] + ylim[1]).in_cgs()
    zctr = 0.5 * (zlim[0] + zlim[1]).in_cgs()

    return xctr, yctr, zctr
    
def minus_inf():
    """ Factory function for negative infinity. """
    
    return float("-inf")

class Metrics:
    """ Class for defining different measurements of the position of the flame front. """
    
    # Global maxima
    _globmax = defaultdict(minus_inf)
    
    def __init__(self, ds, args):
        
        # Complete set of fields, including position
        self.__dict__.update(self.makefrbs(ds, args))
        
        # Other parameters
        self.time = ds.current_time
        self._metrics = args.metrics
        self._upper_branch = args.branch > 0
        self._use_global_max = args.use_global_max
        self._local_maxima = dict()
        
        for field in args.metrics.keys():
            local_max = self[field].max()
            self._local_maxima[field] = local_max
            Metrics._globmax[field] = max(Metrics._globmax[field], local_max)
        
    def __getitem__(self, field):
        
        return getattr(self, field)
        
    @staticmethod
    def makefrbs(ds, args):
        
        fields = args.metrics.keys()
        width = get_width(ds, args.xlim, args.ylim, args.zlim)
        center = get_center(ds, args.xlim, args.ylim, args.zlim)
        
        region = \
        [
            slice(*args.xlim, complex(0, args.res[0])),
            slice(*args.ylim, complex(0, args.res[1])),
            slice(*args.zlim, complex(0, args.res[2]))
        ]
        
        for axis in args.transform[Transform.SLC]:
            
            _, center, _ = get_window_parameters(ds, axis, width, center)
            region[axis] = center[axis]
        
        # The resolution in yt FixedResolutionBuffers is backwards
        # If we're doing a slice, we need to swap the steps
        is_slice = [isinstance(bounds, slice) for bounds in region]
        dim = sum(is_slice)
        
        if dim == 2:
            
            normal = is_slice.index(False)
            xax = ds.coordinates.x_axis[normal]
            yax = ds.coordinates.y_axis[normal]
            xslc, yslc = region[xax], region[yax]
            
            region[xax] = slice(xslc.start, xslc.stop, yslc.step)
            region[yax] = slice(yslc.start, yslc.stop, xslc.step)

        # yt will raise an error no matter what for non-3D datasets
        # Here we cheat and hope nothing goes wrong
        old_dim = ds.dimensionality
        ds.dimensionality = 3
        frr = ds.r[region[0], region[1], region[2]]
        ds.dimensionality = old_dim

        # Create an FRB for each field and average over the remaining extra dimensions
        frbs = dict(pos=frr[args.axis_name])
        for field in fields: frbs[field] = frr[field]
        
        axes = sorted([args.axis] + args.transform[Transform.AVG])
        
        for i, ax in enumerate(reversed(axes)):
            
            if ax == args.axis:
                continue
            for field in frbs:
                frbs[field] = frbs[field].mean(axis=i)
                
        return frbs
        
    @staticmethod    
    def tostring(field, fac):
        
        return f"{field}[{fac*100}%]"
        
    @property
    def fields(self):
        
        return list(self._metrics.keys())
        
    @property
    def local_maxima(self):
        
        return self._local_maxima
        
    def local_max(self, field):
        
        return self._local_maxima[field]
        
    def locate(self, field, fac):
        """ Returns position where `field` drops to or reaches `fac * max(field)`. """
        
        maxind = self[field].argmax()
        
        if self._use_global_max:
            maxval = self._globmax[field]
        elif fac == 1.0:
            return self.pos[maxind]
        else:
            maxval = self[field].max()
            
        thresh = maxval * fac
        if self._upper_branch:
            ind = (self[field][maxind:] < thresh).argmax()
            ind += maxind
        else:
            ind = (self[field][maxind+1:] > thresh).argmax()
        return self.pos[ind]
        
    def getall(self):
        """ Return positions for all metrics, in a dictionary keyed by metric string. """
        
        locs = dict()
        
        for field, facs in self._metrics.items():
            
            for fac in facs:
                
                locs[self.tostring(field, fac)] = self.locate(field, fac)
                
        return locs

    def plot(self, fields=(), outdir="", bounds=(1e-8, 1)):

        for field in fields:
            if self._use_global_max:
                maxval = self._globmax[field]
            else:
                maxval = self[field].max()
            yvals = self[field] / maxval
            yvals += (self[field] == 0.0)*1e-10
            plt.plot(self.pos, self[field] / maxval)
            plt.yscale("log")
            plt.xlabel("r [cm]")
            plt.ylabel("{}/max[{}]".format(field, field))
            plt.ylim(bounds)
            plt.savefig(os.path.join(outdir, f"{field}_profile_{float(self.time):.3e}.png"))
            plt.gcf().clear()

#########################################
# Compute positions and write data file #
#########################################

def argsort(seq, key=lambda x: x):
    """ Return sorted index space of the input sequence. """
    
    seq = list(seq)
    ind = list(range(len(seq)))
    return sorted(ind, key=lambda i: key(seq[i]))

# Turn the time series into a queue and pop the datasets off one-by-one - this prevents it from
# storing all of the needed fields from all of the datasets at once
args.ts = deque(args.ts)
times = []
metrics = []

while args.ts:
    
    ds = args.ts.popleft()
    times.append(ds.current_time)
    metrics.append(Metrics(ds, args))
    del ds

# Make plots if asked to do so
if args.plot:
    if args.plot_directory is None:
        args.plot_directory = os.getcwd()
    if not os.path.exists(args.plot_directory):
        os.makedirs(args.plot_directory)
    for m in metrics:
        m.plot(args.plot, outdir=args.plot_directory)

# We need the global max to have been computed already to properly track the contour
# The Metrics objects all need to have been initialized before obtaining the position values
loclist = [m.getall() for m in metrics]

if args.units:
    items = sorted(loclist[0].items())
    ind = argsort(items, key=lambda item: item[0])
    cols = [items[i][0] for i in ind]
    units = [items[i][1].units for i in ind]
else:
    cols = sorted(loclist[0].keys())

with open(args.out, 'w') as file:
    
    print("time", *cols, file=file)
    if args.units:
        print(f"{times[0].units}", *units, file=file)
    
    for time, locs in zip(times, loclist):
        
        row = (locs[col].value for col in cols)
        print(time.value, *row, file=file)
        
if args.write_local_maxima_to is not None:
    
    maxlist = [m.local_maxima for m in metrics]
    
    if args.units:
        items = sorted(maxlist[0].items())
        ind = argsort(items, key=lambda item: item[0])
        cols = [items[i][0] for i in ind]
        units = [items[i][1].units for i in ind]
    else:
        cols = sorted(maxlist[0].keys())
    
    with open(args.write_local_maxima_to, 'w') as file:
        
        print("time", *cols, file=file)
        if args.units:
            print(f"{times[0].units}", *units, file=file)
        
        for time, vals in zip(times, maxlist):
            
            row = (vals[field].value for field in cols)
            print(time.value, *row, file=file)

print("Task completed.")
