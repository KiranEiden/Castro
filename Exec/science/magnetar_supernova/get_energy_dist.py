#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from yt.frontends.boxlib.data_structures import CastroDataset

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument("--no_dist", action='store_true')
parser.add_argument('-r', '--refinement', type=int, default=1)
parser.add_argument('-t', '--thresh', type=float, default=7.5e3)
parser.add_argument('--plot_cs_prof', action='store_true')
parser.add_argument('--plot_mask', action='store_true')
parser.add_argument('--plot_M_of_v', action='store_true')
parser.add_argument('-o', '--output', default="energy.dat")
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: CastroDataset(file.rstrip('/'))
ts = map(tf, ts)

def do_plot_cs_prof(r, z, cs):
    
    print(f"Plotting soundspeed profile for {ds}.")
    # Fixed resolution rays don't work in 2d with my yt version
    interp = RegularGridInterpolator((r[0], z[:,0]), cs/cs.min(), bounds_error=False, fill_value=None)
    theta = np.linspace(0.0, np.pi, num=100)
    xi = np.column_stack((np.sin(theta), np.cos(theta)))
    
    avg = np.zeros_like(r[0])
    mxm = np.zeros_like(r[0])
    mnm = np.zeros_like(r[0])
    for i in range(nr):
        pts = interp(r[0,i] * xi)
        avg[i] = pts.mean()
        mxm[i] = pts.max()
        mnm[i] = pts.min()
        
    plt.plot(r[0], avg, label=r"$\langle c_s \rangle_{\theta}$")
    plt.plot(r[0], mxm, linestyle="-.", label=r"$\mathrm{max}_{\theta}(c_s)$")
    plt.plot(r[0], mnm, linestyle=":", label=r"$\mathrm{min}_{\theta}(c_s)$")
    plt.hlines(args.thresh, r[0,0], r[0,-1], color="black", linestyle="--", label="Threshold")
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    plt.ylabel(r"$\tilde{c}_s$")
    plt.yscale("log")
    plt.legend()
    
    plt.savefig(f'cs_prof_{ds}.png')
    plt.gcf().clear()
    
def do_plot_mask(r, z, mask):
    
    print(f"Plotting mask for {ds}.")
    plt.scatter(r[mask], z[mask], s=0.5)
    plt.xlabel('r [cm]')
    plt.ylabel('z [cm]')
    plt.xlim(r[0,0], r[0,-1])
    plt.ylim(z[0,0], z[-1,0])
    plt.savefig(f'mask_{ds}.png')
    plt.gcf().clear()
    
def do_plot_M_of_v(ds):
    
    print(f"Plotting M(v) for {ds}.")
    plot = yt.ProfilePlot(ds, ('boxlib', 'magvel'), [('gas', 'mass')], weight_field=None)
    plot.save()
    
def calc_2d(ds):
    
    do_mask = not args.no_dist
    
    # Make FRB
    slc = ds.slice(2, np.pi)
    nr, nz, _ = ds.domain_dimensions*args.refinement
    rlo, zlo, _ = ds.domain_left_edge
    rhi, zhi, _ = ds.domain_right_edge
    frb = yt.FixedResolutionBuffer(slc, (rlo, rhi, zlo, zhi), (nr, nz))
    
    # Get position data and calculate cell volumes
    r = frb[('index', 'r')].d
    if args.plot_mask or args.plot_cs_prof:
        z = frb[('index', 'z')].d
    dr = (rhi - rlo).d / nr
    dz = (zhi - zlo).d / nz
    vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
    
    # Get data for relevant fields
    rhoE = frb[('boxlib', 'rho_E')].d
    rhoe = frb[('boxlib', 'rho_e')].d
    rho = frb[('boxlib', 'density')].d
    if do_mask or args.plot_mask or args.plot_cs_prof:
        cs = frb[('boxlib', 'soundspeed')].d
    
    if args.plot_cs_prof:
        do_plot_cs_prof(r, z, cs)
        
    """
    x = plt.imshow((cs / cs.min()) - 1.15e4, cmap="seismic", vmin=-2.5e5, vmax=2.5e5)
    plt.gcf().colorbar(x, ax=plt.gca())
    plt.show()
    """

    if do_mask:
        of_mask = cs/cs.min() > args.thresh
        ej_mask = ~of_mask

    if args.plot_mask:
        do_plot_mask(r, z, of_mask)
        
    if args.plot_M_of_v:
        do_plot_M_of_v(ds)
    
    if do_mask:
        
        Etot_of = (rhoE[of_mask] * vol[of_mask]).sum()
        etot_of = (rhoe[of_mask] * vol[of_mask]).sum()
        Mtot_of = (rho[of_mask] * vol[of_mask]).sum()
        Etot_ej = (rhoE[ej_mask] * vol[ej_mask]).sum()
        etot_ej = (rhoe[ej_mask] * vol[ej_mask]).sum()
        Mtot_ej = (rho[ej_mask] * vol[ej_mask]).sum()
        
        return Etot_of, Etot_ej, etot_of, etot_ej, Mtot_of, Mtot_ej
        
    else:
        
        Etot = (rhoE * vol).sum()
        etot = (rhoe * vol).sum()
        Mtot = (rho * vol).sum()
        
        return Etot, etot, Mtot

with open(args.output, 'w') as datfile:
    
    print(f"Data file: {args.output}.")
    print()
    
    if args.no_dist:
        
        print('t', 'E', 'e', 'M', file=datfile)
        print('s', 'erg', 'erg', 'g', file=datfile)
        
    else:
        
        print('t', 'E_of', 'E_ej', 'e_of', 'e_ej', 'M_of', 'M_ej', file=datfile)
        print('s', 'erg', 'erg', 'erg', 'erg', 'g', 'g', file=datfile)
        

    for ds in ts:
        
        n = ds.dimensionality
        fstr = f"calc_{n}d(ds)"
        vals = eval(fstr)
        
        print(ds.current_time.d, *vals, file=datfile)
