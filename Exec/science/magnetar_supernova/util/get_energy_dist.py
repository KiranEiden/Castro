#!/usr/bin/env python3

import yt
import sys
import argparse
import tempfile
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
import analysis_util as au

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument("--no_dist", action='store_true')
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('-t', '--thresh', type=float, default=7.5e3)
parser.add_argument('--plot_cs_prof', action='store_true')
parser.add_argument('--plot_mask', action='store_true')
parser.add_argument('--plot_M_of_v', action='store_true')
parser.add_argument('--use_mpi', action='store_true')
parser.add_argument('-o', '--output', default="energy.dat")
args = parser.parse_args()

if args.use_mpi:
    MPI = au.mpi_importer()
is_main_proc = (not args.use_mpi) or (MPI.COMM_WORLD.Get_rank() == 0)

au.settings['verbose'] = True

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

if is_main_proc:
    print("Will load the following files: {}\n".format(ts))

ts = au.FileLoader(ts, args.use_mpi)

def do_plot_cs_prof(ds, r, z, cs):
    
    print(f"Plotting soundspeed profile for {ds}...")
    
    avg, mxm, mnm, r1d = au.get_avg_prof_2d(ds, 100, r, z, cs/cs.min(), return_minmax=True, return_r=True)
        
    plt.plot(r1d, avg, label=r"$\langle c_s \rangle_{\theta}$")
    plt.plot(r1d, mxm, linestyle="-.", label=r"$\mathrm{max}_{\theta}(c_s)$")
    plt.plot(r1d, mnm, linestyle=":", label=r"$\mathrm{min}_{\theta}(c_s)$")
    plt.hlines(args.thresh, r1d[0], r1d[-1], color="black", linestyle="--", label="Threshold")
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    plt.ylabel(r"$\tilde{c}_s$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    
    plt.savefig(f'cs_prof_{ds}.png')
    plt.gcf().clear()
    
def do_plot_mask(ds, r, z, mask):
    
    print(f"Plotting mask for {ds}...")
    plt.scatter(r[mask], z[mask], s=0.5)
    plt.xlabel('r [cm]')
    plt.ylabel('z [cm]')
    plt.xlim(r[0,0], r[-1,0])
    plt.ylim(z[0,0], z[0,-1])
    plt.savefig(f'mask_{ds}.png')
    plt.gcf().clear()
    
def do_plot_M_of_v(ds):
    
    print(f"Plotting M(v) for {ds}...")
    plot = yt.ProfilePlot(ds, ('boxlib', 'magvel'), [('gas', 'mass')], weight_field=None)
    plot.save()
    
def calc_2d(ds):
    
    do_mask = not args.no_dist
    
    # Make data object and retrieve position data
    ad = au.AMRData(ds, args.level)
    r, z = ad.position_data(units=False)
    dr, dz = ad.dds[:, args.level].d
    vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
    
    # Get data for relevant fields
    rhoE = ad['rho_E'].d
    rhoe = ad['rho_e'].d
    rho = ad['density'].d
    if do_mask or args.plot_mask or args.plot_cs_prof:
        cs = ad['soundspeed'].d
    
    if args.plot_cs_prof:
        do_plot_cs_prof(ds, r, z, cs)
        
    """
    x = plt.imshow((cs / cs.min()) - 1.15e4, cmap="seismic", vmin=-2.5e5, vmax=2.5e5)
    plt.gcf().colorbar(x, ax=plt.gca())
    plt.show()
    """

    if do_mask:
        of_mask = cs/cs.min() > args.thresh
        ej_mask = ~of_mask

    if args.plot_mask:
        do_plot_mask(ds, r, z, of_mask)
        
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
        
def main_serial():
    
    with open(args.output, 'w') as datfile:
        
        if args.no_dist:
            
            print('#', 't', 'E', 'e', 'M', file=datfile)
            print('#', 's', 'erg', 'erg', 'g', file=datfile)
            
        else:
            
            print('#', 't', 'E_of', 'E_ej', 'e_of', 'e_ej', 'M_of', 'M_ej', file=datfile)
            print('#', 's', 'erg', 'erg', 'erg', 'erg', 'g', 'g', file=datfile)
            
        for ds in ts:
            
            n = ds.dimensionality
            fstr = f"calc_{n}d(ds)"
            vals = eval(fstr)
            
            print(ds.current_time.d, *vals, file=datfile)
            
def main_parallel():
    
    nfields = 4 if args.no_dist else 7
    outputs = np.zeros((len(ts), nfields), dtype=np.float64)
    
    for i, ds in ts.parallel_enumerator():
        
        n = ds.dimensionality
        fstr = f"calc_{n}d(ds)"
        vals = eval(fstr)
        
        outputs[i, 0] = ds.current_time.d
        outputs[i, 1:] = vals
        
    MPI.COMM_WORLD.Reduce(MPI.IN_PLACE, [outputs, MPI.DOUBLE], op=MPI.SUM)
    
    if is_main_proc:
        if args.no_dist:
            header = """t E e M\ns erg erg g"""
        else:
            header = """t E_of E_ej e_of e_ej M_of M_ej\ns erg erg erg erg g g"""
            
        np.savetxt(args.output, outputs, header=header)

if is_main_proc:
    print(f"Data file: {args.output}.")
    print()

if args.use_mpi:
    main_parallel()
else:
    main_serial()

if is_main_proc:
    print("Task completed.")
