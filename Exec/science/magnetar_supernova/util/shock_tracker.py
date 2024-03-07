#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
import analysis_util as au
import matplotlib.pyplot as plt
from skimage.filters import meijering, sato, frangi
from scipy.interpolate import RegularGridInterpolator

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('-s', '--sigmas', nargs='+', type=float, default=[1])
parser.add_argument('-x', '--xlim', nargs=2, type=float)
parser.add_argument('-y', '--ylim', nargs=2, type=float)
parser.add_argument('-f', '--func', default='frangi')
parser.add_argument('--summary', action='store_true')
parser.add_argument('--no_calc', action='store_true')
parser.add_argument('-of', '--outfile', default="shock_pos.h5")
parser.add_argument('-od', '--outdir', default='')
parser.add_argument('-t', '--thresh', type=float, default=0.2)
parser.add_argument('-n', '--num_ang', type=int, default=100)
parser.add_argument('--use_mpi', action='store_true')
args = parser.parse_args()

if args.use_mpi:
    MPI = au.mpi_importer()
is_main_proc = (not args.use_mpi) or (MPI.COMM_WORLD.Get_rank() == 0)

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")
if args.no_calc and (not args.summary):
    sys.exit("Nothing to do; double-check arguments.")

if is_main_proc:
    print("Will load the following files: {}\n".format(ts))

ts = au.FileLoader(ts, args.use_mpi, def_decomp='block')

if is_main_proc:
    if not args.outdir:
        args.outdir = os.getcwd()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
def get_log_rho(ds, return_pos_data=False):
    
    ad = au.AMRData(ds, args.level)
    if return_pos_data:
        r, z = ad.position_data(units=False)
    log_rho = np.log10(ad['density'].d)
    
    if args.xlim or args.ylim:
        
        if not args.xlim:
            args.xlim = ad.left_edge[0], ad.right_edge[0]
        if not args.ylim:
            args.ylim = ad.left_edge[1], ad.right_edge[1]
            
        log_rho, _ = ad.select_region(log_rho, *args.xlim, *args.ylim)
        if return_pos_data:
            r, _ = ad.select_region(r, *args.xlim, *args.ylim)
            z, _ = ad.select_region(z, *args.xlim, *args.ylim)
        
    if return_pos_data:
        return log_rho, r, z
    else:
        return log_rho

def plot_summary(log_rho):
    
    meij_d = meijering(log_rho, sigmas=args.sigmas, black_ridges=False)
    sato_d = sato(log_rho, sigmas=args.sigmas, black_ridges=False)
    fran_d = frangi(log_rho, sigmas=args.sigmas, black_ridges=False)
    
    fig, axes = plt.subplots(2, 2, sharey='row', sharex='col')
    (ax1, ax2), (ax3, ax4) = axes
    
    map1 = ax1.imshow(log_rho)
    map2 = ax2.imshow(meij_d)
    map3 = ax3.imshow(sato_d)
    map4 = ax4.imshow(fran_d)
    
    fig.colorbar(map1, ax=ax1, location="bottom", orientation="horizontal")
    fig.colorbar(map2, ax=ax2, location="bottom", orientation="horizontal")
    fig.colorbar(map3, ax=ax3, location="top", orientation="horizontal")
    fig.colorbar(map4, ax=ax4, location="top", orientation="horizontal")
    
    fig.set_size_inches((18, 12))
    sigstr = '_'.join(map(str, args.sigmas))
    plt.savefig(os.path.join(args.outdir, f"{ds}_ridges_sigma_{sigstr}.png"), bbox_inches='tight')
    fig.clear()
    
def get_1d_rays(theta, r, z, data):
    
    # Get 1d list of r values
    r1d = r[:,0]
    
    # Fixed resolution rays don't work in 2d with my yt version
    interp = RegularGridInterpolator((r1d, z[0]), data, bounds_error=False, fill_value=None)
    xi = np.column_stack((np.sin(theta), np.cos(theta)))
    
    rays = np.empty((len(theta), len(r1d)))
    for i in range(len(r1d)):
        pts = interp(r1d[i] * xi)
        rays[:, i] = pts
        
    return rays
    
def do_pos_calc(log_rho, r, z, theta):
        
    func = globals()[args.func]
    res = func(log_rho, sigmas=args.sigmas, black_ridges=False)
    rays = get_1d_rays(theta, r, z, res)
    
    peak_loc = rays.argmax(axis=1)
    bin_rays = (rays > args.thresh)
    inner_loc = bin_rays.argmax(axis=1)
    outer_loc = rays.shape[1] - bin_rays[:, ::-1].argmax(axis=1) - 1
    
    r1d = r[:,0]
    return r1d[peak_loc], r1d[inner_loc], r1d[outer_loc]
    
def write_dataset(times, theta, pos):
    
    file = h5py.File(os.path.join(args.outdir, args.outfile), 'w')
    file.create_dataset("time", data=times, dtype='d')
    file.create_dataset("ang", data=theta, dtype='d')
    pos_grp = file.create_group("shock_pos")
    pos_grp.create_dataset("r_peak", data=pos[0], dtype='d')
    pos_grp.create_dataset("r_inner", data=pos[1], dtype='d')
    pos_grp.create_dataset("r_outer", data=pos[2], dtype='d')
    file.close()
    
def main_serial():
    
    if args.no_calc:
        for ds in ts:
            log_rho = get_log_rho(ds)
            plot_summary(log_rho)
        return
        
    theta = np.linspace(0.0, np.pi, num=args.num_ang)
    times = np.empty((len(ts),))
    pos = np.empty((3, len(ts), len(theta)))

    for i, ds in enumerate(ts):
        
        times[i] = ds.current_time.d
        log_rho, r, z = get_log_rho(ds, True)
        r_peak, r_inner, r_outer = do_pos_calc(log_rho, r, z, theta)
        pos[0, i, :] = r_peak
        pos[1, i, :] = r_inner
        pos[2, i, :] = r_outer
            
        if args.summary:
            plot_summary(log_rho)
            
    write_dataset(times, theta, pos)
    
def main_parallel():
    
    if args.no_calc:
        for ds in ts:
            log_rho = get_log_rho(ds)
            plot_summary(log_rho)
        return
    
    theta = np.linspace(0.0, np.pi, num=args.num_ang)
    nfields = len(theta)*3 + 1
    start, stop, _ = ts.do_decomp()
    
    bufsize = stop - start
    bufsize_arr = np.zeros(MPI.COMM_WORLD.Get_size() - 1, np.int32)
    bufsize_reqs = []
        
    if is_main_proc:
        for i in range(len(bufsize_arr)):
            req = MPI.COMM_WORLD.Irecv([bufsize_arr[i:i+1], 1, MPI.INT], i+1)
            bufsize_reqs.append(req)
    else:
        req = MPI.COMM_WORLD.Isend([stop-start, 1, MPI.INT], 0)
        bufsize_reqs.append(req)
        
    if not bufsize > 0:
        return
    
    outbuf_loc = np.empty((stop - start) * nfields, dtype=np.float64)
    outview_loc = outbuf_loc.reshape((stop - start, nfields))
    if is_main_proc:
        outbuf_glob = np.empty(len(ts) * nfields, dtype=np.float64)
        outview_glob = outbuf_glob.reshape((len(ts), nfields))
    
    for i, ds in enumerate(ts):
    
        log_rho, r, z = get_log_rho(ds, True)
        r_peak, r_inner, r_outer = do_pos_calc(log_rho, r, z, theta)
    
        outview_loc[i, 0] = ds.current_time.d
        outview_loc[i, 1:] = r_peak
        outview_loc[i, (1+len(theta)):] = r_inner
        outview_loc[i, (1+2*len(theta)):] = r_outer
        
        if args.summary:
            plot_summary(log_rho)
    
    MPI.Request.Waitall(bufsize_reqs)
    
    if is_main_proc:
        
        data_reqs = []
        outbuf_glob[:len(outbuf_loc)] = outbuf_loc
        
        for i in range(len(bufsize_arr)):
            
            if bufsize_arr[i] < 1:
                continue
                
            bufstart = len(outbuf_loc) + bufsize_arr[:i].sum() * nfields
            bufend = bufstart + bufsize_arr[i]*nfields
            req = MPI.COMM_WORLD.Irecv([outbuf_glob[bufstart:bufend], MPI.DOUBLE], i+1)
            data_reqs.append(req)
            
    else:
        
        MPI.COMM_WORLD.Isend([outbuf_loc, MPI.DOUBLE], 0)
    
    if is_main_proc:
    
        times = outview_glob[:, 0]
        pos = outview_glob[:, 1:].reshape((len(ts), 3, len(theta)))
        pos = np.swapaxes(pos, 0, 1)
        write_dataset(times, theta, pos)
        
if is_main_proc:
    print(f"Data file: {args.outfile}.")
    print()

if args.use_mpi and MPI.COMM_WORLD.Get_size() > 1:
    main_parallel()
else:
    main_serial()

if is_main_proc:
    print("Task completed.")
