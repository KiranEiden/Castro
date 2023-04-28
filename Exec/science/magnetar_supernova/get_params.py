#!/usr/bin/env python3

import sys
import argparse
import unyt as u
import numpy as np
from colorama import Fore, Style, init

init()

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=float, default=1)
parser.add_argument('-n', type=float, default=10)
parser.add_argument('-s', type=float, default=0)
parser.add_argument('-Mej', type=float, default=1.0)
parser.add_argument('-Mcsm', type=float, default=1e-2)
parser.add_argument('-Mdot', type=float, default=1e-5)
parser.add_argument('-vw', type=float, default=10.0)
parser.add_argument('-Rmax', type=float, default=5e14)
parser.add_argument('-beta', type=float, default=3e-2)
parser.add_argument('-t0', type=float, default=1e4)
parser.add_argument('-eps', type=float, default=1e-8)

args = parser.parse_args(sys.argv[1:])
print("Arguments are:")
print(args)
print()

d = args.d
n = args.n
s = args.s

M_ej = args.Mej * u.Msun
M_csm = args.Mcsm * u.Msun
Mdot = args.Mdot * u.Msun / u.yr
v_w = args.vw * u.km / u.s
R_max = args.Rmax * u.cm
beta = args.beta
t_0 = args.t0 * u.s

delt = 1.0; f_r = 0.25

while delt > args.eps:
    
    r_0 = 1./f_r * beta*u.c * t_0
    c_rho = 1. / (3. - d) + (1. - f_r**(n-3)) / (n - 3.)
    rho0_ej = M_ej / (4.*np.pi*(f_r*r_0)**3) / c_rho
    
    xi = M_csm * v_w * (3. - s) / (Mdot * r_0 * ((R_max/r_0)**(3-s) - 1.))
    rho_a = Mdot / (4.*np.pi * v_w * r_0**2) * xi
    delt = f_r - (rho_a / rho0_ej)**(1./n)
    f_r = f_r - delt
    
zeta = ((n-3) * (3-d)) / ((n-5) * (5-d)) * ((n - d - (5-d)*f_r**(n-5)) / (n - d - (3-d)*f_r**(n-3)))
E_sn = 0.5 * M_ej * zeta * (beta * u.c)**2
    
print(f"{Fore.BLUE}f_r{Style.RESET_ALL}: {Fore.GREEN}{f_r.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}1/f_r{Style.RESET_ALL}: {Fore.GREEN}{1./f_r.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}r_0{Style.RESET_ALL}: {Fore.GREEN}{r_0.to(u.cm):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}r_t{Style.RESET_ALL}: {Fore.GREEN}{(r_0*f_r).to(u.cm):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}rho0_ej{Style.RESET_ALL}: {Fore.GREEN}{rho0_ej.to(u.g/u.cm**3):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}f_a{Style.RESET_ALL}: {Fore.GREEN}{xi.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}rho_a{Style.RESET_ALL}: {Fore.GREEN}{rho_a.to(u.g/u.cm**3):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}beta_max{Style.RESET_ALL}: {Fore.GREEN}{(beta*1./f_r).d:.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}E_sn{Style.RESET_ALL}: {Fore.GREEN}{E_sn.to(u.erg):.15e}{Style.RESET_ALL}")
