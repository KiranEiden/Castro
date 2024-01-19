#!/usr/bin/env python3

import re
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
parser.add_argument('-Rmax', type=float, default=1e14)
parser.add_argument('-beta', type=float, default=3e-2)
parser.add_argument('-t0', type=float, default=0.1)
parser.add_argument('-tramp', type=float, default=0.05)
parser.add_argument('-eps', type=float, default=1e-8)
parser.add_argument('-Pmag', type=float, default=1.0)
parser.add_argument('-tmax', type=float, default=5.0)
parser.add_argument('-fdep', type=float, default=0.03)
parser.add_argument('-depscale', type=float, default=2.0)
parser.add_argument('-chi', type=float, default=1e-3)
parser.add_argument('-Mmag', type=float, default=1.25728227)
parser.add_argument('-mu', type=float, default=1e-3)
parser.add_argument('-res', type=int, default=8192)
parser.add_argument('-ni_mass', type=float, default=0.125)
parser.add_argument('-o_mass', type=float, default=0.25)
parser.add_argument('-blend_thresh', type=float, default=1e-30)
parser.add_argument('-ref_ratio', type=int, default=64)
parser.add_argument('-plot_per', type=float, default=0.25)
parser.add_argument('-small_plot_per', type=float, default=0.01)
parser.add_argument('-out')
parser.add_argument('-template', default="inputs.2d.template")

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
P_mag = args.Pmag * u.ms
t_m = (P_mag / (1*u.ms))**2 * 2047.49866274 * u.s
E_0 = 1.97392088e52 * u.erg / ((P_mag / (1*u.ms))**2)
t_0 = args.t0 * t_m
t_ramp = args.tramp * t_m
t_max = args.tmax * t_m
f_dep = args.fdep
depscale = args.depscale
chi = args.chi
M_mag = args.Mmag * u.Msun
mu = args.mu
res = args.res
dr = R_max / res

ni_mass = args.ni_mass
o_mass = args.o_mass
blend_thresh = args.blend_thresh

base_res_r = res // args.ref_ratio
base_res_z = 2 * base_res_r
assert res % base_res_r == 0

plot_per = args.plot_per * t_m
small_plot_per = args.small_plot_per * t_m

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

p_dep = np.log(chi * f_dep**3 * E_sn / E_0 * (1. - 1 / (t_max/t_m + 1.))) / (1. - depscale)
ebg = np.exp(p_dep * depscale)
eb = np.exp(p_dep)
frac_lum_base = p_dep**3*ebg/eb / (p_dep**3*ebg/eb - 3.*p_dep**2*depscale**2 + 3.*p_dep**2*ebg/eb
        - 6.*p_dep*depscale + 6.*p_dep*ebg/eb + 6.*ebg/eb - 6.)

alph = (6 - d) / (5 - d)
zeta_tr = 2 * (n - 5) * (9 - 2*d) * (11 - 2*d) / ((5 - d)**2 * (n - d) * (3 - d))
t_tr = zeta_tr * E_sn / E_0 * t_m
t_tag = (f_dep*r_0*depscale * t_tr**(alph - 1) / (beta * u.c))**(1/alph)

M_inj = (E_0 * t_max / t_m / (t_max/t_m + 1.) / u.c**2).to(u.g)
v_esc = ((4.0*u.G*M_mag/dr)**0.5).to(u.cm/u.s)
E_K_inj = (0.5 * M_inj * v_esc**2).to(u.erg)
M_mult = (mu*E_0 / E_K_inj).to(u.dimensionless)

dep_cells = int((f_dep * r_0 * depscale) / dr + 0.5)

r_t = f_r * r_0
r = (np.arange(res) + 0.5) * dr
r_1 = r[r < r_t]
r_2 = r[r >= r_t]
M_enc = np.zeros(res)
M_enc[r < r_t] = (4.*np.pi*r_1**3 * rho0_ej * (r_1/r_t)**(-d) / (3. - d)).to(u.Msun).d
M_enc[r >= r_t] = (4.*np.pi*r_t**3 * rho0_ej / (3. - d) + (4.*np.pi*r_2**3 * rho0_ej * (r_2/r_t)**(-n)
        - 4.*np.pi*r_t**3 * rho0_ej) / (3. - n)).to(u.Msun).d
M_enc = M_enc * u.Msun
M_enc = M_enc / M_ej

ni_cells = (M_enc < ni_mass).sum()
o_cells = (M_enc < (ni_mass + o_mass)).sum() - ni_cells
blend_scale = -ni_mass / np.log(blend_thresh)

print(f"{Fore.BLUE}t_max{Style.RESET_ALL}: {Fore.GREEN}{t_max.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}t_ramp{Style.RESET_ALL}: {Fore.GREEN}{t_ramp.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}f_r{Style.RESET_ALL}: {Fore.GREEN}{f_r.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}1/f_r{Style.RESET_ALL}: {Fore.GREEN}{1./f_r.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}r_0{Style.RESET_ALL}: {Fore.GREEN}{r_0.to(u.cm):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}r_t{Style.RESET_ALL}: {Fore.GREEN}{(r_t).to(u.cm):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}rho0_ej{Style.RESET_ALL}: {Fore.GREEN}{rho0_ej.to(u.g/u.cm**3):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}f_a{Style.RESET_ALL}: {Fore.GREEN}{xi.d}{Style.RESET_ALL}")
print(f"{Fore.BLUE}rho_a{Style.RESET_ALL}: {Fore.GREEN}{rho_a.to(u.g/u.cm**3):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}beta_max{Style.RESET_ALL}: {Fore.GREEN}{(beta*1./f_r).d:.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}E_sn{Style.RESET_ALL}: {Fore.GREEN}{E_sn.to(u.erg):.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}p_dep{Style.RESET_ALL}: {Fore.GREEN}{p_dep:.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}frac_lum_base{Style.RESET_ALL}: {Fore.GREEN}{frac_lum_base:.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}t_tag{Style.RESET_ALL}: {Fore.GREEN}{t_tag:.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}M_mult{Style.RESET_ALL}: {Fore.GREEN}{M_mult:.15e}{Style.RESET_ALL}")
print(f"{Fore.BLUE}dep_cells{Style.RESET_ALL}: {Fore.GREEN}{dep_cells}{Style.RESET_ALL}")
print(f"{Fore.BLUE}ni_cells{Style.RESET_ALL}: {Fore.GREEN}{ni_cells}{Style.RESET_ALL}")
print(f"{Fore.BLUE}o_cells{Style.RESET_ALL}: {Fore.GREEN}{o_cells}{Style.RESET_ALL}")
print(f"{Fore.BLUE}blend_scale{Style.RESET_ALL}: {Fore.GREEN}{blend_scale:.15e}{Style.RESET_ALL}")
    
def stringify(num, n=4, imin=4, smallexp=3):
    
    if isinstance(num, int):
        return str(num)
    
    # Round if there is a string of 'n' zeros or 'n' nines
    string = f"{num:.18e}"
    exp = float(string[string.find('e')+1:])
    
    i = string.find("0"*n, imin)
    j = string.find("9"*n, imin)
    k = max(i, j)
    
    if k < imin:
        k = 17
    
    if np.abs(exp) <= smallexp:
        return str(np.round(num, k-2))
    else:
        fstr = "{:.%ie}" % (k-2)
        return fstr.format(num)

def repl(match):
    val = globals()[match['var']]
    if not isinstance(val, u.unyt_quantity):
        return stringify(val)
    if match['unit'] is None:
        val = val.in_cgs()
    else:
        val = val.to(match['unit'])
    return stringify(val.d)

if args.out:
    
    with open(args.template, 'r') as f:
        contents = f.read()
    pat = re.compile('&!(?P<var>\w+)(?:\{(?P<unit>[\w/]+)\})?')
    new_contents = pat.sub(repl, contents)
    
    with open(args.out, 'w') as f:
        f.write(new_contents)
