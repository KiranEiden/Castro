#!/usr/bin/env python3
"""
Read and plot 2D cylindrical kilonova ejecta model at t=1 minute.

This script shows you how to read the HDF5 model file and visualize
the ejecta properties in cylindrical coordinates (r_cyl, z).

=============================================================================
FILE FORMAT: model_2D_cylindrical_1min.h5
=============================================================================

Datasets:
---------
  rho        (nr, nz)         Density [g/cm^3]
  temp       (nr, nz)         Temperature [K]
  vx         (nr, nz)         Physical velocity in r_cyl direction [cm/s]
  vz         (nr, nz)         Physical velocity in z direction [cm/s]
  v_radial   (nr, nz)         Radial (spherical) velocity magnitude [cm/s]
  comp       (nr, nz, n_elem) Mass fractions X(Z) for each element
  erad       (nr, nz)         Radiation energy density (initialized to 0)
  Z          (n_elem,)        Atomic numbers of elements
  A          (n_elem,)        Mass numbers of elements
  x_out      (nr,)            Outer edge of each r_cyl cell [cm]
  z_out      (nz,)            Outer edge of each z cell [cm]
  rmin       (2,)             [r_cyl_min, z_min] = [0, z_min] [cm]
  time       (1,)             Model time [s] (= 60 s = 1 minute)

Attributes:
-----------
  rho_floor       Floor density value [g/cm^3]
  temp_floor      Floor temperature value [K]
  total_mass_Msun Total ejecta mass [M_sun]
  T_HYDRO         Hydrodynamic time [s]
  velocity_type   'physical_real_velocity'
  velocity_note   Description of velocity storage

Grid:
-----
  - Cylindrical coordinates: (r_cyl, z) where r_cyl >= 0, z spans both hemispheres
  - Grid is uniform in both dimensions
  - Cell edges: r_edges = [0, x_out], z_edges = [rmin[1], z_out]
  - Cell centers: midpoints of edges

Velocity:
---------
  IMPORTANT: vx and vz store PHYSICAL velocities, NOT velocity coordinates!
  - vx = v_radial * sin(theta)  [cylindrical radial component]
  - vz = v_radial * cos(theta)  [vertical component]
  - These are the actual fluid velocities from the SNEC simulation at t=1 min
  - Do NOT assume homologous expansion (v = r/t)

Composition:
------------
  - comp[:,:,k] is the mass fraction of element with atomic number Z[k]
  - Elements are sorted by Z (Z[0]=1 for H, Z[1]=2 for He, etc.)
  - Lanthanides: 57 <= Z <= 71
  - Sum of mass fractions = 1 

Units:
------
  - Length: cm
  - Velocity: cm/s
  - Density: g/cm^3
  - Temperature: K
  - Time: s
  - Mass fractions: dimensionless

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

# =============================================================================
# SETTINGS
# =============================================================================
MODEL_FILE = "model_2D_cylindrical_1min.h5"

# Physical constants
M_SUN = 1.989e33   # g
C_LIGHT = 2.998e10  # cm/s

# Floor value for masking (cells below this are "empty")
RHO_FLOOR = 1e-40


# =============================================================================
# READ MODEL
# =============================================================================
print("=" * 70)
print("Reading 2D Cylindrical Model")
print("=" * 70)

with h5py.File(MODEL_FILE, "r") as f:
    # Grid
    x_out = f["x_out"][:]      # Outer r_cyl edges (nr,)
    z_out = f["z_out"][:]      # Outer z edges (nz,)
    rmin = f["rmin"][:]        # [r_min, z_min]
    time = f["time"][0]        # Model time [s]

    # Properties
    rho = f["rho"][:]          # Density (nr, nz)
    temp = f["temp"][:]        # Temperature (nr, nz)
    vx = f["vx"][:]            # v_r_cyl (nr, nz)
    vz = f["vz"][:]            # v_z (nr, nz)
    v_radial = f["v_radial"][:] if "v_radial" in f else np.sqrt(vx**2 + vz**2)
    comp = f["comp"][:]        # Composition (nr, nz, n_elem)

    # Element info
    Z = f["Z"][:]              # Atomic numbers
    A = f["A"][:]              # Mass numbers

    # Attributes
    rho_floor = f.attrs.get("rho_floor", 1e-40)
    total_mass = f.attrs.get("total_mass_Msun", 0)

nr, nz = rho.shape
n_elem = len(Z)

# Reconstruct grid edges and centers
r_edges = np.concatenate([[rmin[0]], x_out])  # (nr+1,)
z_edges = np.concatenate([[rmin[1]], z_out])  # (nz+1,)

r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])  # (nr,)
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])  # (nz,)

dr = r_edges[1] - r_edges[0]
dz = z_edges[1] - z_edges[0]

# Create 2D coordinate arrays
R_cyl, Z_coord = np.meshgrid(r_centers, z_centers, indexing='ij')
R_sph = np.sqrt(R_cyl**2 + Z_coord**2)  # Spherical radius

# Mask for cells with actual data (not floor values)
mask_data = rho > rho_floor * 10

# Compute cell volumes and masses
V_cell = 2 * np.pi * R_cyl * dr * dz  # Cylindrical shell volume
M_cell = rho * V_cell
M_total = np.sum(M_cell[mask_data])

# Compute composition groups
X_light = np.sum(comp[:, :, Z < 57], axis=2)           # Z < 57
X_lan = np.sum(comp[:, :, (Z >= 57) & (Z <= 71)], axis=2)  # Lanthanides
X_heavy = np.sum(comp[:, :, Z > 71], axis=2)           # Z > 71 (actinides, etc.)

# Print summary
print(f"\nModel file: {MODEL_FILE}")
print(f"Model time: {time:.1f} s = {time/60:.2f} min")
print(f"\nGrid: {nr} x {nz} cells")
print(f"  r_cyl: [{r_edges[0]:.3e}, {r_edges[-1]:.3e}] cm")
print(f"  z:     [{z_edges[0]:.3e}, {z_edges[-1]:.3e}] cm")
print(f"  dr = {dr:.3e} cm, dz = {dz:.3e} cm")
print(f"\nNumber of elements: {n_elem}")
print(f"  Z range: {Z.min()} to {Z.max()}")
print(f"\nTotal mass: {M_total/M_SUN:.6f} M_sun")
print(f"  (from file attribute: {total_mass:.6f} M_sun)")
print(f"\nCells with data: {np.sum(mask_data)} / {nr*nz}")
print(f"\nDensity range (data cells): {rho[mask_data].min():.3e} - {rho[mask_data].max():.3e} g/cm^3")
print(f"Temperature range (data cells): {temp[mask_data].min():.1f} - {temp[mask_data].max():.1e} K")
print(f"Velocity range (data cells): {v_radial[mask_data].min()/C_LIGHT:.4f} - {v_radial[mask_data].max()/C_LIGHT:.4f} c")


# =============================================================================
# PLOT: 2D HYDRO PROPERTIES
# =============================================================================
print("\n" + "=" * 70)
print("Creating plots...")
print("=" * 70)

# Extent for imshow (in units of 10^14 cm)
scale = 1e14
extent = [r_edges[0]/scale, r_edges[-1]/scale,
          z_edges[0]/scale, z_edges[-1]/scale]

# Colormaps with white for masked regions
cmap_viridis = cm.viridis.copy()
cmap_viridis.set_bad('white')
cmap_inferno = cm.inferno.copy()
cmap_inferno.set_bad('white')
cmap_coolwarm = cm.coolwarm.copy()
cmap_coolwarm.set_bad('white')

# Figure 1: Hydro properties
fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
fig1.suptitle(f'2D Cylindrical Model (t = {time/60:.0f} min): Hydro Properties',
              fontsize=14, fontweight='bold')

# Panel 1: Density
ax = axes[0, 0]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, rho)
im = ax.imshow(np.log10(data_plot.T + 1e-50), origin='lower', aspect='equal',
               extent=extent, cmap=cmap_viridis)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$\log_{10}(\rho)$ [g/cm$^3$]')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 2: Temperature
ax = axes[0, 1]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, temp)
im = ax.imshow(np.log10(data_plot.T + 1), origin='lower', aspect='equal',
               extent=extent, cmap=cmap_inferno)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$\log_{10}(T)$ [K]')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 3: Velocity magnitude
ax = axes[0, 2]
ax.set_facecolor('white')
v_mag = np.sqrt(vx**2 + vz**2)
data_plot = np.ma.masked_where(~mask_data, v_mag / C_LIGHT)
im = ax.imshow(data_plot.T, origin='lower', aspect='equal',
               extent=extent, cmap=cmap_viridis, vmin=0, vmax=0.5)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$|v|/c$')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 4: v_r_cyl
ax = axes[1, 0]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, vx / C_LIGHT)
im = ax.imshow(data_plot.T, origin='lower', aspect='equal',
               extent=extent, cmap=cmap_coolwarm, vmin=-0.3, vmax=0.3)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$v_{r,\rm cyl}/c$')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 5: v_z
ax = axes[1, 1]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, vz / C_LIGHT)
im = ax.imshow(data_plot.T, origin='lower', aspect='equal',
               extent=extent, cmap=cmap_coolwarm, vmin=-0.3, vmax=0.3)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$v_z/c$')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 6: Lanthanide fraction
ax = axes[1, 2]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, X_lan)
im = ax.imshow(np.log10(data_plot.T + 1e-10), origin='lower', aspect='equal',
               extent=extent, cmap=cmap_inferno, vmin=-4, vmax=0)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$\log_{10}(X_{\rm lan})$')
plt.colorbar(im, ax=ax, shrink=0.8)

fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('plot_2D_hydro.png', dpi=150, bbox_inches='tight')
print(f"  Saved: plot_2D_hydro.png")


# Figure 2: Composition
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(f'2D Cylindrical Model (t = {time/60:.0f} min): Composition',
              fontsize=14, fontweight='bold')

# Panel 1: Light elements (Z < 57)
ax = axes[0]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, X_light)
im = ax.imshow(data_plot.T, origin='lower', aspect='equal',
               extent=extent, cmap=cmap_viridis, vmin=0, vmax=1)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$X$(Z < 57) Light elements')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 2: Lanthanides (57 <= Z <= 71)
ax = axes[1]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, X_lan)
im = ax.imshow(np.log10(data_plot.T + 1e-10), origin='lower', aspect='equal',
               extent=extent, cmap=cmap_inferno, vmin=-4, vmax=0)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$\log_{10}(X_{\rm lan})$ Lanthanides')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 3: Heavy elements (Z > 71)
ax = axes[2]
ax.set_facecolor('white')
data_plot = np.ma.masked_where(~mask_data, X_heavy)
im = ax.imshow(np.log10(data_plot.T + 1e-10), origin='lower', aspect='equal',
               extent=extent, cmap=cmap_inferno, vmin=-4, vmax=0)
ax.set_xlabel(r'$r_{\rm cyl}$ [$10^{14}$ cm]')
ax.set_ylabel(r'$z$ [$10^{14}$ cm]')
ax.set_title(r'$\log_{10}(X)$ Heavy (Z > 71)')
plt.colorbar(im, ax=ax, shrink=0.8)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig('plot_2D_composition.png', dpi=150, bbox_inches='tight')
print(f"  Saved: plot_2D_composition.png")


# =============================================================================
# PRINT DATA FORMAT SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("DATA FORMAT SUMMARY")
print("=" * 70)
print("""
To read this file in your code:

    import h5py
    import numpy as np

    with h5py.File('model_2D_cylindrical_1min.h5', 'r') as f:
        # Grid (IMPORTANT: use these for spatial coordinates)
        x_out = f['x_out'][:]   # Outer r_cyl edges [cm]
        z_out = f['z_out'][:]   # Outer z edges [cm]
        rmin  = f['rmin'][:]    # [r_min, z_min] [cm]
        time  = f['time'][0]    # Model time [s]

        # Reconstruct cell centers
        r_edges = np.concatenate([[rmin[0]], x_out])
        z_edges = np.concatenate([[rmin[1]], z_out])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Properties
        rho  = f['rho'][:]       # Density [g/cm^3]
        temp = f['temp'][:]      # Temperature [K]
        vx   = f['vx'][:]        # v_r_cyl [cm/s] (PHYSICAL velocity)
        vz   = f['vz'][:]        # v_z [cm/s] (PHYSICAL velocity)
        comp = f['comp'][:]      # Mass fractions (nr, nz, n_elem)
        Z    = f['Z'][:]         # Atomic numbers
        A    = f['A'][:]         # Mass numbers
""")

plt.show()
print("\nDone.")
