import streamlit as st
import numpy as np
import json
from reweight_omega_gw import reweight, get_mass_distribution, get_redshift_distribution

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = (0, (1, 5))
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 20

mpl.use('agg')
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


st.set_page_config(layout="centered",
                   page_title="Gravitational-Wave Background Visualizer",
                   page_icon="https://gwosc.org/static/images/icons/gwosc-fav.ico",
                   menu_items = { "Get help": "https://ask.igwn.org"})

st.title('Gravitational-Wave Background Visualizer')

st.markdown("""Visualize the stochastic gravitational-wave background that various populations of stellar-mass binary black holes would create.
""")

st.markdown("""
 * Use the menu on the left to select the merger rate for various bins of redshift.
 * Then also use the menu to select the shape of the mass distribution at each redshift bin.
 * Your plots will appear below.
""")

wz = st.sidebar.slider('Mass Slope Redshift Evolution', -1.0, 1.0, 0.0)  # min, max, default
rate = st.sidebar.slider('Local Merger Rate', 1.0, 50.0, 15.0) 
gamma = st.sidebar.slider('Low-Redshift Power-Law Slope', -3.0, 6.0, 3.0)  
z_peak = st.sidebar.slider('Merger Rate Peak Redshift', 1.0, 5.0, 1.9) 
kappa = st.sidebar.slider('High-Redshift Power-Law Slope', -6.0, 20.0, 5.6) 

### RUN POPSTOCK AND ITERATE THROUGH REDSHIFT BINS ###

@st.cache_data
def load_json(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data

@st.cache_data
def load_npy(file):
    np_data = np.load(file)
    return np_data

freq_grid = xp.arange(10, 2000, 2.5)
redshift_grid = xp.linspace(0, 10, 1000)
mass_grid = xp.linspace(2.01, 100, 1000)
redshift_bins = xp.linspace(0, 10, bins+1)

omegas, psis, masses = [], [], []
for ii in range(len(redshift_bins)-1):

    zmin = redshift_bins[ii]
    zmax = redshift_bins[ii+1]
    
    samples_file = base_samples_name + f'_z{zmin:.0f}-{zmax:.0f}.json'
    energies_file = base_energies_name + f'_z{zmin:.0f}-{zmax:.0f}.npy'

    # load energies and samples file (and cache)
    proposal_samples = load_json(samples_file)
    energies = load_npy(energies_file)
    
    omega_gw = reweight(proposal_samples, energies, zmin, zmax, wz=wz, rate=rate, gamma=gamma, kappa=kappa, z_peak=z_peak)
    dRdm = get_mass_distribution(zmin, zmax, wz=wz, rate=rate)
    Rz = get_redshift_distribution(zmin, zmax, rate=rate, gamma=gamma, kappa=kappa, z_peak=z_peak):

        
### PLOT ###

fig, ax = plt.subplots(1,3)
fig.set_size_inches(22, 6)
cmap = cm.get_cmap('viridis', 10)
norm = mcolors.Normalize(vmin=0, vmax=10)

total_omega_gw = xp.zeros(omegas[0].shape)
dz = 10 / omegas.shape[0]
for ii, line in enumerate(omegas):
    z = ii * dz - dz/2
    total_omega_gw += line
    ax[2].plot(freq_grid, line, lw=1, color=cmap(norm(z)))
    ax[1].plot(mass_grid, xp.log10(mass[ii]) + ii, lw=4, color=cmap(norm(z)))
    ax[0].fill_between(redshift_grid, psis[ii], 0, edgecolor='black', lw=1, ls='-', alpha=0.5, color=cmap(norm(z)))

ax[2].plot(freqs, total_omega_gw, lw=3, color='black', label=r'total $\bar{\Omega}_{\rm GW}$')

ax[0].set_xlim(0, 10)
ax[0].set_ylim(1e-1, 1e2)
ax[0].set_yscale('log')
ax[0].set_xlabel('$z$')
ax[0].set_ylabel(r'$\psi(z)$')
ax[0].text(0.11, 70, r'MDR: $\gamma=3.55$, $z_{\rm peak}=1.98$, $\kappa=4.79$', fontsize=16)

ax[1].set_xlim(2, 100)
ax[1].set_ylim(-4, 10)
ax[1].set_xlabel('$m_{1}$ [$M_{\odot}$]')
ax[1].set_ylabel(r'd$\mathcal{R}$/d$m_{1}$ [Gpc$^{-3}$ yr$^{-1}$ M$^{-1}_{\odot}$]')
ax[1].text(62, 9.2, r'PP: $\alpha = 3.29 - 0.2z$', fontsize=16)
ax[1].set_yticklabels([])

ax[2].set_xlim(1e1, 2e3)
ax[2].set_ylim(1e-14, 1e-8)
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_xlabel('$f$ [Hz]')
ax[2].set_ylabel(r'$\bar{\Omega}_{\rm GW}$')
ax[2].legend(framealpha=0, fontsize=14, handlelength=1.5, loc='upper right')

for axis in ax:
    axis.tick_params(axis='both', which='major', direction='in', 
                     bottom=True, top=True, left=True, right=True)
    axis.tick_params(axis='both', which='minor', direction='in', 
                     bottom=True, top=True, left=True, right=True)
