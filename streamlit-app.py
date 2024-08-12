import streamlit as st
import numpy as np
from gwpopulation.utils import xp
import json
from reweight_omega_gw import reweight, get_mass_distribution, get_redshift_distribution

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = (0, (1, 5))
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 22

mpl.use('agg')
from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)


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

rate = st.sidebar.slider('Local Merger Rate', 1.0, 50.0, 15.0) 
gamma = st.sidebar.slider('Low-Redshift Power-Law Slope', -3.0, 6.0, 3.0)  
z_peak = st.sidebar.slider('Merger Rate Peak Redshift', 1.0, 5.0, 1.9) 
kappa = st.sidebar.slider('High-Redshift Power-Law Slope', -6.0, 20.0, 5.6) 
wz = st.sidebar.slider('Mass Slope Redshift Evolution', -1.0, 1.0, 0.0)  # min, max, default

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

@st.cache_data
def load_txt(file):
    np_data = np.loadtxt(file, unpack=True, skiprows=1, usecols=(0,1))
    return np_data

bins = 10
freq_grid = xp.arange(10, 2000, 2.5)
redshift_grid = xp.linspace(0, 10, 1000)
mass_grid = xp.linspace(2.01, 100, 1000)
redshift_bins = xp.linspace(0, 10, bins+1)

data_dir = 'data/'
#data_dir = 'https://github.com/jssumme1/gwb-viewer/tree/main/data/'

omegas, psis, masses = [], [], []
for ii in range(len(redshift_bins)-1):

    zmin = redshift_bins[ii]
    zmax = redshift_bins[ii+1]
    
    samples_file = data_dir + f'proposal_samples_z{zmin:.0f}-{zmax:.0f}.json'
    energies_file = data_dir + f'energies_z{zmin:.0f}-{zmax:.0f}.npy'

    # load energies and samples file (and cache)
    proposal_samples = load_json(samples_file)
    energies = load_npy(energies_file)
    
    omega_gw = reweight(proposal_samples, energies, zmin, zmax, wz=wz, rate=rate, gamma=gamma, kappa=kappa, z_peak=z_peak)
    dRdm = get_mass_distribution(zmin, zmax, wz=wz, rate=rate)
    Rz = get_redshift_distribution(zmin, zmax, rate=rate, gamma=gamma, kappa=kappa, z_peak=z_peak)

    omegas.append(omega_gw)
    masses.append(dRdm)
    psis.append(Rz)

omegas = np.array(omegas)
### PLOT ###

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig1.set_size_inches(15, 10)
fig2.set_size_inches(15, 10)
fig3.set_size_inches(15, 10)
cmap = cm.get_cmap('viridis', 10)
norm = mcolors.Normalize(vmin=0, vmax=10)

total_omega_gw = xp.zeros(omegas[0].shape)
dz = 10 / omegas.shape[0]
for ii, line in enumerate(omegas):
    z = ii * dz - dz/2
    total_omega_gw += line
    ax3.plot(freq_grid, line, lw=1, color=cmap(norm(z)))
    ax2.plot(mass_grid, xp.log10(masses[ii]) + ii, lw=4, color=cmap(norm(z)))
    ax1.fill_between(redshift_grid, psis[ii], 0, edgecolor='black', lw=1, ls='-', alpha=0.5, color=cmap(norm(z)))

ax1.plot([0,0], [8,8], alpha=0, label=r'MDR: $\gamma=$' + f' ${gamma}$, ' + r'$z_{\rm peak}=$' + f' ${z_peak}$, ' + r'$\kappa=$' + f' ${kappa}$')
ax2.plot([20, 40], [5, 5], alpha=0, label=r'PP: $\alpha =$' + f' ${3.2} {wz:+.2f}z$')
ax3.plot(freq_grid, total_omega_gw, lw=3, color='black', label=r'total $\bar{\Omega}_{\rm GW}$')

# Load PI curve data
freqs_O3, PI_O3 = load_txt(data_dir + 'PICurve_O3.dat')
freqs_design, PI_design = load_txt(data_dir + 'PICurve_HLV_Design.dat')
freqs_aplus, PI_aplus = load_txt(data_dir + 'PICurve_Aplus_Design.dat')

ax3.plot(freqs_O3,2.*PI_O3,color='black',label='O3 Sensitivity',zorder=20)
ax3.plot(freqs_design,2.*PI_design,color='#5e5e5e',dashes=(2,1.5),label='Design HLV',lw=1.2,zorder=20)
ax3.plot(freqs_aplus,2.*PI_aplus,color='#5e5e5e',dashes=(3,1,1,1),label='Design A+',lw=1.2,zorder=20)

ax1.set_xlim(0, 10)
ax1.set_yscale('log')
ax1.set_xlabel('$z$')
ax1.set_ylabel(r'$\mathcal{R}(z)$ [Gpc$^{-3}$ yr$^{-1}$]')

ax2.set_xlim(2, 100)
ax2.set_ylim(-4, 10)
ax2.set_xlabel('$m_{1}$ [$M_{\odot}$]')
ax2.set_ylabel(r'd$\mathcal{R}$/d$m_{1}$ [Gpc$^{-3}$ yr$^{-1}$ M$^{-1}_{\odot}$]')
ax2.set_yticklabels([])

ax3.set_xlim(1e1, 2e3)
ax3.set_ylim(1e-13, 2e-7)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('$f$ [Hz]')
ax3.set_ylabel(r'$\bar{\Omega}_{\rm GW}$')
ax1.legend(framealpha=0, handlelength=1.5, loc='upper right')
ax2.legend(framealpha=0, handlelength=1.5, loc='upper right')
ax3.legend(framealpha=0, handlelength=1.5, loc='upper right')

for axis in [ax1, ax2, ax3]:
    axis.tick_params(axis='both', which='major', direction='in', 
                     bottom=True, top=True, left=True, right=True)
    axis.tick_params(axis='both', which='minor', direction='in', 
                     bottom=True, top=True, left=True, right=True)


st.subheader('Merger Rate vs Redshift')

st.pyplot(fig1)


with st.expander("See notes"):
    st.markdown(r"""
 * This plot shows how many binary black hole coalescences occur per cubic gigaparsec, per year, as a function of redshift. 
 * Here we parameterize the redshift model with the Madau & Dickinson star-formation rate model, which slopes upward (power-law of $\gamma$), then peaks (at redshift $z_{\rm peak}$), then slopes downward (with power-law slope $\kappa$, unless you mess with the parameters!). You can change these parameters with the panel on the left.
 * Each redshift bin (of width 1) is colored in a way that corresponds to the colored lines in the following plots.
""")

st.subheader('Differential Merger Rate vs Primary Mass')

st.pyplot(fig2)

with st.expander("See notes"):
    st.markdown(r"""
 * This plot shows the distribution of primary (larger of the two black holes) masses in our population model. The area under the curve between masses on the x-axis corresponds to the merger rate (like the above plot). The height here is arbitrary, however, as curves corresponding to high-redshift bins are shifted upwards to increase visibility.
 * The population model we use here is known as the Power Law + Peak model. It parameterizes the distribution of primary black hole masses as a power-law ($m_1^{-\alpha}$) plus a peak (a Gaussian distribution), and smooths out the low-mass behavior (making it look like there is also a peak around $7~{\rm M}_{\odot}$). The mass ratio (smaller black hole mass / larger black hole mass) is also parameterized with a power-law ($q^\beta$).
 * You can change the evolution of the parameter $\alpha$ (the power-law slope of the primary mass model) with the panel on the left. Notice how the curves change in slope with a different redshift evolution.
""")

st.subheader('GW Background Energy Density vs Frequency')

st.pyplot(fig3)

with st.expander("See notes"):
    st.markdown(r"""
 * This plot shows predicted stochastic gravitational-wave background from the population model displayed in the above two plots. The GW background is measured by $\bar{\Omega}_{\rm GW}$, which is equal to the overall energy density of ALL gravitational waves from coalescing stellar-mass binary black holes, divided by the critical energy density to make the universe flat. The GW background is not very energetic, because it is only around one part in 50,000 of the energy density from the photons we now observe from the cosmic microwave background.
 * There are a few factors that can affect how much of the GW background is from high-redshift binary black holes. Since the energy from GWs decays with the inverse square of the distance, high-redshift binary black holes will individually contribute less energy to the GW background. However, at high-redshifts, the volume of the universe increases, so there are more binary black hole coalescences. Additionally, if in the first figure, the merger rate has a positive slope at high-redshifts, then there will be an even greater contribution to the GW background from high-redshifts.
 * Another thing to note is that at large cosmological distances, GWs are redshifted to lower frequencies. You can see this in the GW background curves, as the high-redshift curves are shifted to the left. Also, larger-mass binary black hole systems will show up in the low-frequency area of the GW background because larger black holes are able to "touch" each others' event horizons earlier in their inspiral phase.
""")

st.subheader("About this app")
st.markdown("""
You can see how this works in the [GitHub repository](https://github.com/jssumme1/gwb-viewer).
""")


