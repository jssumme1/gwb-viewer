#import popstock
import gwpopulation as gwpop
from models import MadauDickinsonHeavisideRedshift
gwpop.set_backend('numpy')
from gwpopulation.utils import xp
from popstock.PopulationOmegaGW import PopulationOmegaGW
import json

mass_model = gwpop.models.mass.SinglePeakSmoothedMassDistribution()
redshift_model = MadauDickinsonHeavisideRedshift(z_max=10, cosmo_model='Planck18')

# deal with sample size and save files
N_proposal_samples = int(1e4)
bins = 10
base_samples_name = 'data/proposal_samples'
base_energies_name = 'data/energies'

### SETUP ###

models = {
        'mass_model' : mass_model,
        'redshift_model' : redshift_model,
        }

freqs = xp.arange(10, 2000, 2.5)
redshift_bins = xp.linspace(0, 10, bins+1)

### ITERATE THROUGH REDSHIFT BINS ###

for ii in range(len(redshift_bins)-1):

    zmin = redshift_bins[ii]
    zmax = redshift_bins[ii+1]
    
    samples_file = base_samples_name + f'_z{zmin:.0f}-{zmax:.0f}.json'
    energies_file = base_energies_name + f'_z{zmin:.0f}-{zmax:.0f}.npy'

    # change model params for zmin and zmax
    Lambda_0 =  {'alpha': 3.2, 'beta': 1.2, 'delta_m': 4.5, 'lam': 0.04, 'mmax': 85, 'mmin': 5.21, 
                 'mpp': 33.5, 'sigpp': 3.8, 'gamma': 3.3, 'kappa': 5.6, 'z_peak': 1.9,
                 'zmin': zmin, 'zmax': zmax, 'rate': 13.0}

    newpop = PopulationOmegaGW(models=models, frequency_array=freqs, backend='numpy')


    # calculate omegaGW for this redshift bin
    newpop.draw_and_set_proposal_samples(Lambda_0, N_proposal_samples=N_proposal_samples)
    newpop.calculate_omega_gw(Lambda=Lambda_0, multiprocess=True)
    
    with open(samples_file, "w") as outfile:
        proposal_samples_list = {}
        for key in newpop.proposal_samples:
            proposal_samples_list[key] = list(newpop.proposal_samples[key])
        proposal_samples_list['pdraw'] = list(newpop.pdraws)
        json.dump(proposal_samples_list, outfile)

    # wave_energies is a 2D array
    xp.save(energies_file, newpop.wave_energies)




