import gwpopulation as gwpop
from models import MadauDickinsonHeavisideRedshift
gwpop.set_backend('numpy')
from gwpopulation.utils import xp
from popstock.PopulationOmegaGW import PopulationOmegaGW
import json

def reweight(proposal_samples, energies, zmin, zmax, wz=0, rate=13.0, gamma=3.3, kappa=5.6, z_peak=1.9):
    mass_model = gwpop.models.mass.SinglePeakSmoothedMassDistribution()
    redshift_model = MadauDickinsonHeavisideRedshift(z_max=10, cosmo_model='Planck18')
    
    models = {
              'mass_model' : mass_model,
              'redshift_model' : redshift_model,
              }

    # settings
    freqs = xp.arange(10, 2000, 2.5)
    N_proposal_samples = int(1e4)
    bins = 10
    
    # initialize object
    newpop = PopulationOmegaGW(models=models, frequency_array=freqs, backend='numpy')
    
    zmid = (zmax + zmin) / 2
    
    # change population hyper-parameters
    Lambda_0 =  {'alpha': 3.2 + wz * zmid, 'beta': 1.2, 'delta_m': 4.5, 'lam': 0.04, 'mmax': 85, 'mmin': 5.21, 
                 'mpp': 33.5, 'sigpp': 3.8, 'gamma': gamma, 'kappa': kappa, 'z_peak': z_peak,
                 'zmin': zmin, 'zmax': zmax, 'rate': rate}

    # proposal samples
    for key in proposal_samples:
        proposal_samples[key] = xp.array(proposal_samples[key])
    newpop.set_proposal_samples(proposal_samples=proposal_samples)
    # wave energies
    newpop.wave_energies = energies
    newpop.wave_energies_calculated = True

    # run this after loading wave energies
    newpop.calculate_omega_gw(Lambda=Lambda_0)
    
    return newpop.omega_gw

def get_mass_distribution(zmin, zmax, wz=0, rate=13.0):
    mass_model = gwpop.models.mass.SinglePeakSmoothedMassDistribution()
    mass_grid = xp.linspace(2.01, 100, 1000)
    
    zmid = (zmax + zmin) / 2
    params =  {'alpha': 3.2 + wz * zmid, 'delta_m': 4.5, 'lam': 0.04, 
               'mmax': 85, 'mmin': 5.21, 'mpp': 33.5, 'sigpp': 3.8}
    
    dRdm = mass_model.p_m1({'mass_1': mass_grid}, **params) * rate
    return dRdm

def get_redshift_distribution(zmin, zmax, rate=13.0, gamma=3.3, kappa=5.6, z_peak=1.9):
    redshift_model = gwpop.models.redshift.MadauDickinsonHeavisideRedshift(z_max=10, cosmo_model='Planck18')
    redshift_grid = xp.linspace(0, 10, 1000)
    
    params =  {'gamma': gamma, 'kappa': kappa, 'z_peak': z_peak,
               'zmin': zmin, 'zmax': zmax}
    
    Rz = redshift_model.psi_of_z(redshift_grid, **params) * rate
    return Rz