from gwpopulation.models.redshift import MadauDickinsonRedshift

class MadauDickinsonHeavisideRedshift(MadauDickinsonRedshift):
    base_variable_names = ["gamma", "z_peak", "kappa", "zmin", "zmax"]
    
    def psi_of_z(self, redshift, **parameters):
        zmin = parameters["zmin"]
        zmax = parameters["zmax"]
        
        full_psi_of_z = super().psi_of_z(redshift, **parameters)
        return full_psi_of_z * (redshift >= zmin) * (redshift <= zmax)
