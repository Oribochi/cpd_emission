# Calculating the diffusion time scale for Casassus et. al. 2025 PDS 70c variability paper
from units_astro import *
# the difución time scale is t=M_env*kappa_rosseland/(c L_box)
L_box=0.1*au
M_env=1*Mj
kappa_rosseland=10**(-2) # cm^2/g

t_diffusion=M_env*kappa_rosseland/(c*L_box)
print(f"Diffusion time scale is t_diffusion={t_diffusion/3600:.2f} hours")