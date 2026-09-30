# Obtaining the diffusion time scale for a CPD with only gas opacity from Malygin et al. (2014)
import numpy as np
from units_astro import Mj, yr, au, c

day=24*3600 # seconds in a day

def timescale_difussion(M_env, kappa_R, L_box):
    return (M_env*kappa_R)/(c*L_box)

L_blox=0.1*au # the box size
kappa_R=0.01 # Rosseland mean opacity in cm^2/g
M_env=1*Mj # envelope mass in g
t_diff=timescale_difussion(M_env, kappa_R, L_blox)
print("Diffusion time scale: ", t_diff/day, " days")
