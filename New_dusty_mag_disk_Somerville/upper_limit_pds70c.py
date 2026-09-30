# Upper limit for pds70c
from numpy import log10
from flux_gas_dust_mag_disk import F_nu as F
from magneticCPD import R_T1000, R_truncation
import matplotlib.pyplot as plt
import numpy as np
from units_astro import *
from nautilus import Prior, Sampler
from errors import error_param, upper_limit, strict_upper_limit
import corner

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

upper_lim=sFnus[2]*3 # 3 sigma upper limit for the last point

Bps=10**(2.796625406817652) # G
dust_to_gas=10**(-10.067572749115545)

def upper_limit_F343(Bp, dust_to_gas):
    log_Mpdot_inf=-8
    log_Mpdot_up=-4
    while log_Mpdot_up-log_Mpdot_inf>0.01:
        log_Mpdot=(log_Mpdot_inf+log_Mpdot_up)/2
        if F(nu[2], 10**log_Mpdot*Mj/yr, Bp, dust_to_gas)>upper_lim:
            log_Mpdot_up=log_Mpdot
        else:
            log_Mpdot_inf=log_Mpdot
    return log_Mpdot_up

log_Mpdot_up=upper_limit_F343(Bps, dust_to_gas)
print("Upper limit for log_Mpdot:", log_Mpdot_up)

print('Using Bps =', Bps, 'G', 'and zeta =', dust_to_gas)
print('flux at 343 GHz for the upper limit Mpdot:', F(nu[2], 10**log_Mpdot_up*Mj/yr, Bps, dust_to_gas), 'mJy')

# and for 200 G

Bps=200
log_Mpdot_up=upper_limit_F343(Bps, dust_to_gas)
print("Upper limit for log_Mpdot with Bps=200 G:", log_Mpdot_up)
print('flux at 343 GHz for the upper limit Mpdot with Bps=200 G:', F(nu[2], 10**log_Mpdot_up*Mj/yr, Bps, dust_to_gas), 'mJy')