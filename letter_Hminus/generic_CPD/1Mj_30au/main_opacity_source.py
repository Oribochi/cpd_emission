# This is the code with all the functions to calculate the optical depth in which the maximun contribution to the flux comes from
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
import matplotlib.pyplot as plt

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

from tau_opacity_calc import *

# This function returns the radius at which the maximum contribution to the flux comes from, 
# and main source of optical depth (H-, Metals, H2-) at that radius

def main_opacity_source(nu, Mpdot, Bps, dust_to_gas=0):
    r, tau_metals, tau_Hminus, tau_H2, tau_dust, tau_tot = tau_maximun_contribution_Hminus(nu, Mpdot, Bps, dust_to_gas)
    tau_max_name=['H-', 'Metals', 'H2-']
    tau_max_array=np.array([tau_Hminus, tau_metals, tau_H2])
    tau_max_index=tau_max_array.tolist().index(max(tau_max_array))
    return r, tau_max_name[tau_max_index]

# # example with our frequencies
# B9=671E9

# Mpdot=6.95e-07*Mj/yr
# Bps=70
# dust_to_gas=0

# r, tau_max_source = main_opacity_source(B9, Mpdot, Bps, dust_to_gas)

Mpdot_arr=np.array([6.95e-07, 1e-6, 10**(-5.5), 10**(-5)])*Mj/yr
Bps_range_arr=np.array([[70,70], [84, 366], [150,600], [265,600]]) # Gauss
zeta=0
B9=671E9

# doing an array for all this Mpdot and Bps values for Band 9
cont_min_Bps_arr=[]
cont_max_Bps_arr=[]

for i in range(len(Mpdot_arr)):
    Mpdot=Mpdot_arr[i]
    Bps_min=Bps_range_arr[i,0]
    Bps_max=Bps_range_arr[i,1]
    r_min, tau_max_source_min = main_opacity_source(B9, Mpdot, Bps_min, zeta)
    r_max, tau_max_source_max = main_opacity_source(B9, Mpdot, Bps_max, zeta)
    cont_min_Bps_arr.append([Mpdot/Mj*yr, Bps_min, r_min/au, tau_max_source_min])
    cont_max_Bps_arr.append([Mpdot/Mj*yr, Bps_max, r_max/au, tau_max_source_max])

cont_min_Bps_arr=np.array(cont_min_Bps_arr)
cont_max_Bps_arr=np.array(cont_max_Bps_arr)

# this arrays contains str save it properly in a text file : raise TypeError("Mismatch between array dtype ('%s') and "
#TypeError: Mismatch between array dtype ('<U32') and format specifier ('%.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e')

np.savetxt(f'cont_magnetospheric_accretion_{int(Mp/Mj)}Mj_{int(a/au)}au.txt',
           np.column_stack((cont_min_Bps_arr, cont_max_Bps_arr)),
           fmt=['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s'],
           header='Mpdot_min(Mj/yr) Bps_min(G) r_min(au) tau_max_source_min Mpdot_max(Mj/yr) Bps_max(G) r_max(au) tau_max_source_max')