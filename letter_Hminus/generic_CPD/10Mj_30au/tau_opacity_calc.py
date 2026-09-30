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

from dustyCPD import *
from numpy import exp
from magneticCPD import H, T_z, alpha_calculator
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from H_H2_ratio import H_H2_ratio
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus import k_ff_Hminus, k_bf_Hminus
#from k_bf_H_2021 import k_bf_H
from k_ff_H2_john_1994_linear_interp import k_ff_H2_minus_john_interp as k_ff_H2_minus
# from H_H2_ratio import H_H2_ratio
from k_ff_He_minus import k_ff_He_minus
from k_dust import kappa_mm

# Now similar to flux_simple_disk but using magnetic field disk
####################################################################################3

# this will receive an array of Temperatures and return the plank function for each temperature
def plank(T, nu):    
    x=h*nu/(kb*T)
    mask1 = x < 1e-3  # low frequency limit
    mask2 = x > 1e1   # high frequency limit
    result = np.zeros_like(x) #
    result[mask1] = 2*nu**2/c**2*kb*T[mask1]
    result[mask2] = 2*h*nu**3/c**2*exp(-x[mask2])
    mask3 = ~(mask1 | mask2)  # intermediate values
    result[mask3] = 2*h*nu**3/c**2/(exp(x[mask3])-1)
    return result

###### Case with stratification ######
def find_1(arr):
    k=len(arr)-1
    j=0
    while j<k-1:
        p=(j+k)//2
        if arr[p]>1:
            k=p
        else:
            j=p
    # print(tau_menos, tau_mas)
    return j, k

# The specific intensity at a given radius R, frequency nu, mass accretion rate Mpdot, alpha and dust_to_gas ratio
def I_nu_tau(R, nu, Mpdot, alpha, dust_to_gas):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    dz=z_arr[1]-z_arr[0] # step in z
    Temp_z_arr=T_z(z_arr, R, Mpdot, alpha)
    Sigma_R=Sigma(R, Mpdot, alpha) # surface density at R
    rho_z_arr=Sigma_R/(sqrt(2*pi)*h)*exp(-z_arr**2/(2*h**2))
    ntot_arr=rho_z_arr/(mu*mH)  # total number density at each height
    plank_arr=plank(Temp_z_arr, nu)
    f_arr=simple_ionization_fraction_H_minus(Temp_z_arr, rho_z_arr)
    ne_arr=f_arr*ntot_arr
    Pe_arr=ne_arr*kb*Temp_z_arr # electron pressure in cgs units
    nH_arr, nH2_arr=H_H2_ratio(Temp_z_arr, ntot_arr*kb*Temp_z_arr).T*ntot_arr*a1
    nion_arr=total_ion_abundance(ne_arr, Temp_z_arr)*ntot_arr
    nHe_arr=ntot_arr*a4 # we assume almost all He is neutral
    # the differential tau contribution of all the components (only gas in this case and no molecular Hydrogen)
    d_tau_arr_dust=kappa_mm(c/nu*1e1, dust_to_gas)*rho_z_arr*(dz)  # we convert nu to wavelength in mm
    d_tau_arr_H2=Pe_arr*nH2_arr*k_ff_H2_minus(Temp_z_arr, nu)*(dz)
    d_tau_arr_H=Pe_arr*nH_arr*(k_ff_Hminus(Temp_z_arr, nu)+k_bf_Hminus(Temp_z_arr, nu))*(dz)
    d_tau__arr_metals=Pe_arr*nion_arr*k_ff_metals(Temp_z_arr, nu)*(dz)
    d_tau_arr_He=Pe_arr*nHe_arr*k_ff_He_minus(Temp_z_arr, nu)*(dz)

    d_tau_arr=d_tau_arr_H+d_tau__arr_metals+d_tau_arr_He+d_tau_arr_dust+d_tau_arr_H2 # total differential tau
    tau_arr=np.cumsum(d_tau_arr) # cumulative sum 
    # Now the total tau
    tau_tot=tau_arr[-1]
    # in the case we dont have massive tau we can just integrate directly
    tau_H=np.trapz(d_tau_arr_H, z_arr/dz) # the total tau from H- (both free-free and bound-free)
    tau_metals=np.trapz(d_tau__arr_metals, z_arr/dz) # the total tau from metals
    tau_H2=np.trapz(d_tau_arr_H2, z_arr/dz) # the total tau from H2
    tau_dust=np.trapz(d_tau_arr_dust, z_arr/dz) # the total tau from dust
    # we save the tau from H, the tau from metals and the total tau
    
    if tau_tot<1000:
        integrate=np.trapezoid(plank_arr*exp(-tau_tot+tau_arr), tau_arr)*10**26 # to mJy
        return integrate, tau_metals, tau_H, tau_H2, tau_dust, tau_tot
    # in the case tau is too big we return to the non stratified case
    else:
        l, p = find_1(tau_arr)
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)
        return plank_val*10**26, 1000, 1000, 1000, 1000, 1000

    
# R_array=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au
# I_ff_arr=np.zeros(len(R_array))
# for i in range(len(R_array)):
#     alpha=alpha_calculator(R_array[i], Mpdot, Bps)
#     I_ff_arr[i]=I_ff_metals(R_array[i], 100E9, Mpdot, alpha)
# plt.plot(R_array/au, I_ff_arr)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R (au)')
# plt.ylabel('I_ff (W/m²/Hz/sr)')
# plt.ylim(1e-10, 1e17)
# plt.show()
    
# Mpdot=10**(-5.49)*Mj/yr
# Bps=600

def tau_maximun_contribution_Hminus(nu, Mpdot, Bps, dust_to_gas):
    R_arr=np.linspace(Rin*1.001, Rout, 500)
    I_nu0_r=0
    tau_metals_max=0
    tau_Hminus_max=0
    tau_tot_max=0
    for r in R_arr:
        alpha=alpha_calculator(r, Mpdot, Bps)
        I_nu, tau_metals, tau_Hminus, tau_H2, tau_dust, tau_total  = I_nu_tau(r, nu, Mpdot, alpha, dust_to_gas=dust_to_gas)
        if I_nu*r>I_nu0_r:
            I_nu0_r=I_nu*r
            tau_metals_max=tau_metals
            tau_Hminus_max=tau_Hminus
            tau_tot_max=tau_total
        else: # we stop when the intensity starts to decrease
            break
    
    print(r/au, 'r in au')
    # r=0.028*au
    alpha=alpha_calculator(r, Mpdot, Bps)
    I_nu, tau_metals, tau_Hminus, tau_H2, tau_dust, tau_total = I_nu_tau(r, nu, Mpdot, alpha, dust_to_gas=dust_to_gas)
    tau_metals_max=tau_metals
    tau_Hminus_max=tau_Hminus
    tau_H2_max=tau_H2
    tau_dust_max=tau_dust
    tau_tot_max=tau_total
    return r, tau_metals_max, tau_Hminus_max, tau_H2_max, tau_dust_max, tau_tot_max


# Mpdot=6.95e-07*Mj/yr
# Bps=70
# dust_to_gas=0

# # example with our frequencies
# nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

# array_nus = np.logspace(log10(nus[0]), log10(nus[-1]), 100)
# tau_metals_array = np.zeros(len(array_nus))
# tau_Hminus_array = np.zeros(len(array_nus))
# tau_H2_array = np.zeros(len(array_nus))
# tau_dust_array = np.zeros(len(array_nus))
# tau_tot_array = np.zeros(len(array_nus))
# r=0
# for i in range(len(array_nus)):
#     r, tau_metals_array[i], tau_Hminus_array[i], tau_H2_array[i], tau_dust_array[i], tau_tot_array[i] = tau_maximun_contribution_Hminus(array_nus[i], Mpdot, Bps, dust_to_gas)
#     #print(r/au, tau_metals_array[i]+tau_Hminus_array[i])


# plt.figure(figsize=(4,3), dpi=300)
# plt.plot(array_nus/1E9, tau_Hminus_array, label=r'$\rm H^{-}$', color='C2')
# plt.plot(array_nus/1E9, tau_metals_array, label='Metals', color='C0')
# plt.plot(array_nus/1E9, tau_H2_array, label=r'$\rm H_2^{-}$', color='C1')
# #plt.plot(array_nus/1E9, tau_dust_array, label='Dust', color='C2')
# plt.plot(array_nus/1E9, tau_tot_array, label='Total', color='C3', linestyle='--')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
# plt.ylabel(r'$\tau$', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(loc='best', fontsize=12)
# plt.savefig('tau_H2minus_molecular.pdf', bbox_inches='tight', dpi=300)
# plt.show()

# # save the main source of opacity and the radius at which the maximum contribution to the flux comes from
# tau_max_name=['H-', 'Metals', 'H2-']

# # the max within H-, Metals and H2- is the main source of opacity
# tau_max_array=np.array([tau_Hminus_array[-1], tau_metals_array[-1], tau_H2_array[-1]])

# tau_max_index=tau_max_array.tolist().index(max(tau_max_array))
# # obtain the maximun index
# print('The main source of opacity is:', tau_max_name[tau_max_index], 'at r =', r/au, 'au')