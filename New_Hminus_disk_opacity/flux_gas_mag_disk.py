# The flux of the dusty disk from Zhu 2018
from dustyCPD import *
from numpy import exp
from magneticCPD import H, T_z, alpha_calculator
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus import k_ff_Hminus, k_bf_Hminus
#from k_bf_H_2021 import k_bf_H
# from k_ff_H2_2021_minus import k_ff_H2_minus
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
def I_nu(R, nu, Mpdot, alpha):
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
    nH_arr = ntot_arr*a1 # we assume almost all H is neutral
    nion_arr=total_ion_abundance(ne_arr, Temp_z_arr)*ntot_arr
    nHe_arr=ntot_arr*a4 # we assume almost all He is neutral
    # the differential tau contribution of all the components (only gas in this case and no molecular Hydrogen)
    d_tau_arr_H=Pe_arr*nH_arr*(k_ff_Hminus(Temp_z_arr, nu)+k_bf_Hminus(Temp_z_arr, nu))*(dz)
    d_tau__arr_metals=Pe_arr*nion_arr*k_ff_metals(Temp_z_arr, nu)*(dz)
    d_tau_arr_He=Pe_arr*nHe_arr*k_ff_He_minus(Temp_z_arr, nu)*(dz)
    
    d_tau_arr=d_tau_arr_H+d_tau__arr_metals+d_tau_arr_He # total differential tau
    tau_arr=np.cumsum(d_tau_arr) # cumulative sum 
    # Now the total tau
    tau_tot=tau_arr[-1]
    # in the case we dont have massive tau we can just integrate directly
    if tau_tot<1000:
        integrate=np.trapezoid(plank_arr*exp(-tau_tot+tau_arr), tau_arr)*10**26 # to mJy
        return integrate
    # in the case tau is too big we return to the non stratified case
    else:
        l, p = find_1(tau_arr)
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)
        return plank_val*10**26


# In the magnetic disk case we have to integrate over the radius and alpha varies with the radius
def F_nu(nu, Mpdot, Bps):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    alpha_arr=alpha_calculator(R_arr, Mpdot, Bps)
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_nu(R_arr[i], nu, Mpdot, alpha_arr[i])*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy


# The best fit values for the magnetic disk 
Mpdot=10**(-5.670947617904571)*Mj/yr
Bps=653.5766047878215

# # First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

fluxes = np.zeros(len(nu))-5.5
for i in range(len(nu)):
    fluxes[i]=F_nu(nu[i], Mpdot, Bps)
    print('Flux at', round(nu[i]/1E9, 1), 'GHz =', fluxes[i], 'mJy')
    if i>0:
        print('Spectral index is ', (log10(fluxes[i])-log10(fluxes[i-1]))/(log10(nu[i])-log10(nu[i-1])))

#total spec intex
print('Total spectral index is ', (log10(fluxes[-1])-log10(fluxes[0]))/(log10(nu[-1])-log10(nu[0])))


# import matplotlib.pyplot as plt
# plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
# plt.plot(np.array(nus)/10**9, fluxes, label='model', alpha=0.5, color='red')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
# plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
# plt.legend(loc='upper left')
# plt.show()
