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

# We want to make a plot with the flux per ring indicating the main source of opacity (H- or H2-)
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

# Returns the specific intensity at a given radius R, frequency nu, mass accretion rate Mpdot, alpha and dust_to_gas ratio and the
# tau from metals, H-, H2- and dust
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
    #d_tau_arr_He=Pe_arr*nHe_arr*k_ff_He_minus(Temp_z_arr, nu)*(dz)

    d_tau_arr=d_tau_arr_H+d_tau__arr_metals+d_tau_arr_dust+d_tau_arr_H2 # total differential tau
    tau_arr=np.cumsum(d_tau_arr) # cumulative sum 
    # Now the total tau
    tau_tot=tau_arr[-1]
    # in the case we dont have massive tau we can just integrate directly
    tau_H=np.trapz(d_tau_arr_H, z_arr/dz) # the total tau from H- (both free-free and bound-free)
    #tau_He=np.trapz(d_tau_arr_He, z_arr/dz) # the total tau from He-
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
        # and the tau at 1
        tau_metals=np.trapz(d_tau__arr_metals, z_arr[:p]/dz) # the total tau from metals
        tau_H=np.trapz(d_tau_arr_H, z_arr[:p]/dz) # the total tau from H- (both free-free and bound-free)
        tau_H2=np.trapz(d_tau_arr_H2, z_arr[:p]/dz) # the total tau from H2
        tau_dust=np.trapz(d_tau_arr_dust, z_arr[:p]/dz) # the total tau from dust
        tau_tot=tau_metals+tau_H+tau_H2+tau_dust
        return plank_val*10**26, tau_metals, tau_H, tau_H2, tau_dust, tau_tot

    
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
    # R_arr=np.linspace(Rin*1.001, Rout, 100)
    # I_nu0_r=0
    # tau_metals_max=0
    # tau_Hminus_max=0
    # tau_tot_max=0
    # for r in R_arr:
    #     alpha=alpha_calculator(r, Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus, tau_H2, tau_dust, tau_total  = I_nu_tau(r, nu, Mpdot, alpha, dust_to_gas=dust_to_gas)
    #     if I_nu*r>I_nu0_r:
    #         I_nu0_r=I_nu*r
    #         tau_metals_max=tau_metals
    #         tau_Hminus_max=tau_Hminus
    #         tau_tot_max=tau_total
    #     else: # we stop when the intensity starts to decrease
    #         break
    
    # print(r/au, 'r in au')
    r=0.0625*au
    alpha=alpha_calculator(r, Mpdot, Bps)
    I_nu, tau_metals, tau_Hminus, tau_H2, tau_dust, tau_total = I_nu_tau(r, nu, Mpdot, alpha, dust_to_gas=dust_to_gas)
    tau_metals_max=tau_metals
    tau_Hminus_max=tau_Hminus
    tau_H2_max=tau_H2
    tau_dust_max=tau_dust
    tau_tot_max=tau_total
    return r, tau_metals_max, tau_Hminus_max, tau_H2_max, tau_dust_max, tau_tot_max

# best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
# Mpdot=10**(float(best_fit[0][1]))*Mj/yr
# Bps=10**(float(best_fit[3][1]))
# dust_to_gas=10**(float(best_fit[8][1]))

# # example with our frequencies
# nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

# R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 500)*au

# I_nu_arr=np.zeros(len(R_arr))
# # save an binary array of 4 spaces where the 1 is located in the respective tau that is the largest (tau_Hminus, tau_H2, tau_metals, tau_dust)
# binary_array=np.zeros((len(R_arr), 4))

# for i in range(len(R_arr)):
#     alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
#     I_nu_arr[i], tau_metals, tau_Hminus_arr, tau_H2_arr, tau_dust, tau_total = I_nu_tau(R_arr[i], 671E9, Mpdot, alpha, dust_to_gas=dust_to_gas)
#     j=np.argmax([tau_Hminus_arr, tau_H2_arr, tau_metals, tau_dust])
#     # store an array of 4 spaces with one 1 in the index i
#     binary_array[i] = np.zeros(4)
#     binary_array[i][j] = 1

# # now plot the flux per ring indicating the main source of opacity (metals, H-, H2- or dust) in different colors
# plt.figure(figsize=(4,3))
# plt.plot(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, label='Flux per ring')
# plt.fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,0]==1, color='C0', alpha=0.5, label='H$^-$ dominates')
# plt.fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,1]==1, color='C1', alpha=0.5, label=r'H$_2^-$ dominates')
# #plt.fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,2]==1, color='C2', alpha=0.5, label='Metals dominates')
# plt.fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,3]==1, color='C2', alpha=0.5, label='Dust dominates')
# plt.xlabel(r'$R$ [au]', fontsize=12)
# plt.ylabel(r'$I_{\nu}\times \frac{2 \pi R}{d^2} \times 1 \rm au ~ [\rm mJy]$', fontsize=12)
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1e-6, 1e1)
# # color where H- is larger than H2-
# plt.legend()
# plt.savefig('flux_per_ring_with_opacity_sources.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# # and print how much flux comes from H-, H2- and dust
# flux_Hminus=np.sum(I_nu_arr*2*pi*R_arr*au/dis**2 * binary_array[:,0])
# flux_H2=np.sum(I_nu_arr*2*pi*R_arr*au/dis**2 * binary_array[:,1])
# flux_dust=np.sum(I_nu_arr*2*pi*R_arr*au/dis**2 * binary_array[:,3])
# print(f"Flux from H-: {flux_Hminus} mJy")
# print(f"Flux from H2-: {flux_H2} mJy")
# print(f"Flux from dust: {flux_dust} mJy")

# # the total flux is the sum of the three
# print(f"Total flux: {flux_Hminus+flux_H2+flux_dust} mJy")

# # in percentage
# print(f"Percentage of flux from H-: {flux_Hminus/(flux_Hminus+flux_H2+flux_dust)*100:.2f}%")
# print(f"Percentage of flux from H2-: {flux_H2/(flux_Hminus+flux_H2+flux_dust)*100:.2f}%")
# print(f"Percentage of flux from dust: {flux_dust/(flux_Hminus+flux_H2+flux_dust)*100:.2f}%")