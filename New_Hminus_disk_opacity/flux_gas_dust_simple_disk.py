# The flux of the dusty disk from Zhu 2018
from dustyCPD import *
from numpy import exp
from magneticCPD import H, T_z, alpha_calculator
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus import k_ff_Hminus, k_bf_Hminus
#from k_bf_H_2021 import k_bf_H
from k_ff_H2_2021_minus import k_ff_H2_minus
from H_H2_ratio import H_H2_ratio
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

def I_nu_dust(R, nu, Mpdot, alpha, dust_to_gas=0.01):
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
    # the differential tau contribution of all the components (dust and gas)
    d_tau_arr_dust=kappa_mm(c/nu*1e1, dust_to_gas)*rho_z_arr*(dz)  # we convert nu to wavelength in mm
    d_tau_arr_H2=Pe_arr*nH2_arr*k_ff_H2_minus(Temp_z_arr, nu)*(dz)
    d_tau_arr_H=Pe_arr*nH_arr*(k_ff_Hminus(Temp_z_arr, nu)+k_bf_Hminus(Temp_z_arr, nu))*(dz)
    d_tau_metals=Pe_arr*nion_arr*k_ff_metals(Temp_z_arr, nu)*(dz)
    d_tau_He=Pe_arr*nHe_arr*k_ff_He_minus(Temp_z_arr, nu)*(dz)
    tau_arr=np.cumsum(d_tau_arr_dust+d_tau_arr_H2+d_tau_arr_H+d_tau_metals+d_tau_He) # cumulative sum to get tau at each height

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


# In the simple disk, alpha is constant with R
def F_nu_dust(nu, Mpdot, alpha, zeta=0.01):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_nu_dust(R_arr[i], nu, Mpdot, alpha, zeta)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy