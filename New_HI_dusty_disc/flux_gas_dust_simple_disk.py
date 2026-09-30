# The flux of the dusty disk from Zhu 2018
from dustyCPD import *
from numpy import exp
from magneticCPD import H, T_z
from k_ff_metals import k_ff_metals
from k_bf_H_2021 import k_bf_H
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

#### To calculate the ionsation fraction of the CPD asuming the only important contribution comes from HI ####
def partition_function(T):
    Z=2
    for m in range(2,4):
        zj=2*m**2*exp(-13.6*(1-1/m**2)*eV/(kb*T))
        Z+=zj
    return Z

# The saha equation
def saha(T, n_e):
    return 2/partition_function(T)*((2*pi*me*kb*T/h**2)**(3/2) * exp(-13.6*eV/(kb*T))/n_e)

### Some useful functions to calculate the density and the ionisation fraction at each height in the disk ###

def rho(R, z, Mpdot, alpha):
    return 1/(sqrt(2*pi*H(R, Mpdot, alpha)**2))*Sigma(R,Mpdot,alpha)*exp(-z**2/(2*H(R, Mpdot, alpha)**2))

def n_H(rho):
    return (rho/(mu*mH))

def ionization_fraction_HI(T, R, z, Mpdot, alpha):
    n_h=n_H(rho(R,z,Mpdot,alpha))
    n_e=(sqrt(saha(T,1)**2+4*saha(T,1)*n_h)-saha(T,1))/2
    return n_e/n_h
# the ionisation fraction is then given by

# This function will return the intensity at a given radius and frequency, taking into account the vertical structure of the disk
# and the opacity form HI region plus the dust opacity.
def I_nu(R, nu, Mpdot, alpha, dust_to_gas=0.01):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    dz=z_arr[1]-z_arr[0] # step in z
    Temp_z_arr=T_z(z_arr, R, Mpdot, alpha)
    Sigma_R=Sigma(R, Mpdot, alpha) # surface density at R
    rho_z_arr=Sigma_R/(sqrt(2*pi)*h)*exp(-z_arr**2/(2*h**2))
    ntot_arr=rho_z_arr/(mu*mH)  # total number density at each height
    nH_arr = ntot_arr # we assume all the gas is atomic hydrogen for simplicity
    plank_arr=plank(Temp_z_arr, nu)
    f_arr=ionization_fraction_HI(Temp_z_arr, R, z_arr, Mpdot, alpha)
    ne_arr=f_arr*ntot_arr
    Pe_arr=ne_arr*kb*Temp_z_arr # electron pressure in cgs units
    # nH_arr, nH2_arr=H_H2_ratio(Temp_z_arr, ntot_arr*kb*Temp_z_arr).T*ntot_arr*a1
    nion_arr=ne_arr # We assume all electrons come from ionized hydrogen
    #nHe_arr=ntot_arr*a4 # we assume almost all He is neutral
    # the differential tau contribution of all the components (dust and gas)
    d_tau_arr_dust=kappa_mm(c/nu*1e1, dust_to_gas)*rho_z_arr*(dz)  # we convert nu to wavelength in mm
    #d_tau_arr_H2=Pe_arr*nH2_arr*k_ff_H2_minus(Temp_z_arr, nu)*(dz)
    #d_tau_arr_H=Pe_arr*nH_arr*(k_ff_Hminus(Temp_z_arr, nu)+k_bf_Hminus(Temp_z_arr, nu))*(dz)
    d_tau_metals=Pe_arr*nion_arr*k_ff_metals(Temp_z_arr, nu)*(dz)
    #d_tau_He=Pe_arr*nHe_arr*k_ff_He_minus(Temp_z_arr, nu)*(dz)
    d_tau_H_bf=nH_arr*10**(-17)*k_bf_H(Temp_z_arr, nu)*(dz) # this opacity was in units of 10^-17 cm^2 per neutral H atom
    tau_arr=np.cumsum(d_tau_arr_dust+d_tau_metals+d_tau_H_bf) # cumulative sum to get tau at each height

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
def F_nu(nu, Mpdot, alpha, zeta=0.01):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_nu(R_arr[i], nu, Mpdot, alpha, zeta)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy