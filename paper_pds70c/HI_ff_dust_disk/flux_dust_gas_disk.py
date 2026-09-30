# This is the code with all the functions to calculate the flux and brightness temperature of the magnetic disk model
from units_astro import *
import numpy as np
from numpy import pi, sqrt, log10, exp
from dustyCPD import Sigma, Rin, Rout, dis, mu, Tfloor
from k_dust import kappa_mm
from magneticCPD import H, alpha_calculator, T_z
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus import k_ff_Hminus, k_bf_Hminus
from k_bf_H_2021 import k_bf_H
from k_ff_H2_2021_minus import k_ff_H2_minus
from H_H2_ratio import H_H2_ratio
from k_ff_He_minus import k_ff_He_minus
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-x)
    else:
        return 2*h*nu**3/c**2/(exp(x)-1)

###### Case with stratification ###### To find tau=1 height ######

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

# the total emergent intensity of an annulus at radius R including the opacity from dust, free-free from metals
# and H- (bf+ff), H (bf), H2- (ff) and He- (ff)
def I_nu_tot(R, nu, Mpdot, alpha, dust_to_gas=0.01):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    ne_arr=np.zeros(len(z_arr))
    n_ion_arr=np.zeros(len(z_arr))
    n_H_arr=np.zeros(len(z_arr))
    n_H2_arr=np.zeros(len(z_arr))
    n_He_arr=np.zeros(len(z_arr))
    k_ff_metals_arr=np.zeros(len(z_arr))
    k_ff_bf_Hminus_arr=np.zeros(len(z_arr))
    k_bf_H_arr=np.zeros(len(z_arr))
    k_ff_H2_minus_arr=np.zeros(len(z_arr))
    k_ff_He_minus_arr=np.zeros(len(z_arr))
    tau_arr=np.zeros(len(z_arr))
    Temp_z_arr=np.zeros(len(z_arr))
    rho_z_arr=np.zeros(len(z_arr))
    plank_arr=np.zeros(len(z_arr))
    k=0
    # we go fromals_max=0
    # tau_Hminus_max=0
    # for r in R_arr:
    #     alpha=alpha_calculator(r, Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    #     if 3H to -3H
    for i in range(len(z_arr)):
        m=i
        Temp_z=T_z(z_arr[m], R, Mpdot, alpha)
        Temp_z_arr[m]=Temp_z
        k_ff_metals_arr[m]=k_ff_metals(Temp_z, nu)
        # this function receives the wavelength in Angstroms and the temperature in Kelvin
        k_ff_bf_Hminus_arr[m]=k_ff_Hminus(Temp_z, nu)+k_bf_Hminus(Temp_z, nu)
        # print('k_ff', k_ff_arr[m]) the kappa opacity is of the order of 10**(-27)
        k_bf_H_arr[m]=k_bf_H(Temp_z, nu) # this opacity is in cm^2 per neutral H atom
        # The opacity of H2 minus
        k_ff_H2_minus_arr[m]=k_ff_H2_minus(Temp_z, nu)
        # The opacity of He minus per unit cm² per He atom per electron pressure
        k_ff_He_minus_arr[m]=k_ff_He_minus(Temp_z, nu)
        # now the density at this height
        rho=Sigma(R, Mpdot, alpha)/(sqrt(2*pi)*h)*exp(-z_arr[m]**2/(2*h**2))
        rho_z_arr[m]=rho # mass density for dust opacity
        f=simple_ionization_fraction_H_minus(Temp_z, rho)
        # f=10**(-4)
        ntot=rho/(mu*mH) # the number density of atoms
        # obtaining the number densities of H2 and H
        n_H_arr[m], n_H2_arr[m]=np.array(H_H2_ratio(Temp_z, ntot*kb*Temp_z))*ntot*a1 # number density of H vs the H2 molecules by minimizing the Gibbs energy 
        # print('n_H', n_H_arr[m], 'n_H2', n_H2_arr[m])       
        ne_arr[m]=f*ntot # the number density of electrons
        n_ion_arr[m]=total_ion_abundance(ne_arr[m], Temp_z)*ntot # the number density of ions
        n_He_arr[m]=ntot*a4 # we assume almost all He is neutral
        # tau is the integral of ne**2*k_ff from z to 3H
        tau_metals=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_ion_arr[:m]*k_ff_metals_arr[:m], z_arr[:m])
        tau_Hminus=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_H_arr[:m]*k_ff_bf_Hminus_arr[:m], z_arr[:m])  # We asumme H- contributes to the opacity
        tau_H2_minus=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_H2_arr[:m]*k_ff_H2_minus_arr[:m], z_arr[:m])  # We asumme H2- contributes to the opacity
        tau_He_minus=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_He_arr[:m]*k_ff_He_minus_arr[:m], z_arr[:m])  # We asumme He- contributes to the opacity
        tau_dust=kappa_mm(c/nu*1e1, dust_to_gas)*np.trapezoid(rho_z_arr[:m], z_arr[:m])
        tau_arr[m]=tau_Hminus+tau_metals+tau_H2_minus+tau_He_minus+tau_dust
        plank_arr[m]=plank(Temp_z, nu)

        # in the case tau is too big we return to the non stratified case
        if tau_arr[m]>1000:
            k=m
            break
        
    tau_tot=tau_arr[-1]

    if k==0:
        integrate=np.array([plank_arr[i]*exp(-tau_tot+tau_arr[i]) for i in range(N)])*10**26
        return np.trapezoid(integrate, tau_arr)

    # for the case optically thinner
    #print(tau_arr[0], tau_arr[int(N/2)], tau_arr[-1])
    #print(ne_arr[0]**2*h, ne_arr[int(N/2)]**2*h, ne_arr[-1]**2*h)
    #print(k_ff_arr[0], k_ff_arr[int(N/2)], k_ff_arr[-1])
    # for the case optically thicker tau is too big and only the last value is important
    else:
        l, p = find_1(tau_arr[:k])
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)

        return plank_val*10**26
    
def F_nu_tot(nu, Mpdot, Bps, zeta=0.01):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_tot_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
        F_annulus=I_nu_tot(R_arr[i], nu, Mpdot, alpha, dust_to_gas=zeta)*2*pi*R_arr[i]/dis**2
        F_tot_arr[i]=F_annulus
    return np.trapezoid(F_tot_arr, R_arr)  # to mJy