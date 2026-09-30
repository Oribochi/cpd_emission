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

def EM(R,Mpdot,alpha):
    H_z=H(R,Mpdot,alpha)
    z_arr=np.linspace(0,3*H_z,50)
    n_e_arr=np.zeros(len(z_arr))
    n_ion_arr=np.zeros(len(z_arr))
    dz=z_arr[1]-z_arr[0] # step in z
    Tz=T_z(z_arr, R, Mpdot, alpha)
    Sigma_R=Sigma(R, Mpdot, alpha) # surface density at R
    rho_z_arr=Sigma_R/(sqrt(2*pi)*H_z)*exp(-z_arr**2/(2*H_z**2))
    f=simple_ionization_fraction_H_minus(Tz, rho_z_arr)
    ne_arr=f*rho_z_arr/(mu*mH)
    nion_arr=total_ion_abundance(ne_arr, Tz)*rho_z_arr/(mu*mH)
    em=np.trapezoid(ne_arr*nion_arr, z_arr)*2
    return em


def profiles(Mpdot,Bps):
    R_array=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    alpha_array=alpha_calculator(R_array, Mpdot, Bps)
    Sigma_array=Sigma(R_array, Mpdot, alpha_array)
    T_c_array=T_c_approx(R_array, Mpdot, alpha_array)
    f_array=simple_ionization_fraction_H_minus(T_c_array, Sigma_array/(sqrt(2*pi)*H(R_array, Mpdot, alpha_array)))
    EM_array=np.zeros(len(R_array))
    z_array_mid_f=np.zeros(len(R_array))
    for i in range(len(R_array)):
        EM_array[i]=EM(R_array[i], Mpdot, alpha_array[i])
        # height where ionization fraction is half of the midplane value
        f_mid=f_array[i]
        h=H(R_array[i], Mpdot, alpha_array[i])
        z_arr=np.linspace(0,3*h,100)
        Tz=T_z(z_arr, R_array[i], Mpdot, alpha_array[i])
        Sigma_R=Sigma_array[i] # surface density at R
        rho_z_arr=Sigma_R/(sqrt(2*pi)*h)*exp(-z_arr**2/(2*h**2))
        fz=simple_ionization_fraction_H_minus(Tz, rho_z_arr)
        # find the height where fz is half of f_mid
        index=np.where(fz<=f_mid/2)[0]
        if len(index)>0:
            z_array_mid_f[i]=z_arr[index[0]]
        else:
            z_array_mid_f[i]=3*h
        # replace all z_array_mid_f values that are 0 for 10^-10 au to avoid problems with log10
        z_array_mid_f=np.where(z_array_mid_f==0, 10**-10*au, z_array_mid_f)
    return R_array, alpha_array, Sigma_array, T_c_array, EM_array, f_array, z_array_mid_f
