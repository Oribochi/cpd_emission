#profiles of the MHD diks
import numpy as np
import matplotlib.pyplot as plt
from magneticCPD import R_T1000, R_truncation, profiles, alpha_calculator, Sigma
from units_astro import *
from numpy import log10, pi, sqrt
from flux_magnetic_disk import F_nu_tot
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

def likelihood(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']*Mj/yr
    Bps = 10**param_dict['log_Bps']
    
    R1k = R_T1000(Mpdot, Bps)/au
    Rtrunc = R_truncation(Mpdot, Bps)/au

    diff = abs(Rtrunc-R1k)/R1k
    R_diff = (Rtrunc-R1k)
    # we want to make sure that the truncation radius is bigger than the 1000 radius so the ionized gas can couple the magnetic field
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F_nu_tot(nu[i], Mpdot, Bps)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2
    # The more negative R_diff I want a bigger penalty, else just the sum of the likelihoods
    if R_diff < 0:
        return np.sum(Xis)-10*diff
    else:
        return np.sum(Xis)


Mpdot=10**(-5.596172)*Mj/yr
Bps=10**(2.81155774) # G
print('The Bps is:', Bps, 'G')

print('The Maximun likelihood is', likelihood({'log_Mpdot': np.log10(Mpdot/Mj*yr), 'log_Bps': np.log10(Bps)}))
R_array, alpha_array, Sigma_array, T_c_array, EM_array, f_array, rho_array, z_array_mid_f = profiles(Mpdot,Bps)
# replace the 0 values from z_array_mid_f with the first non-zero value
print(z_array_mid_f)
mu=2.4
ne_array=np.array([rho_array[i]/(mu*mH)*f_array[i] for i in range(len(rho_array))])

Rt=R_truncation(Mpdot,Bps)
print('The truncation radius is:',Rt/Rj, 'Rj')

Ms = 1.0*Msun
# The planet mass in Mjup
Mp = 4*Mj
# The planet radius en Rjup
Rp = Rj
# The planet distance to the star in AU
a = 34*au

Rin = 1*Rp
Rout = 1/3*(Mp/(3*Ms))**(1/3)*a # This is 1/3 of the Hill radius

mu=1

# Find the first R that has 1000K temperature or more
def R_T1000(Mpdot,Bps):
    R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 500)*au
    R_T1000=0
    for i in range(len(R_arr)):
        Tc=T_c_array[i]
        if Tc>1000:
            return R_arr[i]
            break
        else:
            R_T1000=R_arr[i]
    return R_T1000

R_T1000=R_T1000(Mpdot,Bps)
print('The first R with T>1000K is:',R_T1000/Rj, 'Rj')

# Disk mass
def Mdisk(Mpdot, Bps):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)*au
    Sigma_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
        Sigma_arr[i]=Sigma(R_arr[i], Mpdot, alpha)
    return np.trapezoid(Sigma_arr*R_arr, R_arr)*2*pi # to g

print('The disk mass is:', Mdisk(Mpdot, Bps)/Mj, 'Mj')


# the profiles
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# the profiles
fig, ax1 = plt.subplots(3, 1, figsize=(4.8, 7.8), sharex=True, dpi=300)
fig.subplots_adjust(hspace=0.0, wspace=0.2)


ax1[0].plot(R_array/au, log10(T_c_array), label=r'$\log\left(\frac{T_{c\rm }}{\rm K}\right)$', color='C0')
ax1[0].axvline(x=Rt/au, color='C1', linestyle='--', label=r'$R_{\rm t}$')
ax1[0].set_ylim(1.8,3.8)
ax1[0].set_xscale('log')
ax1[0].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)



ax1[1].plot(R_array/au, log10(alpha_array), label=r'$\log(\alpha)$', color='C0')
ax1[1].plot(R_array/au, log10(Sigma_array/1000), label=r'$\log\left(\frac{10^{-3} \,\Sigma}{\rm g\,cm^{-2}}\right)$', color='C2')
ax1[1].axvline(x=Rt/au, color='C1', linestyle='--')
# make the ticks in -4, -2, 0 and 2
ax1[1].set_ylim(-6,2.5)
ax1[1].set_xscale('log')
#ax1[1].set_yticks([-5.0,-3.0,-1.0,1.0])
ax1[1].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)

# ax1[1].plot(R_array/au, log10(Sigma_array), label=r'\Sigma')
# ax1[1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[1].set_xscale('log')
# ax1[1].set_yscale('log')
# ax1[1].set_ylabel(r'Σ (g/cm²)', fontsize=8)
# ax1[1].tick_params(axis='both', which='minor', labelsize=6)
# ax1[1].tick_params(axis='both', which='major', labelsize=6)
# ax1[1].tick_params(direction='in', which='both')

# ax1[2].plot(R_array/au, T_c_array, label=r'T_c \rm (K)')
# ax1[2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[2].set_xscale('log')
# ax1[2].set_yscale('log')
# ax1[2].set_ylabel(r'$T_c$ (K)', fontsize=8)
# ax1[2].tick_params(axis='both', which='minor', labelsize=6)
# ax1[2].tick_params(axis='both', which='major', labelsize=6)
# ax1[2].tick_params(direction='in', which='both')

# replace 0 values in EM_array, f_array and z_array_mid_f with the minimum non-zero value
EM_array = np.copy(EM_array)
EM_array[EM_array == 0] = np.min(EM_array[EM_array > 0])*10**(-6)

f_array = np.copy(f_array)
f_array[f_array == 0] = np.min(f_array[f_array > 0])*10**(-6)

z_array_mid_f = np.copy(z_array_mid_f)
z_array_mid_f[z_array_mid_f == 0] = np.min(z_array_mid_f[z_array_mid_f > 0])*10**(-6)

ax1[2].plot(R_array/au, log10(EM_array/(10**8*pc)), label=r'$\log\left(\frac{10^{-8}\,{\rm EM}}{\rm cm^{-6} pc}\right)$', color='C0')
ax1[2].plot(R_array/au, log10(f_array/10**(-6)), label=r'$\log\left(\frac{f}{10^{-6}}\right)$', color='C2')
ax1[2].plot(R_array/au, log10(z_array_mid_f/(0.01*au)), label=r'$\log\left(\frac{ H_{\rm hwhm}}{10^{-2}\,\rm au}\right)$', color='C3')
ax1[2].set_ylim(-3.5,0.5)
ax1[2].set_xscale('log')
ax1[2].axvline(x=Rt/au, color='C1', linestyle='--')
#ax1[2].set_yticks([-3.0,-2.0,-1.0,0.0])
ax1[2].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)
ax1[2].set_xlabel(r'${R}({\rm au})$')


# ax1[4].plot(R_array/au, f_array, label='f')
# ax1[4].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[4].set_xscale('log')
# ax1[4].set_yscale('log')
# ax1[4].set_ylim(1E-10*max(f_array), max(f_array))
# ax1[4].set_ylabel('f', fontsize=8)
# ax1[4].tick_params(axis='both', which='minor', labelsize=6)
# ax1[4].tick_params(axis='both', which='major', labelsize=6)
# ax1[4].tick_params(direction='in', which='both')

# ax1[5].plot(R_array/au, z_array_mid_f, label=r'ρ (g/cm³)')
# ax1[5].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[5].set_xscale('log')
# ax1[5].set_yscale('log')
# ax1[5].set_ylim(z_array_mid_f[0], max(z_array_mid_f))
# ax1[5].set_ylabel(r'$H_{\rm hwhm}$ (au)', fontsize=8)
# ax1[5].tick_params(axis='both', which='minor', labelsize=6)
# ax1[5].tick_params(axis='both', which='major', labelsize=6)
# ax1[5].tick_params(direction='in', which='both')
plt.savefig('profile_metals_Hminus.png', bbox_inches='tight', dpi=300)
plt.show()
print('The maximun ionization fraction is', max(f_array))

#Search the maximun surface density and the corresponding radius
max_Sigma=max(Sigma_array)
index_max_Sigma=np.where(Sigma_array==max_Sigma)[0][0]
R_max_Sigma=R_array[index_max_Sigma]
# now the volume density at that radius
rho_max_Sigma=rho_array[index_max_Sigma]
# and the number density
n_max_Sigma=rho_max_Sigma/(mu*mH)

# now the max rho is
max_rho=max(rho_array)
index_max_rho=np.where(rho_array==max_rho)[0][0]

print('The maximum surface density is:', max_Sigma, 'g/cm² at a radius of:', R_max_Sigma/Rj, 'Rj with a volume density of:', rho_max_Sigma, 'g/cm³ and a number density of:', n_max_Sigma, 'cm⁻³')
print('The maximum volume density is:', max_rho, 'g/cm³ at a radius of:', R_array[index_max_rho]/Rj, 'Rj')