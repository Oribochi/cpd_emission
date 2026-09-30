# This is the code to plot profiles of discs

# Replicating the Zhu 2018 model of the CPD
from dustyCPD import *
from scipy.special import erf
from matplotlib import pyplot as plt

# example
Mpdot=1e-5*Mj/yr
alpha=0.01


def H(R,Mpdot,alpha):
    T=T_c_approx(R,Mpdot,alpha)
    c_s=sqrt(Rgas*T/(mu))
    Omega=sqrt(G*Mp/R**3)
    H=c_s/Omega
    return H

# Asuming a vertical hydrostatic equilibrium we can calculate the temperature
def T_z(z, R, Mpdot, alpha):
    h=H(R,Mpdot,alpha)
    tau_r=kappa_r*Sigma(R,Mpdot,alpha)*(1-erf(abs(z)/(sqrt(2)*h)))/2
    # we can use that rho is a gaussian
    ########## Cambiar
    Teff_z4=3/8*T_eff(R,Mpdot)**4*tau_r
    Text=T_ext(R,Mpdot)
    return (Teff_z4+Text**4)**(1/4)
    #  return T_c_approx(R,Mpdot,alpha)

# now some profiles for example temperature and density 
def profiles(Mpdot,alpha):
    R_array=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    Sigma_array=np.zeros(len(R_array))
    T_c_array=np.zeros(len(R_array))
    for i in range(len(R_array)):
        Sigma_array[i]=Sigma(R_array[i],Mpdot,alpha)
        T_c_array[i]=T_c_approx(R_array[i],Mpdot,alpha)
    return R_array, Sigma_array, T_c_array

plt.figure()
R_array, Sigma_array, T_c_array=profiles(Mpdot,alpha)
plt.subplot(2,1,1)
plt.plot(R_array/au, Sigma_array)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R (au)')
plt.ylabel('Sigma (g/cm^2)')
plt.subplot(2,1,2)  
plt.plot(R_array/au, T_c_array)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R (au)')
plt.ylabel('T_c (K)')   
plt.tight_layout()
plt.show()

# Stratified temperature at 0.01 au
R_interest=0.01*au
h=H(R_interest,Mpdot,alpha)
z_array=np.linspace(-3*h, 3*h, 100)
T_z_array=np.zeros(len(z_array))
for i in range(len(z_array)):
    T_z_array[i]=T_z(z_array[i], R_interest, Mpdot, alpha)  

plt.figure()
plt.plot(z_array/au, T_z_array)
plt.xlabel('z (au)')
plt.ylabel('T(z) (K)')
plt.title('Stratified temperature at R=0.01 au')
plt.show()  