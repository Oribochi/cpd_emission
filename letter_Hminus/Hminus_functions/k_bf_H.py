# The opacity bound-free of the hydrogen atom
from units_astro import *
from numpy import exp, log, sqrt, pi, arctan
import numpy as np
import matplotlib.pyplot as plt

# some useful constants
alpha=(qe**2/(h*c)*2*pi)
A0=6.3*1e-18
# eletron=(2**9*pi**2/3*alpha*a0**2*10**18/6.3)**1/4
# print("Electron adimensionalization:", eletron)
# print("A0 constant:", A0)

# the photoionization cross section for an hydrogen like element
def photoionization_cross_section(nu,Z=1):
    nu_0=13.6*eV/h
    if nu < nu_0:
        return 0
    else:
        if abs(nu-nu_0)<10000:
            return A0
        else:
            nu_i=nu_0*Z**2
            x=nu_i/nu
            epsilon=sqrt(x**(-1)-1)
            return A0/Z**2*x**4*exp(4*(1-arctan(epsilon)/epsilon))/(1-exp(-2*pi/epsilon))
        
# De Burgess (1965) se tiene que la sección de fotoionización para un nivel n^2 L se puede calcular con
const1=4*pi*alpha*a0**2/3
print("Constante de Burgess:", const1)
# we have sum (2*L+1) l=0->n-1 that is just n^2 so the photoionization cross section is:
def photoionization_cross_section_n2L(nu,n,Z=1):
    # the electron ejected has an energy h(nu-nu_0)
    # nu_0 is the ionization energy of the hydrogen atom divided by n^2
    nu_0=13.6*eV/h/n**2
    e_energy=h*(nu-nu_0)
    if e_energy < 0:
        return 0  # no photoionization if energy is not enough
    else:
        # A0=64*pi*alpha*a0**2/(sqrt(3)*3)?
        val1=A0*n/Z**2
        k_square=e_energy/(13.6*eV)/Z**2
        return val1/(1+n**2*k_square)**3

# example for nu=nu_0    
print("Photoionization cross section for nu=nu_0:", photoionization_cross_section(13.6*eV/h, 1))
print("Photoionization cross section for nu=nu_0:", photoionization_cross_section_n2L(13.6*eV/h, 1))
# now the total opacity for bound-free transitions with n=1,2,3,4,5
def total_cross_section(nu,Z=1):
    return sum(photoionization_cross_section_n2L(nu,n,Z)*n for n in range(1,6))

# example plot nu=(0-20)eV/h
nu_array = np.linspace(0, 20*eV/h, 200)
total_cross_section_array = [total_cross_section(nu) for nu in nu_array]
cross_section_array = [photoionization_cross_section(nu) for nu in nu_array]

plt.figure(figsize=(10, 6))
plt.plot(nu_array*eV/h, total_cross_section_array, label='Total Cross Section', color='blue')
plt.plot(nu_array*eV/h, cross_section_array, label='Photoionization Cross Section', color='red')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Cross Section (cm^2)')
plt.title('Bound-Free Photoionization Cross Section')
plt.legend()
plt.grid()
plt.show()