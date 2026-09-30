# The opacity free-free of the H2- atom using Someville 1964
from units_astro import c

# In the atmospheres of cool stars, most of the hydrogen is in molecular form and continuous absorption 
# by H2- is of importance in determining opacities, similarly to the absorption by H- in hotter stars. (See Someville 1964)
# They use the wave number diference
def wave_number_sq(nu):
    wav=c/nu*1.0e8 # Convert frequency to wavelength (this is in angstroms)
    return 911.27/wav # this is the definition in Someville 1964 eq 19 where delta_k^2=k_f^2-k_i^2


# cm4/dyne this opacity is per per H2 molecule and per Unit Electron Pressure
def k_ff_H2_minus(T, nu):
    delta_k_sq=wave_number_sq(nu)
    theta=5040/T
    A=delta_k_sq
    c0=0.09319*theta + 2.857 - 0.9316/theta
    c1=2.600*theta + 6.731 - 4.993/theta
    c2= 35.29*theta- 9.804 - 10.62/theta
    c3=74.52*theta - 62.48 + 0.4679/theta
    return (c0/(A**2) - c1/A + c2 - c3*A)*1e-29 # in cm4/dyne

