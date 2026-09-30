# Calculator bound-free opacity of the H- using T.L.  John 1987
import numpy as np
from units_astro import *
from numpy import exp, pi, sqrt, log10
import matplotlib.pyplot as plt
from k_ff_metals import k_ff_metals

# here wavelengh unit is in um
def f(wav):
    # this function is the f factor in the bound-free opacity
    # it is a function of the wavelength (using um)
    wav_0=1.6419 # the wavelength of the threshold in um (0.75 eV)
    Cn=[152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
    suma=0
    for i in range(6):
        suma+=Cn[i]*(1/(wav)-1/(wav_0))**(i/2)
    return suma

# the cross section in cm² (this is correct and the same as in Gray 2021)
def cross_section(wav):
    wav_0=1.6419 # the wavelength of the threshold in um (0.75 eV)
    cross_sec=(wav)**3*(1/wav-1/wav_0)**(3/2)*f(wav) # in 10**-18 cm^2
    #print('cross_sec', cross_sec, 'wav', wav)
    return cross_sec*10**(-18) # in cm²


# # example plot the absorption cross-section for wav in 0 to 15000 Angstroms
# wavs=np.linspace(1000, 16000, 100) # in Ang
# a_bf_array=np.array([cross_section(wav*1.0e-4) for wav in wavs])*10**(17) # convert wav Ang to um and in cm² to 10^-17 cm²
# plt.plot(wavs, a_bf_array)
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Absorption cross-section (10⁻¹⁷ cm^2 per H- ion)')
# plt.title('Absorption cross-section for H- bound-free')
# plt.yscale('log')
# plt.ylim(0.1,10)
# plt.show()

# this k_bf_Hminus calculates the bf opacity for a given temperature and frequency
def k_bf_Hminus(T, nu):
    wav=c/nu*1.0e4 # wavelength in um
    wav_0=1.6419 # the wavelength of the threshold in um (0.75 eV)
    if wav>wav_0:
        return 0 # 'wavelength should be smaller than the threshold wavelength'
    else:
        nu0=c/(wav_0*1.0e-4) # threshold frequency in Hz
        k0=h*nu0/(kb*T)
        k=h*nu/(kb*T)
        #a=h*c/kb*10**4 # the constant of the exponential the 10⁴ is to convert cm -> um
        cross_sec=cross_section(wav) # in cm^2
        #theta=5040/T
        return 0.75*T**(-5/2)*exp(k0)*(1-exp(-k))*cross_sec
        #return 4.158*10**(-10)*cross_sec*theta**(5/2)*10**(0.754*theta)

# Now the free-free from the H- is given using the tables of Bell and Berrintong 1987

columnA1=[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
columnA2=[2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520]
columnA3=[-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390]
columnA4=[2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380]
columnA5=[-696.2710, -1841.4000, 8624.9700, -10051.5300, 4400.0670, -655.0200]
columnA6=[88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850]


columnB1=[518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285]
columnB2=[473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600]
columnB3=[-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571]
columnB4=[115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097]
columnB5=[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
columnB6=[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

MatrixA=np.array([columnA1, columnA2, columnA3, columnA4, columnA5, columnA6])
MatrixB=np.array([columnB1, columnB2, columnB3, columnB4, columnB5, columnB6])

def wav_func(wav, column):
    result=wav**2*column[0]
    for j in range(1,len(column)):
        result+=wav**(1-j)*column[j]
    return result

# and the k_ff_Hminus is calculated using John TL, Bell & Berrington. With wavelength in um and T in Kelvin.
def k_ff_Hminus(T, nu):
    wav=c/nu*1.0e4 # in um
    n=6
    i=1
    sum=0
    if wav<0.1823:
        return 0
    elif wav>0.1823 and wav<0.3645:
        Matrix=MatrixB
    else:
        Matrix=MatrixA
    while i<n+1:
        column=Matrix[i-1]
        sum+=(5036.36/T)**((i+1)/2)*wav_func(wav,column)
        i+=1
    return sum*10**(-29) # debería ser por 10**-(29)

# # (Example when k_ff and k_bf are in wavs) NOW is in frecuencies
# k_bf_array=[]
# k_ff_array=[]
# k_tot=[]
# k_ff_metals_array=[]
# waves=np.logspace(3, 5, 100)*1.0e-4 # in um
# nu_array=c/(waves*1.0e-4) # in Hz
# T=5040
# for wav, nu in zip(waves, nu_array):
#     k_bf_array.append(k_bf_Hminus(T, nu))
#     k_ff_array.append(k_ff_Hminus(T, nu))
#     k_tot.append(k_bf_Hminus(T, nu)+k_ff_Hminus(T, nu))
#     k_ff_metals_array.append(k_ff_metals(T, nu))

# plt.plot(waves*10000, k_bf_array)
# plt.plot(waves*10000, k_ff_array)
# plt.plot(waves*10000, k_tot)
# plt.plot(waves*10000, k_ff_metals_array)
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Opacities (cm^2/dynes)')
# plt.title('k_bf for H-')
# plt.legend(['k_bf_H-', 'k_ff_H-', 'k_tot_H-', 'k_ff_metals'])
# plt.loglog()
# plt.ylim(1e-28, 1e-23)
# # plt.xticks(np.array([0.32, 0.6, 1.0, 1.5, 2.5])*10000, ['3200', '6000', '10000', '15000', '25000'])
# plt.show()

# T=5040
# wav_arr=np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.65,1.8,2,2.5,3,4.5,9,11,15]) # in um
# k_ff_bf_arr=np.array([k_bf_Hminus(T, c/(wav*1.0e-4))*10 + k_ff_Hminus(T, c/(wav*1.0e-4)) for wav in wav_arr])
# print('Wavelength (um):', wav_arr)
# print('k_ff + k_bf (cm^2/dynes):', [float(np.round(k*1e26,2)) for k in k_ff_bf_arr])
# # only freefree
# k_ff_arr=np.array([k_ff_Hminus(T, c/(wav*1.0e-4)) for wav in wav_arr])
# print('Only k_ff (cm^2/dynes):', [float(np.round(k*1e26,2)) for k in k_ff_arr])

# The free-free is in the correct units comparing with Bell & Berrington 1987 and John 1988
