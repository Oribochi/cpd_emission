# Adding free free emission from the CPD
from numpy import exp, sqrt, pi, log
from astropy import constants as const
from astropy import units as u
# calculating gaunt factor

euler_constant = 0.5772156649
gamma=exp(euler_constant)
print(gamma)

gaunt_constant = 2**(3/2)*const.k_B**(3/2)/(gamma**(5/2)*const.m_e**(1/2)*const.e.esu**2*pi)

print("gaunt_constant = ", gaunt_constant.cgs)
print("gaunt_constant in 1/GHz = ", gaunt_constant.cgs/(1e9))

# now the kff constatnt

k_ff_const=4*pi*const.e.esu**6/(3*sqrt(3)*const.m_e**2*const.h*const.c)

v_prom_const=sqrt(pi*const.k_B/(2*const.m_e))

cte=k_ff_const/v_prom_const
print("k_ff_const = ", cte.cgs)


