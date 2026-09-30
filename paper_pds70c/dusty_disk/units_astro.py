# Important units for astronomy in cgs
from astropy import constants as const
from astropy import units as u


# now using the astropy
Msun = const.M_sun.cgs.value # g
Rsun = const.R_sun.cgs.value # cm
Lsun = const.L_sun.cgs.value # erg/s
G = const.G.cgs.value # cm^3/g/s^2
sigma = const.sigma_sb.cgs.value # erg/cm^2/s/K^4
c = const.c.cgs.value # cm/s
pc = const.pc.cgs.value # cm
au = const.au.cgs.value # cm
yr = 1*u.yr.to(u.s)
Mearth = const.M_earth.cgs.value # g
Rearth = const.R_earth.cgs.value # cm
Mj = const.M_jup.cgs.value # g
Rj = const.R_jup.cgs.value # cm
Re = const.a0.cgs.value # cm
qe = const.e.esu.value # esu
eV = 1*u.eV.to(u.erg)
kb = const.k_B.cgs.value # erg/K
Rgas = const.R.cgs.value # erg/K/mol
h = const.h.cgs.value # erg s
mH = const.m_p.cgs.value # g
mHe = const.m_p.cgs.value # g
mC = 12*mH # g
mN = 14*mH # g
mO = 16*mH # g
me = const.m_e.cgs.value # g electron