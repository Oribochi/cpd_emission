# making the flux for 0.83, 1.3 , 3 and 10 mm
import numpy as np
import matplotlib.pyplot as plt
from flux_dusty_disk import F
from units_astro import *

wavs=np.array([0.83, 1.3, 3, 10])  # in mm

Mpdot=1E-5*Mj/yr
alpha=0.0001

fluxes= np.zeros(len(wavs))
for i, wav in enumerate(wavs):
    fluxes[i] = F(wav, Mpdot, alpha)*1000

print("Fluxes at 0.83, 1.3, 3 and 10 mm:")
for wav, flux in zip(wavs, fluxes):
    print(f"{wav} mm: {flux:.2f} uJy")