# cross section H

import numpy as np
import matplotlib.pyplot as plt

# Constants
sigma_0 = 6.3e-18  # cm^2 (cross-section at Lyman limit)
E_0 = 13.6  # eV, ionization energy from n=1
hnu = np.linspace(0, 20, 500)  # Photon energy range in eV for n=1

def photoionization_cross_section(n, photon_energy_eV):
    """
    Approximate hydrogen photoionization cross-section from level n.
    photon_energy_eV must be >= 13.6 / n^2
    """
    E_n = E_0 / n**2  # ionization threshold from level n
    sigma = np.zeros_like(photon_energy_eV)
    valid = photon_energy_eV >= E_n
    # Using scaling: σ ∝ (E_n / hν)^3
    try:
        sigma[valid] = sigma_0/n**2 * (E_n / photon_energy_eV[valid])**3
    except:
        if valid:
            sigma= sigma_0/n**2 * (E_n / photon_energy_eV)**3
    return sigma

def total_cross_section(photon_energy_eV):
    total_sigma =0
    for i in range(1, 6):
        sigma = photoionization_cross_section(i, photon_energy_eV)
        total_sigma += sigma
    return total_sigma

print("Total Cross-Section:", total_cross_section(0.544), "cm² at 0.544 eV")
print("Total Cross-Section:", total_cross_section(0.85), "cm² at 0.85 eV")
print("Total Cross-Section:", total_cross_section(1.51), "cm² at 1.51 eV")

# # Plot cross-sections
# plt.figure(figsize=(8, 6))
# colors = ['black', 'blue', 'green', 'orange', 'red']
# sigma_tot=np.zeros(500)
# for n in range(1, 6):
#     E_n = E_0 / n**2
#     hnu_n = np.linspace(0, 20, 500)
#     sigma_n = photoionization_cross_section(n, hnu_n)
#     sigma_tot += sigma_n
#     plt.plot(hnu_n, sigma_n, label=f'n={n}', color=colors[n-1])

# plt.plot(hnu_n, sigma_tot, label='Total Cross-Section', color='purple', lw=2)
# plt.yscale('log')
# plt.xlabel('Photon Energy (eV)')
# plt.ylabel('Photoionization Cross-Section (cm²)')
# plt.title('Hydrogen Photoionization Cross-Section (Approximate)')
# plt.legend()
# plt.grid(True, which='both', ls='--', alpha=0.5)
# plt.tight_layout()
# plt.show()