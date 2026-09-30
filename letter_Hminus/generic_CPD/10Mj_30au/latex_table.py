# making a latex table

import pandas as pd
import numpy as np
# =========================================================
# Read the flux data
# =========================================================

flux_cols = [
    "Mpdot",
    "Bps_min",
    "Bps_max",
    "flux_min_B4",
    "flux_min_B7",
    "flux_min_B9",
    "flux_max_B4",
    "flux_max_B7",
    "flux_max_B9"
]

flux = pd.read_csv(
    "fluxes_magnetospheric_accretion_10Mj_30au.txt",
    sep=r"\s+",
    comment="#",
    names=flux_cols
)


# =========================================================
# Read the source/radius data
# =========================================================

source_cols = [
    "Mpdot_min",
    "Bps_min_source",
    "r_min",
    "source_min",
    "Mpdot_max",
    "Bps_max_source",
    "r_max",
    "source_max"
]

source = pd.read_csv(
    "cont_magnetospheric_accretion_10Mj_30au.txt",
    sep=r"\s+",
    comment="#",
    names=source_cols
)


# =========================================================
# Combine the information
# =========================================================

# what I want is something with this format: 
# $M_{\rm p}$ & $\dot{M}_{\rm p}$  & $B_{\rm ps,min}$\tablefootmark{a} & $B_{\rm ps,max}$\tablefootmark{b} & $r_{\rm min}$\tablefootmark{c} & $r_{\rm max}$\tablefootmark{d}  & $F_{\rm B4,min}$ & $F_{\rm B7,min}$ & $F_{\rm B9,min}$ & $F_{\rm B4,max}$ & $F_{\rm B7,max}$ & $F_{\rm B9,max}$ \\
#  $(M_{\rm Jup})$ & $(M_{\rm Jup}/\rm yr)$ & (G) & (G) & (au) & (au) & (mJy)  & (mJy) & (mJy) & (mJy) & (mJy) & (mJy)\\
#  \hline
#1& $6.95\times 10^{-7}$ & 70 & 70 & 0.028 & 0.028 & $1.15\times 10^{-3}$ & $1.41\times 10^{-3}$ & $1.45\times 10^{-3}$ & $1.15\times 10^{-3}$ & $1.41\times 10^{-3}$ & $1.45\times 10^{-3}$ \\
#1 & $10^{-6}$ & 84 & 366 & 0.035 & 0.021 & $3.71\times 10^{-3}$ & $1.45\times 10^{-3}$ & $2.25\times 10^{-2}$ & $3.87\times 10^{-10}$ & $3.85\times 10^{-10}$ & $3.81\times 10^{-10}$ \\


table = pd.DataFrame()

table["Mp"] = np.ones(len(flux))*10  # Mp is always 10Mjup in this case

table["Mpdot"] = flux["Mpdot"]
table["Bps_min"] = flux["Bps_min"]
table["Bps_max"] = flux["Bps_max"]

table["r_min"] = source["r_min"]
table["r_max"] = source["r_max"]

table["flux_min_B4"] = flux["flux_min_B4"]
table["flux_min_B7"] = flux["flux_min_B7"]
table["flux_min_B9"] = flux["flux_min_B9"]

table["flux_max_B4"] = flux["flux_max_B4"]
table["flux_max_B7"] = flux["flux_max_B7"]
table["flux_max_B9"] = flux["flux_max_B9"]


# =========================================================
# Format the numbers
# =========================================================

# planet mass
table["Mp"] = table["Mp"].map(
    lambda x: f"{x:.0f}"
)
# Accretion rate
table["Mpdot"] = table["Mpdot"].map(
    lambda x: f"{x:.2e}" # write the scientific notation (10^x) with 2 decimal places
)
table["Mpdot"] = table["Mpdot"].map(
    lambda x: x.replace("e", r"$\times 10^{") + "}$"
)

# Magnetic field
table["Bps_min"] = table["Bps_min"].map(
    lambda x: f"{x:.0f}"
)

table["Bps_max"] = table["Bps_max"].map(
    lambda x: f"{x:.0f}"
)

# Radius *1000
table["r_min"] = table["r_min"].map(
    lambda x: f"{x:.3f}"
)

table["r_max"] = table["r_max"].map(
    lambda x: f"{x:.3f}"
)

# Fluxes
for col in [
    "flux_min_B4",
    "flux_min_B7",
    "flux_min_B9",
    "flux_max_B4",
    "flux_max_B7",
    "flux_max_B9"
]:
    table[col] = table[col].map(
        lambda x: f"{x:.2e}"
    )

# replace with scientific notation in LaTeX format
for col in [
    "flux_min_B4",
    "flux_min_B7",
    "flux_min_B9",
    "flux_max_B4",
    "flux_max_B7",
    "flux_max_B9"
]:
    table[col] = table[col].map(
        lambda x: x.replace("e", r"$\times 10^{") + "}$"
    )


# =========================================================
# Rename columns
# =========================================================

table.columns = [
    r"$M_{\rm p}$",
    r"$\dot{M}_{\rm p}$",
    r"$B_{\rm ps,min}$",
    r"$B_{\rm ps,max}$",
    r"$r_{\rm min}$",
    r"$r_{\rm max}$",
    r"$F_{\rm B4,min}$",
    r"$F_{\rm B7,min}$",
    r"$F_{\rm B9,min}$",
    r"$F_{\rm B4,max}$",
    r"$F_{\rm B7,max}$",
    r"$F_{\rm B9,max}$"
]


# =========================================================
# Convert to LaTeX
# =========================================================

latex_table = table.to_latex(
    index=False,
    escape=False,
    column_format="cccccccccccc",
    header=True
)


# =========================================================
# Add table environment
# =========================================================

latex_output = r"""
\begin{table}
    \caption{Predicted flux densities for the circumplanetary disk models.}
    \label{tab:model_fluxes}
    \centering
""" + latex_table + r"""
\end{table}
"""


# Print LaTeX code
print(latex_output)


# Save LaTeX code
with open("model_flux_table.tex", "w") as f:
    f.write(latex_output)