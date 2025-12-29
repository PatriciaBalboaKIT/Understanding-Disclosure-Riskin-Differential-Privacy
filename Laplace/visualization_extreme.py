import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Laplace.census import extract_empirical_hours_distribution

##### Get experimental results
main_dir = Path(__file__).parent
results_dir = main_dir / "results"
fig_dir = main_dir / "plots"
# -----------------------
# Parameters
# -----------------------
etas = [0, 40, 80, 100]
#Census
#---------------------------------------------
#attack_template = "census_attack_eta_{}.csv"
#bound_template = "census_bound_eta_{}.csv"

#Uni
#---------------------------------------------
#attack_template = "uni_attack_eta_{}.csv"
#bound_template = "uni_bound_eta_{}.csv"

#Extreme
#---------------------------------------------
attack_template = "extreme_attack_eta_{}.csv"
bound_template = "extreme_bound_eta_{}.csv"

#ReRo bound Hayes
def f_laplace(alpha, eps):
    if alpha < np.exp(-eps) / 2:
        return 1 - np.exp(eps) * alpha
    elif alpha <= 1/2:
        return np.exp(-eps) / (4 * alpha)
    else:
        return np.exp(-eps) * (1 - alpha)

def bound_hayes(eps,eta):
    if eta<100:
        k=1/2
    else:
        k=1
    
    return 1-f_laplace(k, eps)


colors1 = ["blue","purple","pink", "orange"]
colors2=colors1
colors3=colors1


line_widths = [5, 3, 1, 3.2]
# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(9, 6))

for eta, c_bound, c_rad, c_rero, lw in zip(
        etas, colors1, colors2, colors3, line_widths):


    # ---------
    # Read attack CSV
    # ---------
    eps_attack, rero, rad = [], [], []

    path_attack = Path(results_dir) / attack_template.format(eta)
    with path_attack.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps_attack.append(float(row["Epsilon"]))
            rero.append(float(row["ReRo"]))
            rad.append(float(row["RAD"]))

    eps_attack = np.array(eps_attack)
    rero = np.array(rero)
    rad = np.array(rad)
    
    rero_bound = []
    for eps in eps_attack:
        rero_bound.append(bound_hayes(eps,eta))
    # ---------
    # Read bound CSV
    # ---------
    eps_bound, rad_bound = [], []

    path_bound = Path(results_dir) / bound_template.format(eta)
    with path_bound.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps_bound.append(float(row["Epsilon"]))
            rad_bound.append(float(row["RAD_bound"]))

    eps_bound = np.array(eps_bound)
    rad_bound = np.array(rad_bound)

    # ---------
    # Plot
    # ---------

   
    plt.plot(
        eps_bound,
        rad_bound,
        color=c_bound,
        linewidth=lw,
        label=fr"Th.4.3 ($\eta={eta}$)",
        zorder=2
    )

    
    plt.plot(
        eps_attack,
        rero_bound,
        color=c_rero,
        linewidth=lw,
        linestyle="--",
        alpha=0.6,
        zorder=1
    )

    # RAD (marker x)
    plt.scatter(
        eps_attack,
        rad,
        marker="x",
        color=c_rad,
        s=85,
        linewidths=2,
        label=fr"${eta}$-RAD",
        zorder=5
    )

    # ReRo (marker o)
    plt.scatter(
        eps_attack,
        rero,
        marker="o",
        color=c_rero,
        s=70,
        edgecolors="black",
        linewidths=0.6,
        label=fr"${eta}$-ReRo",
        zorder=4
    )

# -----------------------
# Aesthetics
# -----------------------
plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=20)
plt.ylabel(r"Risk", fontsize=20)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.grid(True, linestyle="--", alpha=0.4)



plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.98])
plt.savefig(fig_dir /"extrem_ReRo_vs_Rad_with_bound.png", dpi=300)
plt.show()
