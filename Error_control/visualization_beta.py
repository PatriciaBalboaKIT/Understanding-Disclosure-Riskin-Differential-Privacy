import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

N_thetas = int(input("Introduce the MC sample size: "))

##### Paths
main_dir = Path(__file__).parent
results_dir = main_dir / "results"
plots_dir = main_dir / "plots"
results_template = "BETA_alpha_0.1_beta_0.1_bound_eta_{eta}_N_{N_thetas}.csv"

# -----------------------
# Parameters
# -----------------------
etas = [0.1,0.25,0.5,0.75]   # <-- pon aquí tus etas
#bound_template = "MC_alpha_2_beta_5_bound_eta_{}.csv"

greens = plt.colormaps["Greens"]
oranges = plt.colormaps["Oranges"]
blues = plt.colormaps["Blues"]

#colors1 = [greens(i) for i in np.linspace(0.4, 1, len(etas))]
colors1 =["blue", "orange","purple","pink","black"]
colors2 = [oranges(i) for i in np.linspace(0.4, 1, len(etas))]
colors3 = [blues(i) for i in np.linspace(0.4, 1, len(etas))]

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 6))

for eta, color1,color2,color3 in zip(etas, colors1,colors2,colors3):
    eps_bound =[]
    rad_bound = []
    c_low= []
    c_high = []
    # ---------
    # Read attack CSV
    # ---------
    path_results = Path(results_dir) / results_template.format(eta=eta,N_thetas=N_thetas)
    with path_results.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps_bound.append(float(row["Epsilon"]))
            rad_bound.append(float(row["RAD_bound"]))
            c_low.append(float(row["CI_L"]))
            c_high.append(float(row["CI_H"]))

    eps_bound = np.array(eps_bound)
    rad_bound = np.array(rad_bound)
    c_low=np.array(c_low)
    c_high=np.array(c_high)

    # ---------
    # Plot
    # ---------
    # Bound: green
    plt.plot(
        eps_bound,
        rad_bound,
        color=color1,
        linewidth=2,
        label=f"$\\eta={eta}$ "
    )
    plt.fill_between(eps_bound, c_low,c_high, alpha=0.5,color=color1)


# -----------------------
# Aesthetics
# -----------------------
plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=16)
plt.ylabel("RAD", fontsize=16)
#plt.ylim(-0.01, 0.55)
plt.grid(True)
plt.legend(fontsize=18, ncol=2)
plt.tight_layout()

plt.savefig(plots_dir/f"beta_{N_thetas}.png", dpi=300)
plt.show()
