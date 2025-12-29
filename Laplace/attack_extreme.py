# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from Laplace.census import extract_empirical_hours_distribution
from Laplace.census_bound import pr_extrems,p_1, marginal_extremes,  marginal_p_theta, I_interior, I_extremes
import pandas as pd
from pathlib import Path
# PARALLELIZATION MAGIC:
from multiprocessing import Pool
from functools import partial
import csv


##### Paths
main_dir = Path(__file__).parent
results_dir = main_dir / "results"
results_template = "extreme_attack_eta_{eta}.csv"



#Experiment parameters

mechanism_iterations =1000
epsilons = np.linspace(1,10,6)
etas = [0,40,80,100]

#--------------------------------------------------------
# Prior Distribution (Discrete) & Data Domain
#--------------------------------------------------------
A=0
B=100
sensitivity=np.abs(A-B)
#domain=np.linspace(A,B,101)
N=1
domain = np.linspace(0, 100, 101)
m=len(domain)
pi = defaultdict(float)
pi[0] = 0.5
pi[100] = 0.5
print(pi)

sample_size = m
num_challengers =m
# ===========================
# Load D_
# ===========================
df = pd.read_csv(main_dir/"adult.csv")
column: str = "hours-per-week"
hours = (
    pd.to_numeric(df[column], errors="coerce")
      .dropna()
      .astype(int)
)

hours_subset = hours.iloc[:999]


total_hours = hours_subset.sum()

print("q(D_)=:", total_hours)

A_sums=total_hours
B_sums=total_hours+100

#----------------------------------------------------------
# Optimal Attack
#-------------------------------------------------------------
def attack(output, scale, eta):
    #a_grid = np.array([0,1]) MIA
    a_grid = domain

    #Success intervals of each z
    a1 = np.maximum(A, a_grid - eta)
    a2 = np.minimum(B, a_grid + eta)

    if output == A:
        candidates_0 = {}
        for a, l, u in zip(a_grid, a1, a2):
                I=I_extremes(scale, A, domain, l, u)
                candidates_0[a]=I
        
        attack_output = max(candidates_0, key=candidates_0.get)

    if  output == B:
        canditates_1 = {}
        for a, l, u in zip(a_grid, a1, a2):
                I=I_extremes(scale, B, domain, l, u)
                canditates_1[a]=I
        
        attack_output = max(canditates_1, key=canditates_1.get)

    else:
        canditates = {}
        for a, l, u in zip(a_grid, a1, a2):
            I=I_interior(scale, output, domain, l, u)
            canditates[a]=I
        attack_output = max(canditates, key=canditates.get)

    return attack_output


# ----------------------
# Core computation 1 mech iteration
# ----------------------
def RAD_computation(epsilon,eta):
    scale=sensitivity/epsilon

    records = domain
    #print("records", records)
    inputs = records + total_hours
    #print("sums", inputs)
    np.random.seed()
    noise = np.random.laplace(0, scale, len(domain))
    global_outputs = np.clip(inputs + noise, A_sums, B_sums)
    #print("mechanism output",global_outputs)
    outputs = global_outputs-total_hours
    #print("attack first step", outputs)
    attack_outputs = []
    for output in outputs:
        attack_outputs.append(attack(output, scale, eta))


    #ReRo 
    ReRo_success = []
    for user,attack_output in zip(records,attack_outputs):
        success = (np.abs(attack_output - user) <= eta/N)
        ReRo_success.append((pi[user] * success))
   
    ReRo = np.sum(ReRo_success)

    #Correction
    corrections = []
    for user,attack_output in zip(records,attack_outputs):
        correction_user = []
        for challenger in records:
            success = (np.abs(attack_output - challenger) <= eta/N)*pi[challenger]
            correction_user.append(success)
        corrections.append(np.sum(correction_user)*pi[user])
    
    Corr = np.sum(corrections)

    Rad = ReRo - Corr
   
    return ReRo, Corr, Rad

#----------------------
# Execution
#------------------------
def run_single_RAD(args):
    epsilon, eta = args
    return RAD_computation(epsilon, eta)

def experiment(epsilon,eta,mechanism_iterations):
    m=mechanism_iterations
    REROs = []
    CORs = []
    RADs = []

    with Pool() as pool:
        results = pool.map(
            run_single_RAD,
            [(epsilon, eta)] * mechanism_iterations
        )

    REROs = [r[0] for r in results]
    CORs  = [r[1] for r in results]
    RADs  = [r[2] for r in results]

    ReRo_final=np.mean(REROs)
    Corrections_final=np.mean(CORs)
    Rad_final=np.mean(RADs)
    return ReRo_final,Corrections_final,Rad_final
# ----------------------
# Plot
# ----------------------
if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    #colors = ['blue', 'green', 'red','orange']
    colors = ['blue', 'green', 'purple','orange','black']
   
    blue_classic = plt.colormaps["Blues"]
    colors_1 = [blue_classic(i) for i in np.linspace(0.2, 1, len(etas))]

    green_classic = plt.colormaps["Greens"]
    colors_0 = [green_classic(i) for i in np.linspace(0.2, 1, len(etas))]

    for eta,color_0,color_1 in zip(etas,colors_0,colors_1):
        print(f"computing eta= {eta}")
        ReRo_census = {}
        Corr_census = {}
        RAD_census = {}
        for eps in epsilons:
            print(f"computing eps={eps}")
            ReRo_census[eps], Corr_census[eps], RAD_census[eps] = experiment(eps,eta,mechanism_iterations)
    
        ReRo_list = [ReRo_census[eps] for eps in epsilons]
        Corr_list = [Corr_census[eps] for eps in epsilons]
        RAD_list = [RAD_census[eps] for eps in epsilons]
        plt.scatter(epsilons, RAD_list,label=f"{eta}-RAD", marker="o",color=color_0)
        plt.scatter(epsilons, ReRo_list,label=f"{eta}-ReRo",marker="x",color=color_1)

        # Save as csv:
        print(f"writting results in census_atextreme_attack_eta_{eta}.csv")
        path_results = Path(results_dir) / results_template.format(eta=eta)
        with path_results.open("w") as fo:
            writer = csv.writer(fo)
            writer.writerow(["Epsilon", "ReRo", "Corr", "RAD"])
            for eps, rero, corr, rad in zip(epsilons, ReRo_list, Corr_list, RAD_list):
                writer.writerow([eps, rero, corr, rad])

    plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=18)
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 10.1)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"extreme_attack_J_{mechanism_iterations}.png", dpi=300)

