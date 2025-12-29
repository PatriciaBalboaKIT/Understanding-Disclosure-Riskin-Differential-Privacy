import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import beta
from scipy.optimize import root_scalar
import csv

from multiprocessing import Pool
from multiprocessing import cpu_count
n_jobs = cpu_count()  # usa todos los cores disponibles

from pathlib import Path

##### Paths
main_dir = Path(__file__).parent
results_dir = main_dir / "results"
results_template = "BETA_alpha_0.1_beta_0.1_bound_eta_{eta}_N_{N_thetas}.csv"
#--------------------------------------------------------
# MC simulation parameters
#--------------------------------------------------------
N_zetas = 500
N_thetas = 500
M=500
#--------------------------------------------------------
# Prior Distribution (Discrete) & Datas Domain
#--------------------------------------------------------
A=0
B=1
sensitivity=np.abs(A-B)
#prior dist formula
alpha_val = 0.1
beta_val = 0.1
def pi(zeta):
    # map zeta ∈ [A,B] → x ∈ [0,1]
    x = (zeta - A) / (B - A)

    if x < 0 or x > 1:
        return 0.0

    return beta.pdf(x, alpha_val, beta_val) / (B - A)
def sampler_beta(size):
    return A + (B - A) * np.random.beta(alpha_val, beta_val, size=size)


#--------------------------------------------------------
# Clipped [A,B] Laplace Distribution Definition
#--------------------------------------------------------
#(A,B)
def p_1(zeta,theta,b):
    p_val = np.exp(-np.abs(zeta-theta)/b)
    normalizer = b*(2-np.exp(-zeta/b)-np.exp(-(1-zeta)/b))
    return p_val/normalizer
def marginal_p_theta(theta,zetas, b):  
    sumand = []
    for zeta in zetas:
            # marginal p(theta)  
            p_zeta_theta = p_1(zeta,theta,b)
            sumand.append(p_zeta_theta)
    marginal=np.mean(sumand)
    return marginal
     
# ----------------------------------------------------
# f(theta)=max_{a} I_a, I_a= int_{l(a)}^u(a) (p(theta|z)-p(theta))Pi_z dz
# ----------------------------------------------------
# (A,B)
def I_interior(b, theta, zetas, l, u,marginal):
    sumand = []
    z_size=len(zetas)
    mask = (zetas>=l)& (zetas<=u)
    for zeta in zetas[mask]:  
            p_zeta_theta = (p_1(zeta,theta,b)-marginal)
            sumand.append(p_zeta_theta)
    integral = (np.sum(sumand)/z_size) #MC2
    
    return integral 

def w(theta,zeta,b,p):
    w_val = (p_1(zeta,theta,b)-p)*pi(zeta)
    return w_val 
# --- Derivatives ---
def g_prime_interior(a, b, theta, p, eta):
    lower = a - eta
    upper = a + eta
    return (w(theta,upper,b,p) - w(theta,lower,b,p))

def g_prime_left(a, b, theta, p, eta):
    upper = a + eta
    return w(theta,upper,b,p)
def g_prime_right(a, b, theta, p, eta):
    lower = a - eta
    return -w(theta,lower,b,p)
    
def find_candidates_to_max_big_etas(theta,eta,p_0,b):
    candidates = []
    try:
        sol = root_scalar(lambda a: g_prime_left(a, b, theta, p_0, eta),bracket=[A, B - eta],method='bisect')
        if sol.converged:
            candidates.append(sol.root)
    except ValueError:
        pass  
    candidates.append(A)
    candidates.append(B-eta)
    try:
        sol = root_scalar(lambda a: g_prime_right(a, b, theta, p_0, eta),bracket=[A+eta, B],method='bisect')
        if sol.converged:
            candidates.append(sol.root)
    except ValueError:
        pass  
    candidates.append(B)
    candidates.append(A+eta)  
    return candidates
def find_candidates_to_max(theta,eta,p_0,b):
    candidates = []
    try:
        sol = root_scalar(lambda a: g_prime_left(a, b, theta, p_0, eta),bracket=[A, A + eta],method='bisect')
        if sol.converged:
            candidates.append(sol.root)
    except ValueError:
        pass  
    candidates.append(A)
    candidates.append(A+eta)
    try:
        sol = root_scalar(lambda a: g_prime_right(a, b, theta, p_0, eta),bracket=[B-eta, B],method='bisect')
        if sol.converged:
            candidates.append(sol.root)
    except ValueError:
        pass  
    candidates.append(B)
    candidates.append(B-eta)
    try:
        sol = root_scalar(lambda a: g_prime_interior(a, b, theta, p_0, eta),bracket=[A+eta, B-eta],method='bisect')
        if sol.converged:
            candidates.append(sol.root)
    except ValueError:
        pass   
    return candidates
# ----------------------------------------------------
# Main
# ----------------------------------------------------
def estimate_R_eps(eps, eta, N_theta,N_zetas):
    b = sensitivity / eps

    #z_i~Pi
    zetas = sampler_beta(N_zetas)
    

    #MC over interval [A,B]
    #-------------------------------------------------------------
    "MC for integral int_{A}^B f(theta) dtheta, we DO multiply by the volume |B-A|"

    thetas = np.random.uniform(A, B, size=N_theta)
    max_vals = []
    for theta in thetas:
        p_theta=marginal_p_theta(theta,zetas, b)
        if eta<0.5:
            canditates = np.array( find_candidates_to_max(theta,eta,p_theta,b))
        else:
            canditates = np.array( find_candidates_to_max_big_etas(theta,eta,p_theta,b))
        a1 = np.maximum(A, canditates - eta)
        a2 = np.minimum(B, canditates + eta)
        integrals = []
        for l, u in zip(a1,a2):
                I=I_interior(b, theta, zetas, l, u,p_theta)
                integrals.append(I)
        max_vals.append(np.max(integrals)) #Empirical max over a_grid

    R_hat = np.mean(max_vals)*(B-A) #mean*Volume

    R_final = R_hat 
    stderr = np.std(max_vals, ddof=1) / math.sqrt(N_theta)
    return R_hat,R_final, stderr


# ----------------------------------------------------
# eps ∈ [1,10]
# ----------------------------------------------------
def _estimate_R_wrapper(args):
    eps, eta, N_theta,N_zetas = args
    R_hat,R_final, stderr = estimate_R_eps(eps=eps, eta=eta, N_theta=N_theta,N_zetas=N_zetas)
    return R_final

def sweep_eps(eps_values, eta, N_theta, M,N_zetas, n_jobs=n_jobs):
    R_finals = []
    CI_lower = []
    CI_upper = []

    
    with Pool(processes=n_jobs) as pool:
        for eps in eps_values:
            
            tasks = [(eps, eta, N_theta,N_zetas)] * M

            R_samples = pool.map(_estimate_R_wrapper, tasks)
            R_samples = np.asarray(R_samples)

            mean_R = np.mean(R_samples)
            ci_l = np.percentile(R_samples, 2.5)
            ci_u = np.percentile(R_samples, 97.5)

            R_finals.append(mean_R)
            CI_lower.append(ci_l)
            CI_upper.append(ci_u)

            print(
                f"eps = {eps:.2f}  ->  R = {mean_R:.6e},  CI_L = {ci_l:.2e}"
            )

    return R_finals, CI_lower, CI_upper

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    eta_values = [0.1,0.25,0.5,0.75,1]
    colors = ['blue', 'green', 'purple','orange','black']
    greens = plt.colormaps["Greens"]
    colors2 = [greens(i) for i in np.linspace(0.4, 1, len(eta_values))]
    
    eps_values = np.linspace(0.1, 10.0, 3)

    #-------------------------------------------------
    ## Theory TV
    #--------------------------------------------------   
    plt.figure(figsize=(8, 6))
    
    for eta, color,color2 in zip(eta_values, colors,colors2):
        print(f"Computing eta=",eta)
        R_finals, CI_low, CI_high = sweep_eps(
            eps_values,
            eta=eta,
            N_theta=N_thetas,
            N_zetas=N_zetas,
            M=M
        )
        
        # ------------------------------------------------
        # PLOT
        # ------------------------------------------------
        plt.plot(eps_values, R_finals,label=f"{eta}-RAD",color=color,linewidth=1)
        plt.fill_between(eps_values, CI_low,CI_high, alpha=0.5,color=color)
        #plt.plot(eps_values, uniform_bound,label=f"{eta}Id Attack",color=color2,linewidth=1)
        
        # ----------------------------------
        # SAVE RESULTS (same format)
        # ----------------------------------
        path_results = Path(results_dir) / results_template.format(eta=eta,N_thetas=N_thetas)
        with path_results.open("w") as fo:
            writer = csv.writer(fo)
            writer.writerow(["Epsilon", "RAD_bound", "CI_L","CI_H"])
            for eps, rad, ci_low,ci_hi in zip(eps_values, R_finals,CI_low, CI_high):
                writer.writerow([eps, rad, ci_low,ci_hi])


    
    plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=18)
    plt.ylabel("RAD", fontsize=16)


    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f"beta_biased_{N_thetas}.png", dpi=300)


