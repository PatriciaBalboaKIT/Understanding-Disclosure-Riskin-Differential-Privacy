import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import beta
import csv
from collections import defaultdict
from pathlib import Path
from Laplace.census import extract_empirical_hours_distribution

# PARALLELIZATION MAGIC:
from multiprocessing import Pool
from functools import partial

##### Paths
main_dir = Path(__file__).parent
results_dir = main_dir / "results"
results_template = "extreme_bound_eta_{eta}.csv"
#--------------------------------------------------------
# Prior Distribution (Discrete) & Data Domain
#--------------------------------------------------------
A=0
B=100
sensitivity=np.abs(A-B)
#empirical_size, pi = extract_empirical_hours_distribution("adult.csv")
#pi={zeta: pi[zeta] for zeta in domain} #uniform 
domain = np.linspace(0, 100, 101)

#domain=np.array([40,41,42,50,60])
m=len(domain)
pi = defaultdict(float)
pi[0] = 0.5
pi[100] = 0.5
print(pi)
#--------------------------------------------------------


#--------------------------------------------------------
# Clipped [A,B] Laplace Distribution Definition
#--------------------------------------------------------
#Clipping Extremes {A,B}
def pr_extrems(zeta,theta,b):
    if theta == A:
        p_val = 1/2*np.exp(-np.abs(A-zeta)/b)
    elif theta == B:
        p_val = 1/2*np.exp(-np.abs(zeta-B)/b)
    else:
        print("This is not a extreme!")

    return p_val

def marginal_extremes(theta,zetas, b):
    sumand= []
    for zeta in zetas:
        p_zeta_theta= pr_extrems(zeta,theta,b)*pi[zeta]
        sumand.append(p_zeta_theta)
    marginal=np.sum(sumand)
    return marginal

#(A,B)
def p_1(zeta,theta,b):
    p_val = (1/(2*b))*np.exp(-np.abs(zeta-theta)/b)
    return p_val

def marginal_p_theta(theta,zetas, b):  
    sumand = []
    for zeta in zetas:
            # marginal p(theta)  
            p_zeta_theta = p_1(zeta,theta,b)*pi[zeta]
            sumand.append(p_zeta_theta)
    marginal=np.sum(sumand)
    return marginal

# ----------------------------------------------------
# I_a 
# ----------------------------------------------------
# (A,B)
def I_interior(b, theta, zetas, l, u):
    sumand = []
    p_theta = marginal_p_theta(theta,zetas, b)
    mask = (zetas>=l)& (zetas<=u)
    for zeta in zetas[mask]:  
            p_zeta_theta = (p_1(zeta,theta,b)-p_theta)*pi[zeta]
            sumand.append(p_zeta_theta)
    integral = np.sum(sumand)
    
    return integral 
#A,B
def I_extremes(b, theta, zetas, l, u):
    sumand = []
    pr_theta = marginal_extremes(theta,zetas, b)
    mask = (zetas>=l)& (zetas<=u)
    for zeta in zetas[mask]: 
            p_zeta_theta = (pr_extrems(zeta,theta,b)-pr_theta)*pi[zeta]
            sumand.append(p_zeta_theta)
    integral = np.sum(sumand)
    
    return integral 

#def out for thetas paralellization
#----------------------------------------------

def max_I_for_theta(theta, b, zetas, a1, a2):
    vals = []
    for l, u in zip(a1, a2):
        I = I_interior(b, theta, zetas, l, u)
        vals.append(I)
    return np.max(vals)
# ----------------------------------------------------
# Monte Carlo  R(eps)
# ----------------------------------------------------
def estimate_R_eps(eps, eta, N_theta):
    b = sensitivity / eps

    #zetas=np.array([0,1]) MIA
    zetas = domain 
    #a_grid = np.array([0,1]) MIA
    a_grid = zetas

    #Success intervals of each z
    a1 = np.maximum(A, a_grid - eta)
    a2 = np.minimum(B, a_grid + eta)
    
    #theta=A
    canditates_0 = []
    for l, u in zip(a1,a2):
            I=I_extremes(b, A, zetas, l, u)
            canditates_0.append(I)
    w_0 = np.max(canditates_0)
    
    

    #theta=B
    canditates_1 = []
    for l, u in zip(a1,a2):
            I=I_extremes(b, B, zetas, l, u)
            canditates_1.append(I)
    w_1 = np.max(canditates_1)
    

    #MC over interval (A,B)
    #thetas = np.random.uniform(A, B, size=N_theta)
    thetas = np.linspace(A, B, N_theta)[1:-1]
    
    max_vals = []
    with Pool() as pool:
        max_vals = pool.map( partial(max_I_for_theta,b=b,zetas=zetas,a1=a1, a2=a2),thetas)

    R_hat = np.mean(max_vals)*(B-A)

    R_final = R_hat + w_0 + w_1
    
    stderr = np.std(max_vals, ddof=1) / math.sqrt(N_theta)
    return R_hat,R_final, stderr


# ----------------------------------------------------
# eps ∈ [1,10]
# ----------------------------------------------------
def sweep_eps(eps_values, eta, N_theta):
    Rs = []
    R_finals = []
    SEs = []

    for k, eps in enumerate(eps_values):
        R, R_final, se = estimate_R_eps(
            eps=eps,
            eta=eta,
            N_theta=N_theta,
        )
        Rs.append(R)
        R_finals.append(R_final)
        SEs.append(se*(B-A))

        print(f"eps = {eps:.2f}  ->  R = {R:.6e},  SE = {se:.2e}")

    return np.array(Rs),np.array(R_finals), np.array(SEs)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    #eta_values = np.linspace(0.0,1.0, 15)  
    eta_values = [0,40,80,100]
    eps_values = np.linspace(1,10,6)

    for eta in eta_values:
        Rs,R_finals, SEs = sweep_eps(
            eps_values,
            eta=eta,
            N_theta=10000
        )

        CI_low = R_finals - 1.96 * SEs
        CI_high = R_finals + 1.96 * SEs
        
        # ----------------------------------
        # SAVE RESULTS (same format)
        # ----------------------------------
        path_results = Path(results_dir) / results_template.format(eta=eta)
        with path_results.open("w") as fo:
            writer = csv.writer(fo)
            writer.writerow(["Epsilon", "RAD_bound", "standad error"])
            for eps, rad, se in zip(eps_values, R_finals,SEs):
                writer.writerow([eps, rad, se])