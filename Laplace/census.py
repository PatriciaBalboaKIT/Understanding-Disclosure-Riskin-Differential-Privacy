import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def extract_empirical_hours_distribution(
    csv_path: str,
    column: str = "hours-per-week",
    min_h: int = 0,
    max_h: int = 100
):
    """
    Extrae la distribución empírica discreta de horas por semana.

    Retorna:
        pi_vector : np.ndarray de tamaño (max_h-min_h+1,)
            Vector [pi_min_h, ..., pi_max_h].

        pi_df : pd.DataFrame
            DataFrame con columnas ['hours','pi'].
    """

    # ===========================
    # Load
    # ===========================
    df = pd.read_csv(csv_path)

    # ===========================
    # Clean column
    # ===========================
    hours = (
        pd.to_numeric(df[column], errors="coerce")
          .dropna()
          .astype(int)
    )

    # Hours Range
    hours = hours[(hours >= min_h) & (hours <= max_h)]
    N=len(hours)

    # ===========================
    # Empirical PMF
    # ===========================

    probs = hours.value_counts(normalize=True)


    pi = {
        h: float(probs.get(h, 0.0))
        for h in range(min_h, max_h + 1)
    }


    

    return N, pi


# ===========================
# 1. EXECUTION
# ===========================
if __name__ == "__main__":
    N,pi = extract_empirical_hours_distribution("adult.csv")
    print(pi.values())
    # ===========================
    # 6. Plot
    # ===========================

    plt.figure(figsize=(12, 6))

    plt.bar(
        pi.keys(),
        pi.values(),
        width=1.0,
        edgecolor="black",
        alpha=0.7
    )

    plt.title("Hours/week distribution (0–100)", fontsize=16)
    plt.xlabel("Hours/week", fontsize=14)
    plt.ylabel("Empirical Probability", fontsize=14)

    plt.xlim(-1, 101)
    plt.xticks(np.arange(0, 101, 10))

    plt.figtext(
        0.15, 0.02,
        f"N = {N} | sum pi = {sum(pi.values()):.6f}",
        fontsize=10,
        bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5}
    )

    plt.tight_layout()
    plt.show()
