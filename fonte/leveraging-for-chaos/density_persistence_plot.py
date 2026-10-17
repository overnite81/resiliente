# Functional of Order-Statistic Density Persistence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.signal import find_peaks

# --------------------------------------------------
# Densidade local O(m) usando two-pointers
# --------------------------------------------------

def local_density_sorted(x, eps):
    """
    x deve estar ordenado.
    Retorna densidade local para cada posição.
    """

    n = len(x)
    density = np.empty(n, dtype=np.int32)

    left = 0
    right = 0

    for i in range(n):

        while x[i] - x[left] > eps:
            left += 1

        while right < n and x[right] - x[i] <= eps:
            right += 1

        density[i] = right - left

    return density


# --------------------------------------------------
# Histograma de persistência
# --------------------------------------------------

def persistence_histogram(samples, eps, top_k):
    """
    samples: array (N,m)

    retorna:
        H : histograma de persistência
    """

    N, m = samples.shape

    H = np.zeros(m, dtype=np.int64)

    for row in samples:

        xs = np.sort(row)

        dens = local_density_sorted(xs, eps)

        winners = np.argsort(dens)[::-1][:top_k]

        H[winners] += 1

    return H


# --------------------------------------------------
# Estatística Qui-Quadrado
# --------------------------------------------------

def persistence_chi2(H, top_k, n_samples):
    """
    Teste:
    H0: todos os ranks possuem mesma chance.
    """

    m = len(H)

    total = n_samples * top_k

    expected = total / m

    stat = np.sum((H - expected) ** 2 / expected)

    df = m - 1

    p = 1 - chi2.cdf(stat, df)

    return stat, p


# --------------------------------------------------
# Jensen-Shannon para medir estrutura
# --------------------------------------------------

def jensen_shannon_uniform(H):

    P = H.astype(float)
    P /= P.sum()

    m = len(H)

    U = np.ones(m) / m

    M = 0.5 * (P + U)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    js = 0.5 * kl(P, M) + 0.5 * kl(U, M)

    return js


# --------------------------------------------------
# Teste Monte Carlo correto
# --------------------------------------------------

def persistence_permutation_test(
        H_obs,
        n_samples,
        top_k,
        n_mc=5000):
    """
    H0:
        ranks equiprováveis
    """

    m = len(H_obs)

    total = n_samples * top_k

    expected = total / m

    stat_obs = np.sum(
        (H_obs - expected) ** 2 / expected
    )

    null_stats = np.empty(n_mc)

    for b in range(n_mc):

        H = np.zeros(m, dtype=np.int64)

        for _ in range(n_samples):

            winners = np.random.choice(
                m,
                size=top_k,
                replace=False
            )

            H[winners] += 1

        null_stats[b] = np.sum(
            (H - expected) ** 2 / expected
        )

    p_value = (1 + np.sum(null_stats >= stat_obs)) / (n_mc + 1)

    return stat_obs, p_value, null_stats


def run_experiment(samples, eps, top_k):

    H = persistence_histogram(
        samples,
        eps,
        top_k
    )

    js = jensen_shannon_uniform(H)

    stat, p, _ = persistence_permutation_test(
        H,
        n_samples=len(samples),
        top_k=top_k,
        n_mc=2000
    )

    return {
        "JS": js,
        "p": p,
        "H": H
    }

# --------------------------------------------------
# Exemplo
# --------------------------------------------------

if __name__ == "__main__":

    np.random.seed(123)

    # ----------------------------------------------------
    # Experimento de resolução modal
    # ----------------------------------------------------

    #distances = [0, 1, 2, 4, 8, 16, 32]
    #distances = [
    #    0,
    #    0.25,
    #    0.50,
    #    0.75,
    #    1.00,
    #    1.25,
    #    1.50,
    #    1.75,
    #    2.00
    #]
    distances = np.arange(
        0.7,
        1.4,
        0.05
    )
    js_values = []
    p_values = []
    histograms = {}

    for d in distances:

        print(f"\nd = {d}")

        m = 256 * 2
        w1 = int(0.7 * m)
        w2 = m - w1
        
        samples = []

        for _ in range(5000):

            a = np.random.normal(
                loc=20,
                scale=0.3,
                size=w1
            )

            b = np.random.normal(
                loc=20 + d,
                scale=0.3,
                size=w2
            )

            row = np.concatenate([a, b])

            samples.append(row)

        samples = np.asarray(samples)

        res = run_experiment(
        samples,
        eps=0.5,
        top_k=20
        )

        histograms[d] = res["H"]

        js = res["JS"]
        p = res["p"]

        js_values.append(js)
        p_values.append(p)

        print(f"JS = {js:.6f}")
        print(f"p  = {p:.6g}")

    # ----------------------------------------------------
    # Resumo tabular
    # ----------------------------------------------------

    print("\nResumo:")
    print("-" * 40)

    for d, js, p in zip(
        distances,
        js_values,
        p_values
    ):
        print(
        f"d={d:>2} | "
        f"JS={js:.6f} | "
        f"p={p:.6g}"
        )

    # ----------------------------------------------------
    # Curva de resolução
    # ----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        distances,
        js_values,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Separação entre modos (d)"
    )

    plt.ylabel(
        "Rank Persistence Divergence (JS)"
    )

    plt.title(
        "Estrutura residual versus separação modal"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()



    fig, axes = plt.subplots(
        len(distances),
        1,
        figsize=(12, 14),
        sharex=True
    )

    for ax, d in zip(axes, distances):

        H = histograms[d]

        ax.plot(H)

        ax.set_ylabel(f"d={d}")

        ax.grid(True)

    axes[-1].set_xlabel("Rank")

    plt.suptitle(
        "Histogramas de Persistência"
    )

    plt.tight_layout()

    plt.show()



    n_peaks = []

    for d in distances:

        H = histograms[d]

        peaks, _ = find_peaks(
            H,
            prominence=np.max(H) * 0.05
        )

        #delta_rank = float("nan")
        delta_rank = 0

        if len(peaks) >= 2:

            peaks = sorted(
                peaks,
                key=lambda p: H[p],
                reverse=True
            )

            p1 = peaks[0]
            p2 = peaks[1]

            delta_rank = abs(p1 - p2)

        n_peaks.append(len(peaks))

        print(
            f"d={d} "
            f"picos={len(peaks)} "
            f"delta de ordem={delta_rank}"
        )

    plt.figure(figsize=(8,5))

    plt.plot(
        distances,
        n_peaks,
        "o-"
    )

    plt.xlabel(
        "Separação modal (d)"
    )

    plt.ylabel(
        "Número de picos"
    )

    plt.title(
        "Resolução modal do método"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


    plt.figure(figsize=(8,5))

    plt.plot(
        distances,
        delta_rank,
        "o-"
    )

    plt.xlabel("Separação física d")
    plt.ylabel("Separação dos picos Δrank")

    plt.grid(True)

    plt.show()
