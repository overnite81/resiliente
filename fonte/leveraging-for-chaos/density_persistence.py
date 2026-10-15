# Functional of Order-Statistic Density Persistence

import numpy as np
from scipy.stats import chi2


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


# --------------------------------------------------
# Exemplo
# --------------------------------------------------

if __name__ == "__main__":

    np.random.seed(123)

    N = 5000
    m = 512

    samples = []

    for _ in range(N):

        noise = np.random.uniform(
            0,
            40,
            size=int(0.85 * m)
        )

        signal = np.random.normal(
            20,
            0.3,
            size=m - len(noise)
        )

        row = np.concatenate([noise, signal])

        samples.append(row)

    samples = np.asarray(samples)

    eps = 0.5
    top_k = 20

    H = persistence_histogram(
        samples,
        eps,
        top_k
    )

    js = jensen_shannon_uniform(H)

    chi2_stat, chi2_p = persistence_chi2(
        H,
        top_k,
        N
    )

    mc_stat, mc_p, _ = persistence_permutation_test(
        H,
        N,
        top_k,
        n_mc=2000
    )

    print("JS =", js)
    print("Chi2 =", chi2_stat)
    print("Chi2 p-value =", chi2_p)
    print("Monte Carlo p-value =", mc_p)