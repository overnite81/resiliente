"""Teste da relação entre o peso da mistura e o centro do histograma H.

Mantém d=4 (modos gaussianos bem separados) e varia o peso do grupo A
entre 0.50 e 0.95. Cada peso usa 5 réplicas de 1000 amostras, totalizando
5000 amostras por condição.
"""

import numpy as np
import matplotlib.pyplot as plt


# Parâmetros do experimento
M = 256 * 2
DISTANCE = 4.0
SIGMA = 0.3
EPS = 0.5
TOP_K = 20
WEIGHTS = np.round(np.arange(0.50, 0.951, 0.05), 2)
N_REPLICATES = 5
SAMPLES_PER_REPLICATE = 1000
SEED = 123


def local_density_sorted(x, eps):
    """Calcula a densidade local de cada posição de um vetor ordenado."""
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


def persistence_histogram(samples, eps=EPS, top_k=TOP_K):
    """Conta a frequência com que cada rank entra nos top_k mais densos."""
    _, m = samples.shape
    histogram = np.zeros(m, dtype=np.int64)

    for row in samples:
        sorted_row = np.sort(row)
        density = local_density_sorted(sorted_row, eps)
        winners = np.argsort(density)[::-1][:top_k]
        histogram[winners] += 1

    return histogram


def histogram_center(histogram):
    """Retorna o centro de massa μ_H do histograma de persistência."""
    total = histogram.sum()
    if total == 0:
        return np.nan
    ranks = np.arange(len(histogram), dtype=float)
    return float(np.sum(ranks * histogram) / total)


def make_samples(rng, weight_a, n_samples):
    """Gera misturas com contagens fixas para os dois grupos."""
    n_a = int(round(weight_a * M))
    n_b = M - n_a
    samples = np.empty((n_samples, M), dtype=float)

    for i in range(n_samples):
        group_a = rng.normal(loc=20.0, scale=SIGMA, size=n_a)
        group_b = rng.normal(
            loc=20.0 + DISTANCE,
            scale=SIGMA,
            size=n_b,
        )
        samples[i] = np.concatenate((group_a, group_b))

    return samples, n_a, n_b


def run_experiment():
    rng = np.random.default_rng(SEED)
    observed_centers = []
    center_standard_errors = []
    widths = []

    print(f"d={DISTANCE:g}, sigma={SIGMA:g}, m={M}, eps={EPS:g}, top_k={TOP_K}")
    print(f"Réplicas por peso: {N_REPLICATES} × {SAMPLES_PER_REPLICATE} amostras")
    print("peso_A  n_A  mu_H    teoria      diferença   erro-padrão  sigma_H")
    print("-" * 76)

    for weight_a in WEIGHTS:
        combined_histogram = np.zeros(M, dtype=np.int64)
        replicate_centers = []
        n_a = n_b = 0

        for _ in range(N_REPLICATES):
            samples, n_a, n_b = make_samples(
                rng,
                weight_a,
                SAMPLES_PER_REPLICATE,
            )
            replicate_histogram = persistence_histogram(samples)
            combined_histogram += replicate_histogram
            replicate_centers.append(histogram_center(replicate_histogram))

        mu_h = histogram_center(combined_histogram)
        replicate_centers = np.asarray(replicate_centers)
        standard_error = np.std(replicate_centers, ddof=1) / np.sqrt(N_REPLICATES)
        ranks = np.arange(M, dtype=float)
        probabilities = combined_histogram / combined_histogram.sum()
        sigma_h = np.sqrt(np.sum(probabilities * (ranks - mu_h) ** 2))

        # Para w>0.5, compara com o centro aproximado do bloco majoritário.
        # Em 50/50, a simetria prevê o centro global m/2.
        theory = M / 2 if np.isclose(weight_a, 0.5) else M * weight_a / 2
        difference = mu_h - theory

        observed_centers.append(mu_h)
        center_standard_errors.append(standard_error)
        widths.append(sigma_h)

        print(
            f"{weight_a:5.2f}  {n_a:3d}  {mu_h:6.2f}  {theory:8.2f}  "
            f"{difference:10.2f}  {standard_error:11.2f}  {sigma_h:7.2f}"
        )

    observed_centers = np.asarray(observed_centers)
    center_standard_errors = np.asarray(center_standard_errors)
    widths = np.asarray(widths)
    theory_curve = np.where(
        np.isclose(WEIGHTS, 0.5),
        M / 2,
        M * WEIGHTS / 2,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].errorbar(
        WEIGHTS,
        observed_centers,
        yerr=center_standard_errors,
        fmt="o",
        capsize=4,
        label="μ_H observado (± erro-padrão entre réplicas)",
    )
    axes[0].plot(
        WEIGHTS,
        theory_curve,
        "--",
        label="referência: m/2 em 50/50; mw/2 para w>0.5",
    )
    axes[0].set_xlabel("Peso do grupo A (w)")
    axes[0].set_ylabel("Centro do histograma (μ_H)")
    axes[0].set_title("Centro observado e referência teórica")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(WEIGHTS, observed_centers - theory_curve, "o-")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Peso do grupo A (w)")
    axes[1].set_ylabel("μ_H observado − referência")
    axes[1].set_title("Desvio em relação à referência")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(WEIGHTS, widths, "o-", color="tab:orange")
    axes[2].set_xlabel("Peso do grupo A (w)")
    axes[2].set_ylabel("Largura do histograma (σ_H)")
    axes[2].set_title("Largura da persistência por peso")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Lei peso–centro da persistência para d={DISTANCE:g}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
