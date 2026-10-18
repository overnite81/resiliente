"""Análise da persistência de ranks com métricas globais do histograma.

Além da divergência JS, calcula centro de massa, largura e entropia.
A detecção de picos é feita sobre histogramas suavizados e serve apenas
como diagnóstico complementar, não como medida principal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Parâmetros do experimento
M = 256 * 2
N_SAMPLES = 5000
SIGMA = 0.3
EPS = 0.5
TOP_K = 20
SMOOTHING_SIGMA = 5
SEED = 123


def local_density_sorted(x, eps):
    """Calcula a densidade local de cada rank em um vetor ordenado."""
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


def persistence_histogram(samples, eps, top_k):
    """Conta quantas vezes cada rank está entre os top_k mais densos."""
    _, m = samples.shape
    histogram = np.zeros(m, dtype=np.int64)

    for row in samples:
        sorted_row = np.sort(row)
        density = local_density_sorted(sorted_row, eps)
        winners = np.argsort(density)[::-1][:top_k]
        histogram[winners] += 1

    return histogram


def jensen_shannon_uniform(histogram):
    """Divergência Jensen-Shannon, em bits, entre H normalizado e uniforme."""
    total = histogram.sum()
    if total == 0:
        return 0.0

    p = histogram.astype(float) / total
    uniform = np.full(len(histogram), 1.0 / len(histogram))
    midpoint = 0.5 * (p + uniform)

    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(
        uniform, midpoint
    )


def histogram_metrics(histogram):
    """Calcula JS, centro de massa, desvio-padrão e entropia de H."""
    total = histogram.sum()
    if total == 0:
        return {"JS": 0.0, "mu": np.nan, "sigma": np.nan, "entropy": 0.0}

    ranks = np.arange(len(histogram), dtype=float)
    probabilities = histogram / total
    mu = np.sum(ranks * probabilities)
    width = np.sqrt(np.sum(probabilities * (ranks - mu) ** 2))
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log2(probabilities[positive]))

    return {
        "JS": jensen_shannon_uniform(histogram),
        "mu": float(mu),
        "sigma": float(width),
        "entropy": float(entropy),
    }


def smoothed_peak_positions(histogram, sigma=SMOOTHING_SIGMA):
    """Retorna até dois picos mais altos após suavização gaussiana."""
    smoothed = gaussian_filter1d(histogram.astype(float), sigma=sigma)
    if smoothed.size == 0 or np.max(smoothed) <= 0:
        return smoothed, []

    peaks, _ = find_peaks(
        smoothed,
        prominence=np.max(smoothed) * 0.05,
    )
    strongest = sorted(peaks, key=lambda rank: smoothed[rank], reverse=True)[:2]
    return smoothed, sorted(int(rank) for rank in strongest)


def make_mixture_samples(rng, distance, weight_a, n_samples=N_SAMPLES):
    """Gera amostras de dois blocos gaussianos com tamanhos exatos."""
    n_a = int(round(weight_a * M))
    n_b = M - n_a
    samples = np.empty((n_samples, M), dtype=float)

    for i in range(n_samples):
        group_a = rng.normal(loc=20.0, scale=SIGMA, size=n_a)
        group_b = rng.normal(loc=20.0 + distance, scale=SIGMA, size=n_b)
        samples[i] = np.concatenate((group_a, group_b))

    return samples


def analyze_setting(rng, distance, weight_a):
    samples = make_mixture_samples(rng, distance, weight_a)
    histogram = persistence_histogram(samples, EPS, TOP_K)
    metrics = histogram_metrics(histogram)
    smoothed, peak_positions = smoothed_peak_positions(histogram)

    return histogram, smoothed, metrics, peak_positions


def print_result(label, metrics, peaks):
    peak_text = ", ".join(map(str, peaks)) if peaks else "nenhum"
    print(
        f"{label} | JS={metrics['JS']:.6f} | "
        f"mu_H={metrics['mu']:.2f} | sigma_H={metrics['sigma']:.2f} | "
        f"entropia={metrics['entropy']:.4f} bits | "
        f"picos suavizados={peak_text}"
    )


def run_factorial_experiment(rng):
    """Testa todas as combinações entre distâncias e pesos da mistura."""
    distances = np.round(np.arange(0.7, 1.4, 0.05), 2)
    weights = (0.5, 0.7, 0.8, 0.9)
    metric_names = ("JS", "mu", "sigma", "entropy")
    metric_titles = {
        "JS": "Divergência JS",
        "mu": "Centro de massa (μ_H)",
        "sigma": "Largura (σ_H)",
        "entropy": "Entropia (bits)",
    }
    results = {weight: [] for weight in weights}

    print(
        "Experimento fatorial: "
        f"{len(distances)} valores de d × {len(weights)} pesos "
        f"= {len(distances) * len(weights)} combinações"
    )
    for weight_a in weights:
        print(f"\nPesos {weight_a:.0%}/{1 - weight_a:.0%}")
        for distance in distances:
            _, _, metrics, peaks = analyze_setting(rng, distance, weight_a)
            results[weight_a].append(metrics)
            print_result(f"d={distance:.2f}", metrics, peaks)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, metric_name in zip(axes.flat, metric_names):
        for weight_a in weights:
            values = [row[metric_name] for row in results[weight_a]]
            ax.plot(
                distances,
                values,
                marker="o",
                label=f"{weight_a:.0%}/{1 - weight_a:.0%}",
            )
        ax.set_title(metric_titles[metric_name])
        ax.set_xlabel("Separação entre modos (d)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Pesos")

    fig.suptitle("Experimento fatorial: métricas por distância e pesos")
    fig.tight_layout()


def main():
    rng = np.random.default_rng(SEED)
    run_factorial_experiment(rng)
    plt.show()


if __name__ == "__main__":
    main()
