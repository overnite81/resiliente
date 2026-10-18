"""Teste da persistência de ranks em uma mistura de três gaussianas.

Testa se os picos do histograma aparecem nos centros dos blocos de ranks
previstos pelas massas acumuladas. Repetir em várias separações permite
verificar se a localização depende principalmente dos quantis, e não de d.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Parâmetros experimentais
M = 256 * 2
WEIGHTS = (0.5, 0.3, 0.2)
DISTANCES = (4.0, 8.0, 16.0)
MEAN_ORIGIN = 20.0
COMPONENT_SIGMA = 0.3
EPS = 0.5
TOP_K = 20
N_REPLICATES = 5
SAMPLES_PER_REPLICATE = 1000
SMOOTHING_SIGMA = 5
PEAK_PROMINENCE_FRACTION = 0.05
PEAK_MATCH_TOLERANCE = 20
SEED = 123


def component_counts(m=M, weights=WEIGHTS):
    """Converte pesos em contagens inteiras que somam exatamente m."""
    counts = [int(round(weight * m)) for weight in weights[:-1]]
    counts.append(m - sum(counts))
    return np.asarray(counts, dtype=int)


def expected_rank_centers(counts):
    """Centro de cada bloco contíguo de ranks, usando índices 0..m-1."""
    starts = np.cumsum(np.r_[0, counts[:-1]])
    ends = starts + counts - 1
    centers = (starts + ends) / 2
    return centers, centers / np.sum(counts)


def local_density_sorted(x, eps):
    """Calcula densidade local para cada posição de um vetor ordenado."""
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


def persistence_histogram(samples):
    """Conta a frequência com que cada rank pertence aos top_k mais densos."""
    histogram = np.zeros(samples.shape[1], dtype=np.int64)
    for row in samples:
        sorted_row = np.sort(row)
        density = local_density_sorted(sorted_row, EPS)
        winners = np.argsort(density)[::-1][:TOP_K]
        histogram[winners] += 1
    return histogram


def smooth_and_find_peaks(histogram):
    """Suaviza H e detecta picos com proeminência relativa ao máximo."""
    smoothed = gaussian_filter1d(histogram.astype(float), sigma=SMOOTHING_SIGMA)
    if smoothed.size == 0 or smoothed.max() <= 0:
        return smoothed, np.asarray([], dtype=int)

    peaks, _ = find_peaks(
        smoothed,
        prominence=smoothed.max() * PEAK_PROMINENCE_FRACTION,
        distance=max(1, int(round(2 * SMOOTHING_SIGMA))),
    )
    return smoothed, peaks


def generate_samples(rng, distance, counts, n_samples):
    """Gera linhas com três componentes ordenadas por suas médias."""
    samples = np.empty((n_samples, M), dtype=float)
    means = MEAN_ORIGIN + distance * np.arange(len(counts))

    for sample_index in range(n_samples):
        groups = [
            rng.normal(loc=mean, scale=COMPONENT_SIGMA, size=count)
            for mean, count in zip(means, counts)
        ]
        samples[sample_index] = np.concatenate(groups)

    return samples


def nearest_peak(target, peaks):
    """Retorna o pico mais próximo do alvo, ou None se não houver picos."""
    if len(peaks) == 0:
        return None, None
    closest = int(peaks[np.argmin(np.abs(peaks - target))])
    return closest, abs(closest - target)


def run_experiment():
    rng = np.random.default_rng(SEED)
    counts = component_counts()
    expected_centers, expected_quantiles = expected_rank_centers(counts)

    print("Mistura de três componentes")
    print(f"m={M}; pesos nominais={WEIGHTS}; contagens={counts.tolist()}")
    print(
        "Centros teóricos dos blocos: "
        + ", ".join(f"rank {center:.1f} (q={quantile:.4f})"
                      for center, quantile in zip(expected_centers, expected_quantiles))
    )
    print(
        f"Amostras por condição: {N_REPLICATES} × {SAMPLES_PER_REPLICATE}; "
        f"eps={EPS}; top_k={TOP_K}; suavização sigma={SMOOTHING_SIGMA}"
    )

    results = []
    for distance in DISTANCES:
        combined_histogram = np.zeros(M, dtype=np.int64)
        replicate_peaks = []

        for _ in range(N_REPLICATES):
            samples = generate_samples(
                rng,
                distance,
                counts,
                SAMPLES_PER_REPLICATE,
            )
            replicate_histogram = persistence_histogram(samples)
            combined_histogram += replicate_histogram
            _, peaks = smooth_and_find_peaks(replicate_histogram)
            replicate_peaks.append(peaks)

        smoothed, peaks = smooth_and_find_peaks(combined_histogram)
        normalized = smoothed / smoothed.sum() if smoothed.sum() else smoothed
        support = [
            sum(
                np.any(np.abs(replicate - center) <= PEAK_MATCH_TOLERANCE)
                for replicate in replicate_peaks
            )
            for center in expected_centers
        ]

        results.append(
            {
                "distance": distance,
                "histogram": combined_histogram,
                "smoothed": normalized,
                "peaks": peaks,
                "support": support,
            }
        )

        print(f"\nd={distance:g}; picos detectados={peaks.tolist()}")
        for index, (center, quantile, replicate_support) in enumerate(
            zip(expected_centers, expected_quantiles, support), start=1
        ):
            peak, error = nearest_peak(center, peaks)
            if peak is None:
                peak_report = "nenhum pico detectado"
            else:
                peak_report = f"pico mais próximo={peak} (erro={error:.1f} ranks)"
            print(
                f"  componente {index}: centro esperado={center:.1f} "
                f"(q={quantile:.4f}); {peak_report}; "
                f"presença em réplicas dentro de ±{PEAK_MATCH_TOLERANCE}="
                f"{replicate_support}/{N_REPLICATES}"
            )

    plot_results(results, expected_centers)


def plot_results(results, expected_centers):
    """Mostra histogramas suavizados para cada d e alvos teóricos."""
    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(12, 3.5 * len(results)),
        sharex=True,
        squeeze=False,
    )
    component_colors = ("tab:blue", "tab:orange", "tab:green")

    for ax, result in zip(axes[:, 0], results):
        ranks = np.arange(M)
        ax.plot(ranks, result["smoothed"], color="black", linewidth=1.5,
                label="H suavizado e normalizado")
        if len(result["peaks"]):
            ax.scatter(
                result["peaks"],
                result["smoothed"][result["peaks"]],
                color="red",
                marker="x",
                zorder=3,
                label="picos detectados",
            )
        for index, (center, color) in enumerate(
            zip(expected_centers, component_colors), start=1
        ):
            ax.axvline(
                center,
                color=color,
                linestyle="--",
                alpha=0.8,
                label=f"centro teórico {index}: {center:.1f}",
            )
        ax.set_ylabel("Probabilidade")
        ax.set_title(f"Separação d={result['distance']:g}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize="small", ncol=2)

    axes[-1, 0].set_xlabel("Rank (índice 0 a m−1)")
    fig.suptitle("Persistência de ranks em uma mistura gaussiana de três componentes")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
