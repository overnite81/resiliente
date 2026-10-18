"""Experimento fatorial da persistência de ranks em misturas gaussianas.

Fatores variados:
- distância entre componentes;
- perfil de pesos e atribuição dos pesos às posições das médias;
- perfil de desvios-padrão por componente;
- tamanho do conjunto Top-K.

Registra métricas de H, proximidade do centro de H ao centro do bloco de ranks
previsto para a componente mais pesada e suporte dos picos entre réplicas.
Resultados detalhados são gravados em density_persistence_factorial.csv.
"""

import csv
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Dimensão e parâmetros da mistura
M = 512
DISTANCES = (4.0, 8.0, 16.0)
MEAN_ORIGIN = 20.0
EPS = 0.5
BASE_SIGMA = 0.3

# Perfis de pesos associados a identificadores fixos de componentes.
WEIGHT_PROFILES = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "near_tie": (0.36, 0.33, 0.31),
    "moderate_dominance": (0.50, 0.30, 0.20),
    "strong_dominance": (0.70, 0.20, 0.10),
}

# Cada perfil é associado aos mesmos identificadores de componentes usados nos pesos.
SIGMA_PROFILES = {
    "equal": (BASE_SIGMA, BASE_SIGMA, BASE_SIGMA),
    "component_0_narrow": (0.15, BASE_SIGMA, BASE_SIGMA),
    "component_1_narrow": (BASE_SIGMA, 0.15, BASE_SIGMA),
    "component_2_narrow": (BASE_SIGMA, BASE_SIGMA, 0.15),
}

TOP_K_VALUES = (5, 10, 20, 40, 80)
N_REPLICATES = 5
SAMPLES_PER_REPLICATE = 200
SMOOTHING_SIGMA = 5
PEAK_PROMINENCE_FRACTION = 0.05
PEAK_MATCH_TOLERANCE = 20
DOMINANCE_MARGIN = 0.05
SEED = 123
CSV_PATH = "density_persistence_factorial.csv"


COMPONENT_ORDERS = tuple(permutations(range(3)))


def integer_counts(weights, total=M):
    """Converte pesos em contagens inteiras cuja soma é exatamente total."""
    counts = [int(round(weight * total)) for weight in weights[:-1]]
    counts.append(total - sum(counts))
    return np.asarray(counts, dtype=int)


def block_centers(order, counts_by_component):
    """Retorna o centro em ranks 0..M-1 de cada bloco ordenado."""
    centers = {}
    start = 0
    for component in order:
        count = int(counts_by_component[component])
        centers[component] = start + (count - 1) / 2
        start += count
    return centers


def local_density_sorted(sorted_values, eps=EPS):
    """Calcula densidades locais usando buscas binárias vetorizadas por linha."""
    left = np.searchsorted(sorted_values, sorted_values - eps, side="left")
    right = np.searchsorted(sorted_values, sorted_values + eps, side="right")
    return right - left


def generate_batch(rng, distance, order, counts_by_component, sigmas, n_samples):
    """Gera uma matriz de amostras; as médias seguem a ordem especificada."""
    groups = []
    for rank_position, component in enumerate(order):
        mean = MEAN_ORIGIN + rank_position * distance
        groups.append(
            rng.normal(
                loc=mean,
                scale=sigmas[component],
                size=(n_samples, int(counts_by_component[component])),
            )
        )
    return np.concatenate(groups, axis=1)


def topk_histograms(samples, top_k_values):
    """Calcula vários histogramas Top-K a partir da mesma ordenação por amostra."""
    sorted_samples = np.sort(samples, axis=1)
    densities = np.empty_like(sorted_samples, dtype=np.int32)

    # searchsorted opera sobre vetores 1D; o loop é sobre o lote, não sobre ranks.
    for row_index, row in enumerate(sorted_samples):
        densities[row_index] = local_density_sorted(row)

    rank_order = np.argsort(densities, axis=1)[:, ::-1]
    max_k = max(top_k_values)
    winners = rank_order[:, :max_k]
    return {
        k: np.bincount(winners[:, :k].ravel(), minlength=samples.shape[1])
        for k in top_k_values
    }


def histogram_statistics(histogram):
    """Calcula μ_H e σ_H usando H normalizado como distribuição de ranks."""
    total = histogram.sum()
    if total == 0:
        return np.nan, np.nan

    ranks = np.arange(len(histogram), dtype=float)
    probabilities = histogram / total
    mu = float(np.sum(ranks * probabilities))
    sigma_h = float(np.sqrt(np.sum(probabilities * (ranks - mu) ** 2)))
    return mu, sigma_h


def detect_peaks(histogram):
    """Suaviza o histograma e detecta picos suficientemente proeminentes."""
    smoothed = gaussian_filter1d(histogram.astype(float), sigma=SMOOTHING_SIGMA)
    if smoothed.size == 0 or smoothed.max() <= 0:
        return smoothed, np.asarray([], dtype=int)

    peaks, _ = find_peaks(
        smoothed,
        prominence=smoothed.max() * PEAK_PROMINENCE_FRACTION,
        distance=2 * SMOOTHING_SIGMA,
    )
    return smoothed, peaks


def nearest_peak_distance(target, peaks):
    """Distância do alvo ao pico detectado mais próximo; NaN se não houver picos."""
    if len(peaks) == 0:
        return np.nan
    return float(np.min(np.abs(peaks - target)))


def encode(values):
    """Representação compacta de listas para a tabela CSV."""
    return ";".join(str(value) for value in values)


def run_factorial():
    rng = np.random.default_rng(SEED)
    rows = []
    total_conditions = (
        len(DISTANCES)
        * len(WEIGHT_PROFILES)
        * len(COMPONENT_ORDERS)
        * len(SIGMA_PROFILES)
    )
    print(
        f"Condições de mistura: {total_conditions}; "
        f"cada uma testará K={TOP_K_VALUES} em {N_REPLICATES} réplicas "
        f"de {SAMPLES_PER_REPLICATE} amostras."
    )

    for distance in DISTANCES:
        for weight_name, profile_weights in WEIGHT_PROFILES.items():
            for order in COMPONENT_ORDERS:
                weights_by_component = np.asarray(profile_weights, dtype=float)
                counts_by_component = integer_counts(weights_by_component)
                centers = block_centers(order, counts_by_component)
                sorted_weights = np.sort(weights_by_component)
                has_clear_dominance = (
                    sorted_weights[-1] - sorted_weights[-2] >= DOMINANCE_MARGIN
                )
                dominant_component = (
                    int(np.argmax(weights_by_component)) if has_clear_dominance else -1
                )
                dominant_center = (
                    centers[dominant_component] if has_clear_dominance else np.nan
                )

                for sigma_name, sigmas in SIGMA_PROFILES.items():
                    combined = {
                        k: np.zeros(M, dtype=np.int64) for k in TOP_K_VALUES
                    }
                    replicate_histograms = {
                        k: [] for k in TOP_K_VALUES
                    }

                    for _ in range(N_REPLICATES):
                        replicate_total = {
                            k: np.zeros(M, dtype=np.int64) for k in TOP_K_VALUES
                        }
                        remaining = SAMPLES_PER_REPLICATE
                        while remaining > 0:
                            batch_size = min(remaining, 200)
                            samples = generate_batch(
                                rng,
                                distance,
                                order,
                                counts_by_component,
                                sigmas,
                                batch_size,
                            )
                            batch_histograms = topk_histograms(samples, TOP_K_VALUES)
                            for k in TOP_K_VALUES:
                                replicate_total[k] += batch_histograms[k]
                            remaining -= batch_size

                        for k in TOP_K_VALUES:
                            combined[k] += replicate_total[k]
                            replicate_histograms[k].append(replicate_total[k])

                    for k in TOP_K_VALUES:
                        histogram = combined[k]
                        mu_h, sigma_h = histogram_statistics(histogram)
                        replicate_mus = np.asarray(
                            [histogram_statistics(h)[0] for h in replicate_histograms[k]]
                        )
                        mu_se = float(
                            np.std(replicate_mus, ddof=1) / np.sqrt(N_REPLICATES)
                        )
                        _, peaks = detect_peaks(histogram)
                        replicate_peaks = [
                            detect_peaks(rep_hist)[1]
                            for rep_hist in replicate_histograms[k]
                        ]
                        peak_support = [
                            sum(
                                np.any(np.abs(rep_peaks - center) <= PEAK_MATCH_TOLERANCE)
                                for rep_peaks in replicate_peaks
                            )
                            for center in centers.values()
                        ]
                        component_centers_in_order = [centers[c] for c in order]
                        component_weights_in_order = [
                            float(weights_by_component[c]) for c in order
                        ]
                        component_sigmas_in_order = [float(sigmas[c]) for c in order]

                        row = {
                            "distance": distance,
                            "weight_profile": weight_name,
                            "sigma_profile": sigma_name,
                            "component_order_low_to_high_mean": "-".join(
                                str(component) for component in order
                            ),
                            "weights_by_component": encode(
                                [f"{w:.6f}" for w in weights_by_component]
                            ),
                            "sigmas_by_component": encode(
                                [f"{s:.6f}" for s in sigmas]
                            ),
                            "weights_low_to_high_mean": encode(
                                [f"{w:.6f}" for w in component_weights_in_order]
                            ),
                            "sigmas_low_to_high_mean": encode(
                                [f"{s:.6f}" for s in component_sigmas_in_order]
                            ),
                            "counts_by_component": encode(counts_by_component.tolist()),
                            "block_centers_low_to_high_mean": encode(
                                [f"{center:.2f}" for center in component_centers_in_order]
                            ),
                            "top_k": k,
                            "top_k_over_m": k / M,
                            "n_replicates": N_REPLICATES,
                            "samples_per_replicate": SAMPLES_PER_REPLICATE,
                            "mu_h": mu_h,
                            "mu_h_standard_error": mu_se,
                            "mu_h_over_m": mu_h / M,
                            "sigma_h": sigma_h,
                            "dominant_component": dominant_component,
                            "dominant_weight": (
                                float(weights_by_component[dominant_component])
                                if has_clear_dominance
                                else np.nan
                            ),
                            "dominant_block_center": dominant_center,
                            "dominant_block_center_over_m": dominant_center / M,
                            "absolute_error_to_dominant_block_center": abs(
                                mu_h - dominant_center
                            ),
                            "peak_positions": encode(peaks.tolist()),
                            "peak_support_by_low_to_high_component": encode(
                                peak_support
                            ),
                            "nearest_peak_distance_to_dominant_block":
                                nearest_peak_distance(dominant_center, peaks),
                        }
                        rows.append(row)

        print(f"Concluído d={distance:g}")

    write_csv(rows)
    print_summary(rows)
    plot_summary(rows)


def write_csv(rows):
    """Grava uma linha por combinação de fatores e valor de K."""
    if not rows:
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDados fatoriais gravados em {CSV_PATH}")


def print_summary(rows):
    """Resume o erro médio do centro por d e K, agregando os demais fatores."""
    print("\nErro absoluto médio de μ_H até o centro do bloco dominante (ranks)")
    print("d      K       erro médio   erro mediano   N")
    print("-" * 48)
    for distance in DISTANCES:
        for k in TOP_K_VALUES:
            selected = [
                row["absolute_error_to_dominant_block_center"]
                for row in rows
                if row["distance"] == distance
                and row["top_k"] == k
                and np.isfinite(row["absolute_error_to_dominant_block_center"])
            ]
            if selected:
                mean_error = np.mean(selected)
                median_error = np.median(selected)
            else:
                mean_error = median_error = np.nan
            print(
                f"{distance:<6g} {k:<7d} {mean_error:11.2f} "
                f"{median_error:13.2f} {len(selected):5d}"
            )


def plot_summary(rows):
    """Gera gráficos de concordância, erro, largura e detecção de picos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_scatter, ax_error, ax_width, ax_support = axes.flat
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(TOP_K_VALUES)))

    for color, k in zip(colors, TOP_K_VALUES):
        subset = [
            row for row in rows
            if row["top_k"] == k and row["dominant_component"] >= 0
        ]
        ax_scatter.scatter(
            [row["dominant_block_center_over_m"] for row in subset],
            [row["mu_h_over_m"] for row in subset],
            s=13,
            alpha=0.45,
            color=color,
            label=f"K={k}",
        )
    ax_scatter.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_scatter.set_xlabel("Centro previsto do bloco dominante / m")
    ax_scatter.set_ylabel("μ_H / m observado")
    ax_scatter.set_title("Centro observado versus centro do bloco")
    ax_scatter.legend(title="Top-K", fontsize="small")
    ax_scatter.grid(True, alpha=0.25)

    for distance in DISTANCES:
        means = []
        for k in TOP_K_VALUES:
            selected = [
                row["absolute_error_to_dominant_block_center"]
                for row in rows
                if row["distance"] == distance
                and row["top_k"] == k
                and np.isfinite(row["absolute_error_to_dominant_block_center"])
            ]
            means.append(np.mean(selected) if selected else np.nan)
        ax_error.plot(TOP_K_VALUES, means, "o-", label=f"d={distance:g}")
    ax_error.set_xlabel("K")
    ax_error.set_ylabel("Erro absoluto médio (ranks)")
    ax_error.set_title("Erro de μ_H para o bloco dominante")
    ax_error.legend()
    ax_error.grid(True, alpha=0.25)

    for weight_name in WEIGHT_PROFILES:
        means = []
        for k in TOP_K_VALUES:
            selected = [
                row["sigma_h"]
                for row in rows
                if row["weight_profile"] == weight_name and row["top_k"] == k
            ]
            means.append(np.mean(selected))
        ax_width.plot(TOP_K_VALUES, means, "o-", label=weight_name)
    ax_width.set_xlabel("K")
    ax_width.set_ylabel("σ_H (ranks)")
    ax_width.set_title("Largura média por perfil de pesos")
    ax_width.legend(fontsize="small")
    ax_width.grid(True, alpha=0.25)

    for distance in DISTANCES:
        means = []
        for k in TOP_K_VALUES:
            supports = []
            for row in rows:
                if (
                    row["distance"] == distance
                    and row["top_k"] == k
                    and row["dominant_component"] >= 0
                ):
                    support = [
                        int(value)
                        for value in row[
                            "peak_support_by_low_to_high_component"
                        ].split(";")
                    ]
                    dominant_position = int(
                        row["component_order_low_to_high_mean"].split("-").index(
                            str(row["dominant_component"])
                        )
                    )
                    supports.append(support[dominant_position] / N_REPLICATES)
            means.append(np.mean(supports))
        ax_support.plot(TOP_K_VALUES, means, "o-", label=f"d={distance:g}")
    ax_support.set_xlabel("K")
    ax_support.set_ylabel("Fração de réplicas com pico perto do bloco dominante")
    ax_support.set_ylim(-0.05, 1.05)
    ax_support.set_title(f"Reprodutibilidade de picos (tolerância ±{PEAK_MATCH_TOLERANCE})")
    ax_support.legend()
    ax_support.grid(True, alpha=0.25)

    fig.suptitle("Experimento fatorial de persistência de ranks")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_factorial()
