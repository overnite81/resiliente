"""Validação fatorial da persistência e da massa de H por bloco de ranks.

Cruza distâncias, pesos, permutações das médias, perfis de variâncias,
valores de K e políticas para desempatar ranks com a mesma densidade local.
Para cada bloco, mede a fração da massa de H, o centro condicional, a presença
reprodutível de picos e a competição entre blocos.

Grava resultados em density_persistence_block_validation.csv.
"""

import csv
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Parâmetros principais
M = 512
DISTANCES = (4.0, 8.0, 16.0)
MEAN_ORIGIN = 20.0
EPS = 0.5
BASE_SIGMA = 0.3

WEIGHT_PROFILES = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "near_tie": (0.36, 0.33, 0.31),
    "moderate_dominance": (0.50, 0.30, 0.20),
    "strong_dominance": (0.70, 0.20, 0.10),
}

SIGMA_PROFILES = {
    "equal": (BASE_SIGMA, BASE_SIGMA, BASE_SIGMA),
    "component_0_narrow": (0.15, BASE_SIGMA, BASE_SIGMA),
    "component_1_narrow": (BASE_SIGMA, 0.15, BASE_SIGMA),
    "component_2_narrow": (BASE_SIGMA, BASE_SIGMA, 0.15),
}

TOP_K_VALUES = (5, 10, 20, 40, 80)
TIE_POLICIES = ("random", "reverse_argsort")
COMPONENT_ORDERS = tuple(permutations(range(3)))
N_REPLICATES = 5
SAMPLES_PER_REPLICATE = 200
BATCH_SIZE = 200
SMOOTHING_SIGMA = 5
PEAK_PROMINENCE_FRACTION = 0.05
PEAK_CENTER_TOLERANCE = 20
DOMINANCE_MARGIN = 0.05
SEED = 123
CSV_PATH = "density_persistence_block_validation.csv"


def integer_counts(weights, total=M):
    """Converte os pesos em contagens inteiras que somam exatamente total."""
    counts = [int(round(weight * total)) for weight in weights[:-1]]
    counts.append(total - sum(counts))
    return np.asarray(counts, dtype=int)


def make_blocks(order, counts_by_component):
    """Retorna início, fim inclusivo e centro de cada componente nos ranks."""
    blocks = {}
    start = 0
    for component in order:
        count = int(counts_by_component[component])
        end = start + count - 1
        blocks[component] = {
            "start": start,
            "end": end,
            "center": (start + end) / 2,
            "count": count,
        }
        start = end + 1
    return blocks


def generate_batch(rng, distance, order, counts_by_component, sigmas, n_samples):
    """Gera uma matriz de misturas com médias crescentes segundo order."""
    groups = []
    for position, component in enumerate(order):
        groups.append(
            rng.normal(
                loc=MEAN_ORIGIN + position * distance,
                scale=sigmas[component],
                size=(n_samples, int(counts_by_component[component])),
            )
        )
    return np.concatenate(groups, axis=1)


def local_density(sorted_values):
    """Conta pontos a até EPS de cada estatística de ordem, inclusive extremos."""
    left = np.searchsorted(sorted_values, sorted_values - EPS, side="left")
    right = np.searchsorted(sorted_values, sorted_values + EPS, side="right")
    return right - left


def rank_orders(samples, rng):
    """Calcula a ordenação de densidades para as duas políticas de desempate."""
    sorted_samples = np.sort(samples, axis=1)
    densities = np.empty(sorted_samples.shape, dtype=np.int32)
    for row_index, row in enumerate(sorted_samples):
        densities[row_index] = local_density(row)

    # Reproduz o padrão argsort(...)[::-1] usado nos scripts anteriores.
    reverse_argsort = np.argsort(densities, axis=1)[:, ::-1]

    # Ruído contínuo apenas desempata contagens iguais, sem inverter contagens
    # inteiras diferentes. Usa a mesma amostra para comparação pareada.
    random_keys = rng.random(densities.shape)
    random_ties = np.lexsort((random_keys, -densities), axis=1)
    return {
        "random": random_ties,
        "reverse_argsort": reverse_argsort,
    }


def batch_histograms(samples, rng):
    """Produz H para todos os K e políticas de desempate do lote."""
    orders = rank_orders(samples, rng)
    output = {}
    for policy, rank_order in orders.items():
        max_k = max(TOP_K_VALUES)
        winners = rank_order[:, :max_k]
        output[policy] = {
            k: np.bincount(winners[:, :k].ravel(), minlength=M)
            for k in TOP_K_VALUES
        }
    return output


def histogram_statistics(histogram):
    """Retorna μ_H e σ_H tratando H normalizado como distribuição de ranks."""
    total = histogram.sum()
    if total == 0:
        return np.nan, np.nan
    ranks = np.arange(len(histogram), dtype=float)
    probabilities = histogram / total
    mu = float(np.sum(ranks * probabilities))
    sigma_h = float(np.sqrt(np.sum(probabilities * (ranks - mu) ** 2)))
    return mu, sigma_h


def block_statistics(histogram, block):
    """Massa de H no bloco e centro condicional à massa dentro do bloco."""
    total = histogram.sum()
    start, end = block["start"], block["end"]
    block_histogram = histogram[start : end + 1]
    block_total = int(block_histogram.sum())
    mass_fraction = block_total / total if total else np.nan

    if block_total:
        ranks = np.arange(start, end + 1, dtype=float)
        conditional_center = float(np.sum(ranks * block_histogram) / block_total)
    else:
        conditional_center = np.nan

    return mass_fraction, conditional_center


def find_histogram_peaks(histogram):
    """Suaviza H e retorna picos com proeminência relativa ao máximo global."""
    smoothed = gaussian_filter1d(histogram.astype(float), sigma=SMOOTHING_SIGMA)
    if not smoothed.size or smoothed.max() <= 0:
        return smoothed, np.asarray([], dtype=int)
    peaks, _ = find_peaks(
        smoothed,
        prominence=smoothed.max() * PEAK_PROMINENCE_FRACTION,
        distance=2 * SMOOTHING_SIGMA,
    )
    return smoothed, peaks


def peaks_near_center(peaks, center):
    """Indica se há pico detectado dentro da tolerância do centro previsto."""
    return bool(
        len(peaks)
        and np.any(np.abs(peaks - center) <= PEAK_CENTER_TOLERANCE)
    )


def unique_max_index(values):
    """Retorna índice do máximo se for único; caso contrário -1."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return -1
    maximum = np.nanmax(values)
    indices = np.flatnonzero(np.isclose(values, maximum, rtol=0, atol=1e-12))
    return int(indices[0]) if len(indices) == 1 else -1


def encode(values):
    """Serializa uma sequência curta para o CSV."""
    return ";".join(str(value) for value in values)


def run_experiment():
    rng = np.random.default_rng(SEED)
    output_rows = []
    n_mixtures = (
        len(DISTANCES)
        * len(WEIGHT_PROFILES)
        * len(COMPONENT_ORDERS)
        * len(SIGMA_PROFILES)
    )
    print(
        f"Misturas: {n_mixtures}; políticas de empate: {TIE_POLICIES}; "
        f"K={TOP_K_VALUES}; amostras/condição="
        f"{N_REPLICATES * SAMPLES_PER_REPLICATE}."
    )

    for distance in DISTANCES:
        for weight_name, weights in WEIGHT_PROFILES.items():
            weights_by_component = np.asarray(weights, dtype=float)
            counts = integer_counts(weights_by_component)
            weight_order = np.argsort(weights_by_component)[::-1]
            sorted_weights = np.sort(weights_by_component)
            has_clear_dominance = (
                sorted_weights[-1] - sorted_weights[-2] >= DOMINANCE_MARGIN
            )
            dominant_component = int(weight_order[0]) if has_clear_dominance else -1

            for order in COMPONENT_ORDERS:
                blocks = make_blocks(order, counts)
                for sigma_name, sigmas in SIGMA_PROFILES.items():
                    combined = {
                        policy: {k: np.zeros(M, dtype=np.int64) for k in TOP_K_VALUES}
                        for policy in TIE_POLICIES
                    }
                    replicate_histograms = {
                        policy: {k: [] for k in TOP_K_VALUES}
                        for policy in TIE_POLICIES
                    }

                    for _ in range(N_REPLICATES):
                        replicate_totals = {
                            policy: {k: np.zeros(M, dtype=np.int64) for k in TOP_K_VALUES}
                            for policy in TIE_POLICIES
                        }
                        remaining = SAMPLES_PER_REPLICATE
                        while remaining:
                            batch_size = min(remaining, BATCH_SIZE)
                            samples = generate_batch(
                                rng,
                                distance,
                                order,
                                counts,
                                sigmas,
                                batch_size,
                            )
                            batch_results = batch_histograms(samples, rng)
                            for policy in TIE_POLICIES:
                                for k in TOP_K_VALUES:
                                    replicate_totals[policy][k] += batch_results[policy][k]
                            remaining -= batch_size

                        for policy in TIE_POLICIES:
                            for k in TOP_K_VALUES:
                                replicate_histograms[policy][k].append(
                                    replicate_totals[policy][k]
                                )
                                combined[policy][k] += replicate_totals[policy][k]

                    for policy in TIE_POLICIES:
                        for k in TOP_K_VALUES:
                            histogram = combined[policy][k]
                            mu_h, sigma_h = histogram_statistics(histogram)
                            _, peaks = find_histogram_peaks(histogram)
                            ordered_components = list(order)
                            masses = {}
                            conditional_centers = {}
                            peak_in_block = {}
                            replicate_peak_support = {}
                            replicate_winning_blocks = []

                            for component in ordered_components:
                                mass, conditional_center = block_statistics(
                                    histogram, blocks[component]
                                )
                                masses[component] = mass
                                conditional_centers[component] = conditional_center
                                peak_in_block[component] = bool(
                                    np.any(
                                        (peaks >= blocks[component]["start"])
                                        & (peaks <= blocks[component]["end"])
                                    )
                                )
                                rep_support = 0
                                for rep_histogram in replicate_histograms[policy][k]:
                                    _, rep_peaks = find_histogram_peaks(rep_histogram)
                                    if peaks_near_center(
                                        rep_peaks, blocks[component]["center"]
                                    ):
                                        rep_support += 1
                                replicate_peak_support[component] = rep_support

                            for rep_histogram in replicate_histograms[policy][k]:
                                replicate_masses = [
                                    block_statistics(rep_histogram, blocks[c])[0]
                                    for c in ordered_components
                                ]
                                winner_position = unique_max_index(replicate_masses)
                                replicate_winning_blocks.append(
                                    ordered_components[winner_position]
                                    if winner_position >= 0
                                    else -1
                                )

                            aggregate_winner_position = unique_max_index(
                                [masses[c] for c in ordered_components]
                            )
                            aggregate_winner = (
                                ordered_components[aggregate_winner_position]
                                if aggregate_winner_position >= 0
                                else -1
                            )
                            if dominant_component >= 0:
                                dominant_block = blocks[dominant_component]
                                dominant_center = dominant_block["center"]
                                mu_error = abs(mu_h - dominant_center)
                                dominant_mass = masses[dominant_component]
                                dominant_win_fraction = sum(
                                    component == dominant_component
                                    for component in replicate_winning_blocks
                                ) / N_REPLICATES
                            else:
                                dominant_block = None
                                dominant_center = np.nan
                                mu_error = np.nan
                                dominant_mass = np.nan
                                dominant_win_fraction = np.nan

                            row_common = {
                                "distance": distance,
                                "weight_profile": weight_name,
                                "sigma_profile": sigma_name,
                                "component_order_low_to_high_mean": encode(order),
                                "tie_policy": policy,
                                "top_k": k,
                                "top_k_over_m": k / M,
                                "n_replicates": N_REPLICATES,
                                "samples_per_replicate": SAMPLES_PER_REPLICATE,
                                "weights_by_component": encode(
                                    [f"{x:.6f}" for x in weights_by_component]
                                ),
                                "sigmas_by_component": encode(
                                    [f"{x:.6f}" for x in sigmas]
                                ),
                                "counts_by_component": encode(counts.tolist()),
                                "mu_h": mu_h,
                                "mu_h_over_m": mu_h / M,
                                "sigma_h": sigma_h,
                                "global_peak_positions": encode(peaks.tolist()),
                                "dominant_component_by_weight": dominant_component,
                                "dominant_block_center": dominant_center,
                                "mu_error_to_weight_dominant_block": mu_error,
                                "weight_dominant_block_mass_fraction": dominant_mass,
                                "weight_dominant_block_peak_support": (
                                    replicate_peak_support.get(dominant_component, np.nan)
                                    / N_REPLICATES
                                    if dominant_component >= 0
                                    else np.nan
                                ),
                                "weight_dominant_winning_block_fraction": dominant_win_fraction,
                                "aggregate_max_persistence_component": aggregate_winner,
                                "replicate_winning_components": encode(
                                    replicate_winning_blocks
                                ),
                            }

                            for component in range(3):
                                block = blocks[component]
                                row = dict(row_common)
                                row.update(
                                    {
                                        "component": component,
                                        "component_weight": float(weights_by_component[component]),
                                        "component_sigma": float(sigmas[component]),
                                        "block_start_rank": block["start"],
                                        "block_end_rank": block["end"],
                                        "block_center_rank": block["center"],
                                        "block_center_quantile": block["center"] / M,
                                        "block_mass_fraction": masses[component],
                                        "block_conditional_mu_h": conditional_centers[component],
                                        "block_has_detected_peak": peak_in_block[component],
                                        "replicate_peak_support_near_center": (
                                            replicate_peak_support[component] / N_REPLICATES
                                        ),
                                        "replicate_win_fraction": sum(
                                            winner == component
                                            for winner in replicate_winning_blocks
                                        ) / N_REPLICATES,
                                    }
                                )
                                output_rows.append(row)

        print(f"Finalizadas as condições com d={distance:g}")

    write_csv(output_rows)
    print_summary(output_rows)
    plot_results(output_rows)


def write_csv(rows):
    """Grava métricas globais e uma linha de resultados para cada bloco."""
    if not rows:
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados detalhados gravados em {CSV_PATH}")


def print_summary(rows):
    """Resume centro e massa do bloco dominante para casos sem empate de peso."""
    print("\nResumo: comparação com o bloco da componente de maior peso")
    print("d    K   política          erro μ médio   massa no bloco   vitórias/rep")
    print("-" * 78)
    seen = set()
    for row in rows:
        key = (row["distance"], row["top_k"], row["tie_policy"])
        if key in seen:
            continue
        seen.add(key)
        matching = [
            candidate
            for candidate in rows
            if candidate["distance"] == key[0]
            and candidate["top_k"] == key[1]
            and candidate["tie_policy"] == key[2]
            and candidate["weight_profile"] == "moderate_dominance"
            and candidate["sigma_profile"] == "equal"
            and candidate["dominant_component_by_weight"] >= 0
            and candidate["component"] == candidate["dominant_component_by_weight"]
        ]
        if matching:
            mean_error = np.mean([x["mu_error_to_weight_dominant_block"] for x in matching])
            mean_mass = np.mean([x["block_mass_fraction"] for x in matching])
            mean_win = np.mean([x["weight_dominant_winning_block_fraction"] for x in matching])
            print(
                f"{key[0]:<4g} {key[1]:<3d} {key[2]:<17s} "
                f"{mean_error:12.2f} {mean_mass:15.3f} {mean_win:13.3f}"
            )


def plot_results(rows):
    """Gera gráficos de concordância, massa por bloco, largura e vitórias."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_center, ax_mass, ax_width, ax_wins = axes.flat

    # Um ponto por mistura/K/política, não uma linha por componente.
    unique_rows = {}
    for row in rows:
        key = (
            row["distance"], row["weight_profile"], row["sigma_profile"],
            row["component_order_low_to_high_mean"], row["tie_policy"], row["top_k"],
        )
        unique_rows.setdefault(key, row)

    for policy, marker in (("random", "o"), ("reverse_argsort", "x")):
        selected = [
            row for row in unique_rows.values()
            if row["tie_policy"] == policy and row["dominant_component_by_weight"] >= 0
        ]
        ax_center.scatter(
            [row["dominant_block_center"] / M for row in selected],
            [row["mu_h_over_m"] for row in selected],
            s=18,
            alpha=0.45,
            marker=marker,
            label=policy,
        )
    ax_center.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_center.set_xlabel("Centro teórico do bloco dominante / m")
    ax_center.set_ylabel("μ_H / m")
    ax_center.set_title("Centro global de H versus centro previsto")
    ax_center.legend()
    ax_center.grid(True, alpha=0.25)

    for policy, marker in (("random", "o-"), ("reverse_argsort", "s--")):
        means = []
        for k in TOP_K_VALUES:
            selected = [
                row["block_mass_fraction"]
                for row in rows
                if row["tie_policy"] == policy
                and row["weight_profile"] == "moderate_dominance"
                and row["sigma_profile"] == "equal"
                and row["component"] == row["dominant_component_by_weight"]
                and row["dominant_component_by_weight"] >= 0
                and row["top_k"] == k
            ]
            means.append(np.mean(selected) if selected else np.nan)
        ax_mass.plot(TOP_K_VALUES, means, marker, label=policy)
    ax_mass.set_xlabel("K")
    ax_mass.set_ylabel("Fração de H dentro do bloco dominante")
    ax_mass.set_title("Massa de persistência no bloco dominante (perfil 50/30/20)")
    ax_mass.set_ylim(-0.05, 1.05)
    ax_mass.legend()
    ax_mass.grid(True, alpha=0.25)

    for profile in WEIGHT_PROFILES:
        means = []
        for k in TOP_K_VALUES:
            selected = [
                row["sigma_h"]
                for row in unique_rows.values()
                if row["weight_profile"] == profile and row["top_k"] == k
            ]
            means.append(np.mean(selected) if selected else np.nan)
        ax_width.plot(TOP_K_VALUES, means, "o-", label=profile)
    ax_width.set_xlabel("K")
    ax_width.set_ylabel("σ_H (ranks)")
    ax_width.set_title("Largura global por perfil de pesos")
    ax_width.legend(fontsize="small")
    ax_width.grid(True, alpha=0.25)

    for policy, marker in (("random", "o-"), ("reverse_argsort", "s--")):
        means = []
        for k in TOP_K_VALUES:
            selected = [
                row["weight_dominant_winning_block_fraction"]
                for row in unique_rows.values()
                if row["tie_policy"] == policy
                and row["top_k"] == k
                and row["weight_profile"] == "moderate_dominance"
                and row["sigma_profile"] == "equal"
                and row["dominant_component_by_weight"] >= 0
            ]
            means.append(np.nanmean(selected) if selected else np.nan)
        ax_wins.plot(TOP_K_VALUES, means, marker, label=policy)
    ax_wins.set_xlabel("K")
    ax_wins.set_ylabel("Fração de réplicas em que o bloco dominante vence")
    ax_wins.set_ylim(-0.05, 1.05)
    ax_wins.set_title("Vitória do bloco de maior peso em cada réplica")
    ax_wins.legend()
    ax_wins.grid(True, alpha=0.25)

    fig.suptitle("Validação fatorial da persistência por blocos de ranks")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
