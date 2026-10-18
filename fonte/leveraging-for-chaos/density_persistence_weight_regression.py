"""Regressão do centro de massa da persistência em função do peso dominante.

Varia w de 0.50 a 0.99 para d = 2, 4, 8 e 16. Para cada condição,
estima μ_H e σ_H a partir do histograma agregado de várias réplicas.
Ajusta μ_H = a + b*w tanto incluindo w=0.50 como apenas para w>0.50,
pois o caso balanceado pode ser um regime especial.

O programa grava density_persistence_weight_regression.csv e exibe gráficos.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


# Parâmetros do experimento
M = 256 * 2
DISTANCES = (2.0, 4.0, 8.0, 16.0)
WEIGHTS = np.round(np.arange(0.50, 1.00, 0.01), 2)
SIGMA = 0.3
EPS = 0.5
TOP_K = 20
N_REPLICATES = 5
SAMPLES_PER_REPLICATE = 1000
SEED = 123
CSV_PATH = "density_persistence_weight_regression.csv"


def local_density_sorted(x, eps):
    """Calcula a densidade local em cada posição de um vetor ordenado."""
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
    """Conta as ocorrências de cada rank entre os top_k mais densos."""
    histogram = np.zeros(samples.shape[1], dtype=np.int64)
    for row in samples:
        sorted_row = np.sort(row)
        density = local_density_sorted(sorted_row, EPS)
        winners = np.argsort(density)[::-1][:TOP_K]
        histogram[winners] += 1
    return histogram


def make_samples(rng, distance, weight_a, n_samples):
    """Gera uma mistura com contagens fixas e grupo A como componente menor."""
    n_a = int(round(weight_a * M))
    n_b = M - n_a
    samples = np.empty((n_samples, M), dtype=float)

    for index in range(n_samples):
        group_a = rng.normal(loc=20.0, scale=SIGMA, size=n_a)
        group_b = rng.normal(loc=20.0 + distance, scale=SIGMA, size=n_b)
        samples[index] = np.concatenate((group_a, group_b))

    return samples, n_a, n_b


def histogram_statistics(histogram):
    """Retorna centro de massa μ_H e desvio-padrão σ_H do histograma."""
    total = histogram.sum()
    if total == 0:
        return np.nan, np.nan

    ranks = np.arange(len(histogram), dtype=float)
    probabilities = histogram / total
    mu = np.sum(ranks * probabilities)
    sigma_h = np.sqrt(np.sum(probabilities * (ranks - mu) ** 2))
    return float(mu), float(sigma_h)


def linear_fit(x, y):
    """Ajusta y=a+b*x e retorna a, b e R²."""
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = intercept + slope * x
    residual_sum = np.sum((y - predicted) ** 2)
    total_sum = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    return float(intercept), float(slope), float(r_squared)


def run_experiment():
    rng = np.random.default_rng(SEED)
    observations = {distance: [] for distance in DISTANCES}
    total_samples = N_REPLICATES * SAMPLES_PER_REPLICATE

    print(
        f"m={M}, pesos={WEIGHTS[0]:.2f}..{WEIGHTS[-1]:.2f}, "
        f"réplicas por condição={N_REPLICATES}×{SAMPLES_PER_REPLICATE}"
    )
    print(f"Total: {len(DISTANCES) * len(WEIGHTS)} condições, {total_samples} amostras/condição")

    for distance in DISTANCES:
        print(f"\nDistância d={distance:g}")
        for weight_a in WEIGHTS:
            combined_histogram = np.zeros(M, dtype=np.int64)
            replicate_centers = []
            n_a = n_b = 0

            for _ in range(N_REPLICATES):
                samples, n_a, n_b = make_samples(
                    rng,
                    distance,
                    weight_a,
                    SAMPLES_PER_REPLICATE,
                )
                replicate_histogram = persistence_histogram(samples)
                combined_histogram += replicate_histogram
                replicate_mu, _ = histogram_statistics(replicate_histogram)
                replicate_centers.append(replicate_mu)

            mu_h, sigma_h = histogram_statistics(combined_histogram)
            standard_error = np.std(replicate_centers, ddof=1) / np.sqrt(N_REPLICATES)
            observations[distance].append(
                {
                    "weight": float(weight_a),
                    "n_a": n_a,
                    "n_b": n_b,
                    "mu": mu_h,
                    "mu_se": float(standard_error),
                    "sigma_h": sigma_h,
                }
            )
            print(
                f"w={weight_a:.2f} ({n_a}/{n_b}) "
                f"mu_H={mu_h:7.2f} ± {standard_error:5.2f} "
                f"sigma_H={sigma_h:6.2f}"
            )

    regression_results = {}
    print("\nRegressões de μ_H = a + b·w")
    print("d     faixa       a          b          R²       RMSE vs m·w/2")
    print("-" * 72)

    for distance in DISTANCES:
        rows = observations[distance]
        weights = np.array([row["weight"] for row in rows])
        centers = np.array([row["mu"] for row in rows])
        dominant = weights > 0.5

        fit_all = linear_fit(weights, centers)
        fit_dominant = linear_fit(weights[dominant], centers[dominant])
        theory = M * weights[dominant] / 2
        rmse_theory = float(np.sqrt(np.mean((centers[dominant] - theory) ** 2)))
        regression_results[distance] = {
            "all": fit_all,
            "dominant": fit_dominant,
            "rmse_theory": rmse_theory,
        }

        print(
            f"{distance:<5g} todos   {fit_all[0]:9.3f} {fit_all[1]:10.3f} "
            f"{fit_all[2]:9.5f}          -"
        )
        print(
            f"{distance:<5g} w>0.5   {fit_dominant[0]:9.3f} "
            f"{fit_dominant[1]:10.3f} {fit_dominant[2]:9.5f} "
            f"{rmse_theory:12.3f}"
        )

    write_csv(observations, regression_results)
    plot_results(observations, regression_results)


def write_csv(observations, regression_results):
    """Grava as medidas por condição e os parâmetros das regressões."""
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "distance",
                "weight_a",
                "weight_b",
                "n_a",
                "n_b",
                "n_samples",
                "mu_h",
                "mu_h_standard_error",
                "sigma_h",
                "fit_all_intercept",
                "fit_all_slope",
                "fit_all_r_squared",
                "fit_dominant_intercept",
                "fit_dominant_slope",
                "fit_dominant_r_squared",
                "dominant_rmse_vs_mw_over_2",
            ]
        )

        for distance, rows in observations.items():
            fit_all = regression_results[distance]["all"]
            fit_dominant = regression_results[distance]["dominant"]
            rmse_theory = regression_results[distance]["rmse_theory"]
            for row in rows:
                writer.writerow(
                    [
                        distance,
                        row["weight"],
                        1.0 - row["weight"],
                        row["n_a"],
                        row["n_b"],
                        N_REPLICATES * SAMPLES_PER_REPLICATE,
                        row["mu"],
                        row["mu_se"],
                        row["sigma_h"],
                        fit_all[0],
                        fit_all[1],
                        fit_all[2],
                        fit_dominant[0],
                        fit_dominant[1],
                        fit_dominant[2],
                        rmse_theory,
                    ]
                )
    print(f"\nResultados gravados em {CSV_PATH}")


def plot_results(observations, regression_results):
    """Plota μ_H, σ_H e a qualidade dos ajustes para cada distância."""
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DISTANCES)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax_mu, ax_sigma, ax_slope, ax_r2 = axes.flat
    for color, distance in zip(colors, DISTANCES):
        rows = observations[distance]
        weights = np.array([row["weight"] for row in rows])
        centers = np.array([row["mu"] for row in rows])
        errors = np.array([row["mu_se"] for row in rows])
        widths = np.array([row["sigma_h"] for row in rows])

        ax_mu.errorbar(
            weights,
            centers,
            yerr=errors,
            marker=".",
            markersize=4,
            linewidth=1,
            capsize=2,
            color=color,
            label=f"d={distance:g}",
        )
        ax_sigma.plot(weights, widths, marker=".", linewidth=1, color=color)

        intercept, slope, r_squared = regression_results[distance]["dominant"]
        fit_weights = weights[weights > 0.5]
        ax_mu.plot(
            fit_weights,
            intercept + slope * fit_weights,
            linestyle="--",
            color=color,
            alpha=0.8,
        )

    ax_mu.plot(
        WEIGHTS[WEIGHTS > 0.5],
        M * WEIGHTS[WEIGHTS > 0.5] / 2,
        "k:",
        linewidth=2,
        label="referência m·w/2",
    )
    ax_mu.set_xlabel("Peso do grupo A (w)")
    ax_mu.set_ylabel("Centro do histograma (μ_H)")
    ax_mu.set_title("μ_H em função do peso (tracejado: ajuste para w>0.5)")
    ax_mu.legend(fontsize="small")
    ax_mu.grid(True, alpha=0.3)

    ax_sigma.axvline(0.5, color="black", linestyle=":", linewidth=1)
    ax_sigma.set_xlabel("Peso do grupo A (w)")
    ax_sigma.set_ylabel("Largura do histograma (σ_H)")
    ax_sigma.set_title("Largura e possível comportamento especial em 50/50")
    ax_sigma.grid(True, alpha=0.3)

    slopes = [regression_results[d]["dominant"][1] for d in DISTANCES]
    ax_slope.plot(DISTANCES, slopes, "o-", label="inclinação ajustada")
    ax_slope.axhline(M / 2, color="black", linestyle="--", label=f"m/2={M / 2:g}")
    ax_slope.set_xlabel("Separação modal (d)")
    ax_slope.set_ylabel("Inclinação b")
    ax_slope.set_title("Inclinação da regressão para w>0.5")
    ax_slope.legend()
    ax_slope.grid(True, alpha=0.3)

    r2_all = [regression_results[d]["all"][2] for d in DISTANCES]
    r2_dominant = [regression_results[d]["dominant"][2] for d in DISTANCES]
    positions = np.arange(len(DISTANCES))
    width = 0.36
    ax_r2.bar(positions - width / 2, r2_all, width, label="todos os pesos")
    ax_r2.bar(positions + width / 2, r2_dominant, width, label="w>0.5")
    ax_r2.set_xticks(positions, [f"{d:g}" for d in DISTANCES])
    ax_r2.set_ylim(0, 1.05)
    ax_r2.set_xlabel("Separação modal (d)")
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("Qualidade do ajuste linear")
    ax_r2.legend()
    ax_r2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Regressão peso–centro do histograma de persistência")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
