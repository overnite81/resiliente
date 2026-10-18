"""Experimentos de detectabilidade de anomalias por persistência de ranks.

Modelo: X ~ (1-p) U(0, 1) + p N(MIXTURE_MU, sigma).

O nulo de Monte Carlo usa exatamente o mesmo m, K, número de amostras
agregadas em H e desempate aleatório das densidades locais. A decisão primária
de detecção usa somente pvalue_JS < ALPHA; os outros p-values são métricas
secundárias. Os valores por ensaio, nulos e resumos são gravados em CSV.
"""

import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# Configuração do modelo e do método
MIXTURE_MU = 0.8
BASE_SIGMA = 0.005
EPS = 0.02
M_VALUES = (128, 256, 512, 1024, 2048)
K_VALUES = (5, 10, 20, 40, 80)
P_VALUES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
ANOMALY_SIGMAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
SIGMA_SWEEP_P = 0.02

# Cada ensaio forma H agregando SAMPLES_PER_H amostras de tamanho m.
SAMPLES_PER_H = 30
N_TRIALS = 100
B_NULL = 1000
ALPHA = 0.05
SMOOTHING_SIGMA = 2.0
SEED = 2025

TRIALS_CSV = "density_persistence_anomaly_trials.csv"
NULL_CSV = "density_persistence_anomaly_null.csv"
SUMMARY_CSV = "density_persistence_anomaly_summary.csv"


METRIC_NAMES = ("JS", "maxH", "peak_mass")


def generate_samples(rng, m, p, sigma, n_samples=SAMPLES_PER_H):
    """Gera amostras iid da mistura, sem forçar a contagem de anomalias."""
    values = rng.uniform(0.0, 1.0, size=(n_samples, m))
    anomaly_mask = rng.random((n_samples, m)) < p
    count = int(anomaly_mask.sum())
    if count:
        values[anomaly_mask] = rng.normal(
            loc=MIXTURE_MU,
            scale=sigma,
            size=count,
        )
    return values


def generate_null_samples(rng, m, n_samples=SAMPLES_PER_H):
    """Gera amostras sob H0: uniforme puro."""
    return rng.uniform(0.0, 1.0, size=(n_samples, m))


def local_density(sorted_values):
    """Número de observações a até EPS do valor ordenado de cada rank."""
    left = np.searchsorted(sorted_values, sorted_values - EPS, side="left")
    right = np.searchsorted(sorted_values, sorted_values + EPS, side="right")
    return right - left


def persistence_histograms(samples, k_values, rng):
    """Calcula H para vários K, com desempate aleatório entre densidades iguais."""
    m = samples.shape[1]
    histograms = {k: np.zeros(m, dtype=np.int64) for k in k_values}
    max_k = min(max(k_values), m)

    for sample in samples:
        ordered_values = np.sort(sample)
        density = local_density(ordered_values)
        tie_breakers = rng.random(m)
        # Densidade decrescente; dentro dos empates, ordem aleatória.
        rank_order = np.lexsort((tie_breakers, -density))
        top_ranks = rank_order[:max_k]
        for k in k_values:
            histograms[k] += np.bincount(top_ranks[: min(k, m)], minlength=m)

    return histograms


def jensen_shannon_uniform(histogram):
    """Divergência Jensen-Shannon em bits entre H normalizado e uniforme."""
    total = histogram.sum()
    if total == 0:
        return 0.0
    p = histogram.astype(float) / total
    uniform = np.full(len(histogram), 1.0 / len(histogram))
    midpoint = 0.5 * (p + uniform)

    positive_p = p > 0
    positive_u = uniform > 0
    kl_p = np.sum(p[positive_p] * np.log2(p[positive_p] / midpoint[positive_p]))
    kl_u = np.sum(uniform[positive_u] * np.log2(uniform[positive_u] / midpoint[positive_u]))
    return float(0.5 * kl_p + 0.5 * kl_u)


def histogram_metrics(histogram):
    """Métricas de H, incluindo pico bruto, massa FWHM e largura global."""
    total = int(histogram.sum())
    if total == 0:
        return {
            "JS": 0.0,
            "maxH": 0,
            "peak_mass": 0.0,
            "peak_rank": 0,
            "peak_quantile": 0.0,
            "peak_width": 0,
            "sigma_H": 0.0,
        }

    probabilities = histogram / total
    ranks = np.arange(len(histogram), dtype=float)
    mu_h = float(np.sum(ranks * probabilities))
    sigma_h = float(np.sqrt(np.sum(probabilities * (ranks - mu_h) ** 2)))

    # A posição reportada segue a definição solicitada: argmax(H) sem suavização.
    peak_rank = int(np.argmax(histogram))
    peak_quantile = peak_rank / len(histogram)

    # Massa do maior pico: massa normalizada de H dentro da largura a meia altura
    # do máximo do histograma suavizado. O suavizado estabiliza a estimativa FWHM.
    smoothed = gaussian_filter1d(histogram.astype(float), sigma=SMOOTHING_SIGMA)
    smooth_peak = int(np.argmax(smoothed))
    half_height = smoothed[smooth_peak] / 2.0
    left = smooth_peak
    right = smooth_peak
    while left > 0 and smoothed[left - 1] >= half_height:
        left -= 1
    while right < len(smoothed) - 1 and smoothed[right + 1] >= half_height:
        right += 1
    peak_width = right - left + 1
    peak_mass = float(probabilities[left : right + 1].sum())

    return {
        "JS": jensen_shannon_uniform(histogram),
        "maxH": int(histogram.max()),
        "peak_mass": peak_mass,
        "peak_rank": peak_rank,
        "peak_quantile": peak_quantile,
        "peak_width": int(peak_width),
        "sigma_H": sigma_h,
    }


def monte_carlo_pvalue(observed, null_values):
    """P-value unilateral superior com correção +1 de Monte Carlo."""
    null_values = np.asarray(null_values)
    return float((1 + np.count_nonzero(null_values >= observed)) / (len(null_values) + 1))


def true_cluster_quantile(p):
    """CDF da mistura na mediana do cluster: F(MIXTURE_MU)."""
    # U(0,1) tem CDF MIXTURE_MU nesse ponto; a normal tem CDF 1/2.
    return (1.0 - p) * MIXTURE_MU + p * 0.5


def make_null_distributions(rng):
    """Estima distribuições nulas de estatísticas para cada m e cada K."""
    null = {
        m: {k: {name: [] for name in METRIC_NAMES} for k in K_VALUES}
        for m in M_VALUES
    }
    null_rows = []

    print(
        f"Monte Carlo nulo: B={B_NULL}, {SAMPLES_PER_H} amostras por H, "
        f"m={M_VALUES}, K={K_VALUES}"
    )
    for m in M_VALUES:
        for simulation in range(B_NULL):
            samples = generate_null_samples(rng, m)
            histograms = persistence_histograms(samples, K_VALUES, rng)
            for k in K_VALUES:
                metrics = histogram_metrics(histograms[k])
                for metric_name in METRIC_NAMES:
                    null[m][k][metric_name].append(metrics[metric_name])
                null_rows.append(
                    {
                        "m": m,
                        "K": k,
                        "simulation": simulation,
                        **metrics,
                    }
                )
        print(f"  nulo concluído para m={m}")

    write_csv(NULL_CSV, null_rows)
    return null


def run_condition(rng, null, experiment, m, p, sigma):
    """Executa N_TRIALS para p, sigma e m, calculando todos os valores de K."""
    trial_rows = []
    for trial in range(N_TRIALS):
        samples = generate_samples(rng, m, p, sigma)
        histograms = persistence_histograms(samples, K_VALUES, rng)
        for k in K_VALUES:
            metrics = histogram_metrics(histograms[k])
            pvalues = {
                f"pvalue_{metric_name}": monte_carlo_pvalue(
                    metrics[metric_name], null[m][k][metric_name]
                )
                for metric_name in METRIC_NAMES
            }
            detected = int(pvalues["pvalue_JS"] < ALPHA)
            q_true = true_cluster_quantile(p)
            trial_rows.append(
                {
                    "experiment": experiment,
                    "trial": trial,
                    "p": p,
                    "sigma": sigma,
                    "m": m,
                    "K": k,
                    "K_over_m": k / m,
                    "R_pm_over_K": p * m / k,
                    **metrics,
                    **pvalues,
                    "peak_error": abs(metrics["peak_quantile"] - MIXTURE_MU),
                    "true_cluster_quantile": q_true,
                    "peak_error_to_true_quantile": abs(
                        metrics["peak_quantile"] - q_true
                    ),
                    "detected": detected,
                    "alpha": ALPHA,
                }
            )
    return trial_rows


def summarize_trials(trial_rows):
    """Agrega potência, métricas médias e erros por condição experimental."""
    groups = defaultdict(list)
    for row in trial_rows:
        key = (
            row["experiment"], row["p"], row["sigma"], row["m"], row["K"]
        )
        groups[key].append(row)

    summaries = []
    for key, rows in groups.items():
        experiment, p, sigma, m, k = key
        summary = {
            "experiment": experiment,
            "p": p,
            "sigma": sigma,
            "m": m,
            "K": k,
            "K_over_m": k / m,
            "R_pm_over_K": p * m / k,
            "n_trials": len(rows),
            "power_JS": float(np.mean([r["detected"] for r in rows])),
            "JS_mean": float(np.mean([r["JS"] for r in rows])),
            "maxH_mean": float(np.mean([r["maxH"] for r in rows])),
            "peak_mass_mean": float(np.mean([r["peak_mass"] for r in rows])),
            "peak_rank_mean": float(np.mean([r["peak_rank"] for r in rows])),
            "peak_quantile_mean": float(np.mean([r["peak_quantile"] for r in rows])),
            "peak_error_mean": float(np.mean([r["peak_error"] for r in rows])),
            "peak_error_max": float(np.max([r["peak_error"] for r in rows])),
            "peak_error_std": float(np.std([r["peak_error"] for r in rows], ddof=1))
            if len(rows) > 1
            else 0.0,
            "peak_error_true_quantile_mean": float(
                np.mean([r["peak_error_to_true_quantile"] for r in rows])
            ),
            "sigma_H_mean": float(np.mean([r["sigma_H"] for r in rows])),
            "pvalue_JS_mean": float(np.mean([r["pvalue_JS"] for r in rows])),
            "pvalue_maxH_mean": float(np.mean([r["pvalue_maxH"] for r in rows])),
            "pvalue_mass_mean": float(np.mean([r["pvalue_peak_mass"] for r in rows])),
        }
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    """Grava uma lista de dicionários em CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Arquivo gravado: {path}")


def find_summary(summaries, experiment, m, k, p=None, sigma=None):
    """Localiza uma condição agregada pelo conjunto de fatores informado."""
    for row in summaries:
        if row["experiment"] != experiment or row["m"] != m or row["K"] != k:
            continue
        if p is not None and not np.isclose(row["p"], p):
            continue
        if sigma is not None and not np.isclose(row["sigma"], sigma):
            continue
        return row
    return None


def plot_results(summaries):
    """Gera os gráficos pedidos: dose, potência, localização, escala e largura."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    ax_js, ax_power, ax_error, ax_scale = axes[0]
    ax_width, ax_sigma, ax_collapse, ax_null = axes[1]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(M_VALUES)))

    # Experimentos com m=512 e K=20, variando p.
    for color, m in zip(colors, M_VALUES):
        rows = sorted(
            [
                row for row in summaries
                if row["experiment"] == "p_sweep" and row["m"] == m and row["K"] == 20
            ],
            key=lambda row: row["p"],
        )
        if rows:
            ax_power.plot([r["p"] for r in rows], [r["power_JS"] for r in rows], "o-",
                          color=color, label=f"m={m}")
    ax_power.set_xlabel("Fração anômala p")
    ax_power.set_ylabel("Potência (rejeição por JS)")
    ax_power.set_title("Potência versus p (K=20)")
    ax_power.set_ylim(-0.05, 1.05)
    ax_power.legend(fontsize="small")
    ax_power.grid(True, alpha=0.25)

    for color, m in zip(colors, M_VALUES):
        rows = sorted(
            [
                row for row in summaries
                if row["experiment"] == "p_sweep" and row["m"] == m and row["K"] == 20
            ],
            key=lambda row: row["p"],
        )
        if rows:
            ax_js.plot([r["p"] for r in rows], [r["JS_mean"] for r in rows], "o-",
                       color=color, label=f"m={m}")
    ax_js.set_xlabel("Fração anômala p")
    ax_js.set_ylabel("JS médio (bits)")
    ax_js.set_title("JS versus p (K=20)")
    ax_js.legend(fontsize="small")
    ax_js.grid(True, alpha=0.25)

    location_rows = [
        row for row in summaries
        if row["experiment"] == "p_sweep" and row["m"] == 512 and row["K"] == 20
    ]
    location_rows.sort(key=lambda row: row["p"])
    if location_rows:
        ax_error.errorbar(
            [r["p"] for r in location_rows],
            [r["peak_error_true_quantile_mean"] for r in location_rows],
            yerr=[r["peak_error_std"] for r in location_rows],
            fmt="o-",
            capsize=3,
        )
    ax_error.set_xlabel("Fração anômala p")
    ax_error.set_ylabel("Erro médio do quantil do pico")
    ax_error.set_title("Localização (m=512, K=20)")
    ax_error.grid(True, alpha=0.25)

    scale_rows = [
        row for row in summaries
        if row["experiment"] == "p_sweep" and row["p"] > 0
    ]
    for k in K_VALUES:
        rows = sorted(
            [row for row in scale_rows if row["K"] == k],
            key=lambda row: row["R_pm_over_K"],
        )
        if rows:
            ax_scale.scatter(
                [r["R_pm_over_K"] for r in rows],
                [r["power_JS"] for r in rows],
                s=18,
                alpha=0.55,
                label=f"K={k}",
            )
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("R = p·m/K")
    ax_scale.set_ylabel("Potência")
    ax_scale.set_title("Potência versus R")
    ax_scale.set_ylim(-0.05, 1.05)
    ax_scale.legend(fontsize="small")
    ax_scale.grid(True, which="both", alpha=0.25)

    for p in P_VALUES:
        rows = sorted(
            [
                row for row in summaries
                if row["experiment"] == "p_sweep" and row["m"] == 512
                and np.isclose(row["p"], p)
            ],
            key=lambda row: row["K"],
        )
        if rows:
            ax_width.plot([r["K"] for r in rows], [r["sigma_H_mean"] for r in rows],
                          "o-", label=f"p={p:g}")
    ax_width.set_xlabel("K")
    ax_width.set_ylabel("σ_H médio (ranks)")
    ax_width.set_title("Largura versus K (m=512)")
    ax_width.legend(fontsize="x-small", ncol=2)
    ax_width.grid(True, alpha=0.25)

    sigma_rows = [
        row for row in summaries
        if row["experiment"] == "sigma_sweep" and row["m"] == 512 and row["K"] == 20
    ]
    sigma_rows.sort(key=lambda row: row["sigma"])
    if sigma_rows:
        ax_sigma.plot([r["sigma"] for r in sigma_rows], [r["power_JS"] for r in sigma_rows],
                      "o-")
    ax_sigma.set_xlabel("Desvio-padrão da anomalia σ")
    ax_sigma.set_ylabel("Potência")
    ax_sigma.set_title(f"Potência versus σ (p={SIGMA_SWEEP_P:g}, m=512, K=20)")
    ax_sigma.set_ylim(-0.05, 1.05)
    ax_sigma.grid(True, alpha=0.25)

    # Curva de colapso: binned average de potência em bins logarítmicos de R.
    if scale_rows:
        r_values = np.array([row["R_pm_over_K"] for row in scale_rows])
        powers = np.array([row["power_JS"] for row in scale_rows])
        bins = np.geomspace(r_values.min(), r_values.max(), 9)
        centers = np.sqrt(bins[:-1] * bins[1:])
        mean_power = []
        for low, high in zip(bins[:-1], bins[1:]):
            selected = (r_values >= low) & (r_values < high)
            mean_power.append(np.mean(powers[selected]) if selected.any() else np.nan)
        ax_collapse.plot(centers, mean_power, "o-")
    ax_collapse.set_xscale("log")
    ax_collapse.set_xlabel("R = p·m/K")
    ax_collapse.set_ylabel("Potência média em bins de R")
    ax_collapse.set_title("Colapso de escala (binned)")
    ax_collapse.set_ylim(-0.05, 1.05)
    ax_collapse.grid(True, which="both", alpha=0.25)

    # Verificação visual dos níveis nulos no m=512, K=20.
    null_summary = find_summary(summaries, "p_sweep", 512, 20, p=0.0)
    if null_summary:
        ax_null.bar(
            ["JS", "maxH", "massa"],
            [null_summary["JS_mean"], null_summary["maxH_mean"],
             null_summary["peak_mass_mean"]],
        )
    ax_null.set_title("Médias sob p=0 (m=512, K=20)")
    ax_null.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Detectabilidade de clusters por persistência de ranks")
    fig.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(SEED)
    null = make_null_distributions(rng)
    trial_rows = []

    # Experimentos 1, 3 e 4: varredura de p para todos os m e K.
    for m in M_VALUES:
        for p in P_VALUES:
            trial_rows.extend(
                run_condition(rng, null, "p_sweep", m, p, BASE_SIGMA)
            )

    # Experimento 7: varredura da compactação em m=512; todos os K são registrados.
    for sigma in ANOMALY_SIGMAS:
        trial_rows.extend(
            run_condition(rng, null, "sigma_sweep", 512, SIGMA_SWEEP_P, sigma)
        )

    summaries = summarize_trials(trial_rows)
    write_csv(TRIALS_CSV, trial_rows)
    write_csv(SUMMARY_CSV, summaries)
    plot_results(summaries)


if __name__ == "__main__":
    main()
