"""Rank Selection Persistence e homologia persistente H0 no grafo dos ranks.

Implementa os cenários e métricas de PLAN.md. Cada amostra ordenada define
um grafo caminho nos ranks, ponderado pela densidade local. A filtração de
supernível acompanha componentes conexas; empates são ativados em conjunto.

Exemplo rápido:
    python rank_persistence_topology.py --quick

Execução completa (pode ser longa e gerar arquivos grandes):
    python rank_persistence_topology.py --replicates 100 --samples-per-ensemble 30
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit
from scipy.stats import spearmanr


FULL_M_VALUES = (128, 256, 512, 1024, 2048)
FULL_K_VALUES = (5, 10, 20, 40, 80)
CORE_EPSILONS = (0.05, 0.10, 0.20)
ANOMALY_EPSILONS = (0.02, 0.05, 0.10)
TIE_POLICIES = ("deterministic", "include_all", "fractional")
ALPHA = 0.05
MIN_PERSISTENCE_FRACTION = 0.05


@dataclass
class Barcode:
    component_id: int
    birth: float
    death: float
    persistence: float
    members: set[int]
    peak_rank: int
    essential: bool


def local_density(sorted_values: np.ndarray, epsilon: float) -> np.ndarray:
    """D(i): quantidade de valores a distância <= epsilon do valor no rank i."""
    left = np.searchsorted(sorted_values, sorted_values - epsilon, side="left")
    right = np.searchsorted(sorted_values, sorted_values + epsilon, side="right")
    return (right - left).astype(np.int32)


def select_rank_weights(
    density: np.ndarray,
    k: int,
    policy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pesos de seleção; fractional soma exatamente K mesmo em platôs."""
    m = len(density)
    k = min(max(int(k), 1), m)
    weights = np.zeros(m, dtype=float)

    if policy == "deterministic":
        order = np.lexsort((np.arange(m), -density))
        weights[order[:k]] = 1.0
    elif policy == "include_all":
        threshold = np.partition(density, m - k)[m - k]
        weights[density >= threshold] = 1.0
    elif policy == "fractional":
        threshold = np.partition(density, m - k)[m - k]
        above = density > threshold
        tied = density == threshold
        weights[above] = 1.0
        remaining = k - int(above.sum())
        weights[tied] = remaining / int(tied.sum())
    else:
        raise ValueError(f"Política de empate desconhecida: {policy}")
    return weights


def superlevel_component_counts(density: np.ndarray) -> list[tuple[int, int]]:
    """Retorna (limiar, beta0) para cada nível distinto da filtração."""
    levels = sorted(np.unique(density), reverse=True)
    result = []
    for level in levels:
        active = density >= level
        starts = active & np.concatenate(([True], ~active[:-1]))
        result.append((int(level), int(starts.sum())))
    return result


def h0_superlevel_path(density: np.ndarray) -> tuple[list[Barcode], np.ndarray]:
    """Barcode H0 da filtração {i: D(i)>=lambda} no grafo caminho.

    Todos os vértices de mesmo nível são ativados simultaneamente. A regra
    elder mantém a componente com maior nascimento; empates usam menor rank.
    O último componente morre em nível 0. T(i) agrega apenas classes H0 finitas,
    evitando que a classe essencial torne o perfil constante em todo o caminho.
    """
    density = np.asarray(density, dtype=np.int32)
    m = len(density)
    active = np.zeros(m, dtype=bool)
    component_at_rank = np.full(m, -1, dtype=np.int32)
    parent: dict[int, int] = {}
    states: dict[int, dict] = {}
    bars: list[Barcode] = []
    next_id = 0

    def find(component_id: int) -> int:
        root = component_id
        while parent[root] != root:
            root = parent[root]
        while parent[component_id] != component_id:
            next_parent = parent[component_id]
            parent[component_id] = root
            component_id = next_parent
        return root

    def elder_key(component_id: int) -> tuple[float, int]:
        state = states[component_id]
        return (state["birth"], -state["birth_rank"])

    def emit_bar(component_id: int, death: float, essential: bool) -> None:
        state = states[component_id]
        members = set(state["members"])
        if not members:
            return
        peak_rank = min(members, key=lambda rank: (-density[rank], rank))
        persistence = max(0.0, float(state["birth"] - death))
        bars.append(
            Barcode(
                component_id=component_id,
                birth=float(state["birth"]),
                death=float(death),
                persistence=persistence,
                members=members,
                peak_rank=int(peak_rank),
                essential=essential,
            )
        )

    # Agrupamos ranks contíguos que nascem no mismo nível para evitar barras
    # espúrias de persistência zero causadas pela ordem de processamento.
    for level in sorted(np.unique(density), reverse=True):
        new_ranks = np.flatnonzero(density == level)
        if not len(new_ranks):
            continue
        run_start = 0
        while run_start < len(new_ranks):
            run_end = run_start
            while (
                run_end + 1 < len(new_ranks)
                and new_ranks[run_end + 1] == new_ranks[run_end] + 1
            ):
                run_end += 1
            left_rank = int(new_ranks[run_start])
            right_rank = int(new_ranks[run_end])
            new_members = set(range(left_rank, right_rank + 1))

            neighboring_components = set()
            if left_rank > 0 and active[left_rank - 1]:
                neighboring_components.add(find(int(component_at_rank[left_rank - 1])))
            if right_rank + 1 < m and active[right_rank + 1]:
                neighboring_components.add(find(int(component_at_rank[right_rank + 1])))

            if not neighboring_components:
                component_id = next_id
                next_id += 1
                parent[component_id] = component_id
                states[component_id] = {
                    "birth": float(level),
                    "birth_rank": left_rank,
                    "members": new_members,
                }
            else:
                component_id = max(neighboring_components, key=elder_key)
                losers = sorted(neighboring_components - {component_id})
                for loser in losers:
                    # Registra o suporte do componente imediatamente antes da fusão.
                    emit_bar(loser, float(level), essential=False)
                    winner_members = states[component_id]["members"]
                    loser_members = states[loser]["members"]
                    if len(winner_members) >= len(loser_members):
                        winner_members.update(loser_members)
                        states[component_id]["members"] = winner_members
                    else:
                        loser_members.update(winner_members)
                        states[component_id]["members"] = loser_members
                    parent[loser] = component_id
                    states[loser]["members"] = set()
                states[component_id]["members"].update(new_members)

            active[left_rank : right_rank + 1] = True
            component_at_rank[left_rank : right_rank + 1] = component_id
            run_start = run_end + 1

    surviving = {find(int(component_at_rank[i])) for i in range(m)}
    for component_id in sorted(surviving):
        emit_bar(component_id, 0.0, essential=True)

    topological_score = np.zeros(m, dtype=float)
    for bar in bars:
        if bar.persistence <= 0 or bar.essential:
            continue
        indices = np.fromiter(bar.members, dtype=int)
        topological_score[indices] = np.maximum(
            topological_score[indices], bar.persistence
        )
    return bars, topological_score


def persistence_selection_js(profile: np.ndarray) -> float:
    """Jensen-Shannon em bits entre perfil normalizado e uniforme."""
    total = profile.sum()
    if total <= 0:
        return 0.0
    p = profile.astype(float) / total
    uniform = np.full(len(p), 1.0 / len(p))
    midpoint = 0.5 * (p + uniform)
    mask = p > 0
    kl_p = np.sum(p[mask] * np.log2(p[mask] / midpoint[mask]))
    kl_u = np.sum(uniform * np.log2(uniform / midpoint))
    return float(0.5 * (kl_p + kl_u))


def profile_metrics(profile: np.ndarray) -> dict:
    """Métricas de perfil de seleção normalizado."""
    total = float(profile.sum())
    p = profile / total if total else np.full(len(profile), 1.0 / len(profile))
    ranks = np.arange(len(profile), dtype=float)
    center = float(np.sum(ranks * p))
    width = float(np.sqrt(np.sum(p * (ranks - center) ** 2)))
    peak = int(np.argmax(profile))
    half = float(profile[peak]) / 2
    left = peak
    right = peak
    while left > 0 and profile[left - 1] >= half and half > 0:
        left -= 1
    while right < len(profile) - 1 and profile[right + 1] >= half and half > 0:
        right += 1
    return {
        "JS_uniform": persistence_selection_js(profile),
        "max_profile": float(np.max(p)) if len(p) else 0.0,
        "center_rank": center,
        "center_quantile": center / len(profile),
        "peak_rank": peak,
        "peak_quantile": peak / len(profile),
        "peak_width_fwhm": right - left + 1,
        "sigma_H": width,
    }


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rho, representando perfis constantes como NaN."""
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    result = spearmanr(left, right)
    return float(result.statistic)


def diagram_wasserstein1(left: list[Barcode], right: list[Barcode]) -> float:
    """Distância de matching entre diagramas H0 com custo L-infinito.

    Pontos podem ser emparelhados entre diagramas ou apagados/criados ao custo
    da distância L-infinito até a diagonal. O resultado é custo total por barra.
    """
    left_points = np.asarray([(bar.birth, bar.death) for bar in left], dtype=float)
    right_points = np.asarray([(bar.birth, bar.death) for bar in right], dtype=float)
    n, m = len(left_points), len(right_points)
    if n == 0 and m == 0:
        return 0.0
    size = n + m
    large = 1e9
    cost = np.full((size, size), large, dtype=float)
    if n and m:
        cost[:n, :m] = np.max(
            np.abs(left_points[:, None, :] - right_points[None, :, :]), axis=2
        )
    for i, point in enumerate(left_points):
        cost[i, m + i] = abs(point[0] - point[1]) / 2
    for j, point in enumerate(right_points):
        cost[n + j, j] = abs(point[0] - point[1]) / 2
    if m and n:
        cost[n:, m:] = 0.0
    elif n:
        cost[n:, m:] = 0.0
    elif m:
        cost[n:, m:] = 0.0
    row_indices, col_indices = linear_sum_assignment(cost)
    total = float(cost[row_indices, col_indices].sum())
    return total / max(n, m, 1)


def generate_sample(rng: np.random.Generator, scenario: dict, m: int):
    """Sample, labels conhecidos e quantil-alvo para os cenários do plano."""
    name = scenario["scenario"]
    params = scenario["params"]
    labels = np.zeros(m, dtype=np.int8)
    target_quantile = float("nan")

    if name == "uniform":
        sample = rng.uniform(0.0, 1.0, m)
    elif name == "normal":
        sample = rng.normal(0.0, 1.0, m)
        target_quantile = 0.5
    elif name == "bimodal_separated":
        labels = (rng.random(m) >= 0.5).astype(np.int8)
        sample = rng.normal(np.where(labels == 0, -2.0, 2.0), 0.5)
        target_quantile = 0.25
    elif name == "bimodal_overlap":
        labels = (rng.random(m) >= 0.5).astype(np.int8)
        sample = rng.normal(np.where(labels == 0, -0.75, 0.75), 1.0)
        # Componentes sobrepostas não definem um bloco de ranks verdadeiro único.
        target_quantile = float("nan")
    elif name == "weighted_mixture":
        labels = (rng.random(m) >= 0.75).astype(np.int8)
        sample = rng.normal(np.where(labels == 0, -2.0, 2.0), 0.5)
        target_quantile = 0.375
    elif name == "cluster_anomaly":
        p = float(params["p"])
        sigma = float(params["sigma"])
        q = float(params["position"])
        sample = rng.uniform(0.0, 1.0, m)
        labels = (rng.random(m) < p).astype(np.int8)
        sample[labels == 1] = rng.normal(q, sigma, int(labels.sum()))
        target_quantile = (
            (1.0 - p) * q + 0.5 * p if p > 0 else float("nan")
        )
    elif name == "outliers":
        p = float(params.get("p", 0.02))
        sample = rng.normal(0.0, 1.0, m)
        labels = (rng.random(m) < p).astype(np.int8)
        sample[labels == 1] = rng.uniform(-8.0, 8.0, int(labels.sum()))
        target_quantile = float("nan")
    else:
        raise ValueError(f"Cenário desconhecido: {name}")

    return np.asarray(sample, dtype=float), labels, target_quantile


def build_scenarios(full_m_values: tuple[int, ...], quick: bool) -> list[dict]:
    """Constrói casos base e varreduras específicas do cluster anômalo."""
    if quick:
        return [
            {"scenario": "uniform", "case_id": "uniform", "params": {},
             "m_values": (64, 128), "epsilons": (0.05,)},
            {"scenario": "normal", "case_id": "normal", "params": {},
             "m_values": (64, 128), "epsilons": (0.10,)},
            {"scenario": "bimodal_separated", "case_id": "bimodal_separated", "params": {},
             "m_values": (64, 128), "epsilons": (0.10,)},
            {"scenario": "cluster_anomaly", "case_id": "anomaly_p0.05", 
             "params": {"p": 0.05, "sigma": 0.02, "position": 0.8},
             "m_values": (64, 128), "epsilons": (0.05,)},
            {"scenario": "outliers", "case_id": "outliers", "params": {"p": 0.05},
             "m_values": (64, 128), "epsilons": (0.10,)},
        ]

    scenarios = []
    core = (
        ("uniform", {}),
        ("normal", {}),
        ("bimodal_separated", {}),
        ("bimodal_overlap", {}),
        ("weighted_mixture", {}),
        ("outliers", {"p": 0.02}),
    )
    for name, params in core:
        scenarios.append(
            {
                "scenario": name,
                "case_id": name,
                "params": params,
                "m_values": full_m_values,
                "epsilons": (
                    tuple(sorted(set(CORE_EPSILONS + ANOMALY_EPSILONS)))
                    if name == "uniform"
                    else CORE_EPSILONS
                ),
            }
        )

    # Varia p em todos os m para medir a curva de potência e o colapso de escala.
    for p in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10):
        for m in full_m_values:
            params = {"p": p, "sigma": 0.01, "position": 0.8}
            scenarios.append(
                {"scenario": "cluster_anomaly", "case_id": f"p_sweep_p{p:g}_m{m}",
                 "params": params, "m_values": (m,), "epsilons": ANOMALY_EPSILONS}
            )

    # Scaling with m, including the complete m grid.
    for m in full_m_values:
        scenarios.append(
            {"scenario": "cluster_anomaly", "case_id": f"m_sweep_m{m}",
             "params": {"p": 0.02, "sigma": 0.01, "position": 0.8},
             "m_values": (m,), "epsilons": (0.05,)}
        )

    # Compactness and rank-location experiments.
    for sigma in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        scenarios.append(
            {"scenario": "cluster_anomaly", "case_id": f"sigma_sweep_s{sigma:g}",
             "params": {"p": 0.02, "sigma": sigma, "position": 0.8},
             "m_values": (512,), "epsilons": (0.05,)}
        )
    for position in (0.2, 0.5, 0.8):
        scenarios.append(
            {"scenario": "cluster_anomaly", "case_id": f"position_sweep_q{position:g}",
             "params": {"p": 0.02, "sigma": 0.01, "position": position},
             "m_values": (512,), "epsilons": (0.05,)}
        )
    return scenarios


def jaccard_similarity(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Jaccard entre máscaras booleanas; NaN se ambas forem vazias."""
    union = np.count_nonzero(estimated | truth)
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(estimated & truth) / union)


def save_dict_rows(path: Path, rows: list[dict]) -> None:
    """Escreve CSV sem assumir previamente as colunas."""
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    rng: np.random.Generator,
    scenario: dict,
    m: int,
    epsilon: float,
    k_values: tuple[int, ...],
    replicate_count: int,
    samples_per_ensemble: int,
    perturb: bool,
):
    """Executa um caso; retorna linhas de ensemble, barcodes e perfis."""
    case_name = scenario["scenario"]
    case_id = scenario["case_id"]
    params = scenario["params"]
    pi_sum = {
        policy: {k: np.zeros(m, dtype=float) for k in k_values}
        for policy in TIE_POLICIES
    }
    topo_sum = np.zeros(m, dtype=float)
    truth_sum = np.zeros(m, dtype=float)
    bar_rows = []
    filtration_rows = []
    realization_rows = []
    replicate_profiles = {
        policy: {k: [] for k in k_values} for policy in TIE_POLICIES
    }
    all_topology_component_counts = []

    for replicate in range(replicate_count):
        h_replicate = {
            policy: {k: np.zeros(m, dtype=float) for k in k_values}
            for policy in TIE_POLICIES
        }
        t_replicate = np.zeros(m, dtype=float)
        truth_replicate = np.zeros(m, dtype=float)
        diagram_distances = []
        perturb_delta = {policy: {k: [] for k in k_values} for policy in TIE_POLICIES}
        perturbed_h_replicate = {
            policy: {k: np.zeros(m, dtype=float) for k in k_values}
            for policy in TIE_POLICIES
        }
        topo_component_counts = []
        topo_long_component_counts = []
        topo_lifespans = []

        for sample_index in range(samples_per_ensemble):
            sample, labels, target_quantile = generate_sample(rng, scenario, m)
            sort_order = np.argsort(sample, kind="mergesort")
            sorted_sample = sample[sort_order]
            sorted_labels = labels[sort_order]
            truth_replicate += sorted_labels.astype(float)
            density = local_density(sorted_sample, epsilon)
            bars, topo_score = h0_superlevel_path(density)
            for threshold, beta_0 in superlevel_component_counts(density):
                filtration_rows.append(
                    {
                        "scenario": case_name,
                        "case_id": case_id,
                        "m": m,
                        "epsilon": epsilon,
                        "replicate": replicate,
                        "sample": sample_index,
                        "threshold": threshold,
                        "beta_0": beta_0,
                    }
                )
            t_replicate += topo_score
            topo_component_counts.append(len(bars))
            topo_lifespans.extend(bar.persistence for bar in bars)
            persistence_cutoff = MIN_PERSISTENCE_FRACTION * float(np.max(density))
            topo_long_component_counts.append(
                sum(bar.persistence >= persistence_cutoff for bar in bars)
            )

            for policy in TIE_POLICIES:
                for k in k_values:
                    h_replicate[policy][k] += select_rank_weights(density, k, policy, rng)

            for bar in bars:
                bar_rows.append(
                    {
                        "scenario": case_name,
                        "case_id": case_id,
                        "m": m,
                        "epsilon": epsilon,
                        "replicate": replicate,
                        "sample": sample_index,
                        "component_id": bar.component_id,
                        "birth_level": bar.birth,
                        "death_level": bar.death,
                        "persistence": bar.persistence,
                        "rank_start": min(bar.members) if bar.members else -1,
                        "rank_end": max(bar.members) if bar.members else -1,
                        "rank_width": len(bar.members),
                        "component_density_mass": int(
                            sum(int(density[rank]) for rank in bar.members)
                        ),
                        "component_density_mass_fraction": float(
                            sum(int(density[rank]) for rank in bar.members)
                            / max(int(density.sum()), 1)
                        ),
                        "peak_rank": bar.peak_rank,
                        "peak_quantile": bar.peak_rank / m,
                        "essential": int(bar.essential),
                    }
                )

            if perturb:
                perturbed = sample + rng.normal(0.0, max(epsilon * 0.02, 1e-8), m)
                perturbed_order = np.argsort(perturbed, kind="mergesort")
                perturbed_sorted = perturbed[perturbed_order]
                perturbed_density = local_density(perturbed_sorted, epsilon)
                perturbed_bars, _ = h0_superlevel_path(perturbed_density)
                diagram_distances.append(diagram_wasserstein1(bars, perturbed_bars))
                for policy in TIE_POLICIES:
                    for k in k_values:
                        changed_weight = select_rank_weights(
                            perturbed_density, k, policy, rng
                        )
                        perturbed_h_replicate[policy][k] += changed_weight

        if perturb:
            for policy in TIE_POLICIES:
                for k in k_values:
                    original_profile = h_replicate[policy][k] / samples_per_ensemble
                    perturbed_profile = (
                        perturbed_h_replicate[policy][k] / samples_per_ensemble
                    )
                    perturb_delta[policy][k].append(
                        float(np.mean(np.abs(original_profile - perturbed_profile)))
                    )

        all_topology_component_counts.extend(topo_component_counts)
        truth_sum += truth_replicate / samples_per_ensemble
        topo_sum += t_replicate / samples_per_ensemble
        topology_average = t_replicate / samples_per_ensemble
        active_fraction = truth_replicate / samples_per_ensemble
        truth_mask = active_fraction >= 0.5

        for policy in TIE_POLICIES:
            for k in k_values:
                profile = h_replicate[policy][k] / samples_per_ensemble
                h_replicate[policy][k] = profile
                pi_sum[policy][k] += profile
                replicate_profiles[policy][k].append(profile.copy())
                metrics = profile_metrics(profile)
                estimated_region = profile >= (0.5 * np.max(profile)) if np.max(profile) else np.zeros(m, bool)
                jaccard = jaccard_similarity(estimated_region, truth_mask)
                t_rho = spearman_correlation(profile, topology_average)
                target_q = target_quantile
                realization_rows.append(
                    {
                        "scenario": case_name,
                        "case_id": case_id,
                        "m": m,
                        "epsilon": epsilon,
                        "replicate": replicate,
                        "K": k,
                        "tie_policy": policy,
                        "p": params.get("p", np.nan),
                        "sigma": params.get("sigma", np.nan),
                        "true_position": params.get("position", np.nan),
                        "true_cluster_quantile": target_q,
                        "JS_selection_uniform": metrics["JS_uniform"],
                        "max_profile": metrics["max_profile"],
                        "peak_rank": metrics["peak_rank"],
                        "peak_quantile": metrics["peak_quantile"],
                        "peak_error": abs(metrics["peak_quantile"] - target_q)
                        if np.isfinite(target_q)
                        else np.nan,
                        "sigma_H": metrics["sigma_H"],
                        "peak_width_fwhm": metrics["peak_width_fwhm"],
                        "n_h0_components": float(np.mean(topo_component_counts)),
                        "n_h0_components_above_5pct_birth": float(
                            np.mean(topo_long_component_counts)
                        ),
                        "max_h0_persistence": max(topo_lifespans, default=0.0),
                        "spearman_selection_topology": t_rho,
                        "selection_region_jaccard": jaccard,
                        "topology_region_jaccard": jaccard_similarity(
                            topology_average >= 0.5 * np.max(topology_average), truth_mask
                        ) if np.max(topology_average) else np.nan,
                        "delta_pi_perturbation": float(np.mean(perturb_delta[policy][k]))
                        if perturb_delta[policy][k]
                        else np.nan,
                        "diagram_wasserstein1_perturbation": float(np.mean(diagram_distances))
                        if diagram_distances
                        else np.nan,
                    }
                )

        # All selection policies preserve selection mass K except include_all.
        for policy in TIE_POLICIES:
            for k in k_values:
                total = float(h_replicate[policy][k].sum())
                replicate_profiles[policy][k][-1] = h_replicate[policy][k].copy()
        # T and truth are accumulated once per sample ensemble, not per K.

    profile_rows = []
    summary_rows = []
    for policy in TIE_POLICIES:
        for k in k_values:
            profile = pi_sum[policy][k] / replicate_count
            topology_profile = topo_sum / replicate_count
            metrics = profile_metrics(profile)
            target_quantile = generate_target_quantile(case_name, params)
            true_mask = truth_sum / replicate_count >= 0.5
            estimated_mask = profile >= 0.5 * np.max(profile) if np.max(profile) else np.zeros(m, bool)
            rho = (
                float(params.get("p", 0.0)) * m / k
                if case_name == "cluster_anomaly"
                else np.nan
            )
            summary_rows.append(
                {
                    "scenario": case_name,
                    "case_id": case_id,
                    "m": m,
                    "epsilon": epsilon,
                    "K": k,
                    "K_over_m": k / m,
                    "tie_policy": policy,
                    "p": params.get("p", np.nan),
                    "sigma": params.get("sigma", np.nan),
                    "true_position": params.get("position", np.nan),
                    "rho_pm_over_K": rho,
                    "JS_selection_uniform": metrics["JS_uniform"],
                    "max_profile": metrics["max_profile"],
                    "peak_rank": metrics["peak_rank"],
                    "peak_quantile": metrics["peak_quantile"],
                    "peak_error": abs(metrics["peak_quantile"] - target_quantile)
                    if np.isfinite(target_quantile)
                    else np.nan,
                    "sigma_H": metrics["sigma_H"],
                    "peak_width_fwhm": metrics["peak_width_fwhm"],
                    "mean_h0_component_count": float(
                        np.mean(all_topology_component_counts)
                    ),
                    "spearman_selection_topology": spearman_correlation(profile, topology_profile),
                    "selection_region_jaccard": jaccard_similarity(estimated_mask, true_mask),
                    "topology_region_jaccard": jaccard_similarity(
                        topology_profile >= 0.5 * np.max(topology_profile), true_mask
                    ) if np.max(topology_profile) else np.nan,
                    "n_replicates": replicate_count,
                    "samples_per_ensemble": samples_per_ensemble,
                }
            )
            for rank in range(m):
                profile_rows.append(
                    {
                        "scenario": case_name,
                        "case_id": case_id,
                        "m": m,
                        "epsilon": epsilon,
                        "K": k,
                        "tie_policy": policy,
                        "rank": rank,
                        "quantile": rank / m,
                        "Pi_hat_K": profile[rank],
                        "T_hat": topology_profile[rank],
                        "true_region_frequency": truth_sum[rank] / replicate_count,
                    }
                )

    return realization_rows, bar_rows, filtration_rows, profile_rows, summary_rows


def generate_target_quantile(scenario_name: str, params: dict) -> float:
    """Quantil-alvo teórico (para misturas anômalas uniformes e cenários base)."""
    if scenario_name == "cluster_anomaly":
        p = float(params["p"])
        q = float(params["position"])
        return (1.0 - p) * q + 0.5 * p if p > 0 else float("nan")
    if scenario_name == "bimodal_separated":
        return 0.25
    if scenario_name == "weighted_mixture":
        return 0.375
    if scenario_name == "normal":
        return 0.5
    return float("nan")


def add_empirical_null_pvalues(realization_rows: list[dict]) -> None:
    """P-values JS versus uniform matched on m, epsilon, K and tie policy."""
    null_groups: dict[tuple, list[float]] = {}
    for row in realization_rows:
        if row["scenario"] == "uniform":
            key = (row["m"], row["epsilon"], row["K"], row["tie_policy"])
            null_groups.setdefault(key, []).append(row["JS_selection_uniform"])

    for row in realization_rows:
        key = (row["m"], row["epsilon"], row["K"], row["tie_policy"])
        null_values = null_groups.get(key, [])
        if row["scenario"] == "uniform":
            null_values = [
                value for index, value in enumerate(null_values)
                if index != row["replicate"]
            ]
        if null_values:
            pvalue = (1 + sum(value >= row["JS_selection_uniform"] for value in null_values)) / (
                len(null_values) + 1
            )
        else:
            pvalue = np.nan
        row["pvalue_JS_uniform_null"] = float(pvalue)
        row["detected_alpha_0.05"] = int(pvalue < ALPHA) if np.isfinite(pvalue) else 0


def aggregate_final(realization_rows: list[dict], summary_rows: list[dict]) -> list[dict]:
    """Agrega potência e estabilidade entre réplicas às linhas de resumo."""
    keyed: dict[tuple, list[dict]] = {}
    for row in realization_rows:
        key = (
            row["scenario"], row["case_id"], row["m"], row["epsilon"],
            row["K"], row["tie_policy"],
        )
        keyed.setdefault(key, []).append(row)

    for summary in summary_rows:
        key = (
            summary["scenario"], summary["case_id"], summary["m"], summary["epsilon"],
            summary["K"], summary["tie_policy"],
        )
        rows = keyed.get(key, [])
        pvalues = [row["pvalue_JS_uniform_null"] for row in rows]
        summary["power_or_type1_rate"] = float(
            np.mean([row["detected_alpha_0.05"] for row in rows])
        ) if rows else np.nan
        finite_pvalues = np.asarray(pvalues, dtype=float)
        summary["pvalue_JS_mean"] = (
            float(np.nanmean(finite_pvalues))
            if np.isfinite(finite_pvalues).any()
            else np.nan
        )
        summary["peak_error_mean"] = float(
            np.nanmean([row["peak_error"] for row in rows])
        ) if rows and np.isfinite([row["peak_error"] for row in rows]).any() else np.nan
        summary["peak_error_max"] = float(
            np.nanmax([row["peak_error"] for row in rows])
        ) if rows and np.isfinite([row["peak_error"] for row in rows]).any() else np.nan
        summary["peak_error_std"] = float(
            np.nanstd([row["peak_error"] for row in rows], ddof=1)
        ) if rows and np.isfinite([row["peak_error"] for row in rows]).sum() > 1 else np.nan
        jaccard_values = np.asarray(
            [row["selection_region_jaccard"] for row in rows], dtype=float
        )
        summary["mean_selection_jaccard"] = (
            float(np.nanmean(jaccard_values))
            if np.isfinite(jaccard_values).any()
            else np.nan
        )
        summary["topology_region_jaccard"] = (
            rows[0]["topology_region_jaccard"] if rows else np.nan
        )
        delta_values = np.asarray(
            [row["delta_pi_perturbation"] for row in rows], dtype=float
        )
        diagram_values = np.asarray(
            [row["diagram_wasserstein1_perturbation"] for row in rows], dtype=float
        )
        summary["mean_perturbation_delta_pi"] = (
            float(np.nanmean(delta_values)) if np.isfinite(delta_values).any() else np.nan
        )
        summary["mean_diagram_wasserstein1"] = (
            float(np.nanmean(diagram_values)) if np.isfinite(diagram_values).any() else np.nan
        )
    return summary_rows


def fit_detectability_models(summary_rows: list[dict]) -> list[dict]:
    """Ajusta regressões logísticas de potência: escala Rho e modelo expandido."""
    anomaly_rows = [
        row for row in summary_rows
        if row["scenario"] == "cluster_anomaly"
        and row.get("n_replicates", 0) > 0
        and np.isfinite(row.get("power_or_type1_rate", np.nan))
        and np.isfinite(row.get("rho_pm_over_K", np.nan))
    ]
    if len(anomaly_rows) < 5:
        return [{"model": "not_fitted", "reason": "insufficient_data"}]

    rho = np.asarray([row["rho_pm_over_K"] for row in anomaly_rows], dtype=float)
    k = np.asarray([row["K"] for row in anomaly_rows], dtype=float)
    m = np.asarray([row["m"] for row in anomaly_rows], dtype=float)
    sigma = np.asarray([row["sigma"] for row in anomaly_rows], dtype=float)
    sigma = np.nan_to_num(sigma, nan=0.01)
    log_rho = np.log(np.maximum(rho, 1e-6))
    log_k = np.log(k)
    log_m = np.log(m)

    designs = {
        "rho_only": np.column_stack((np.ones(len(rho)), log_rho)),
        "expanded": np.column_stack(
            (np.ones(len(rho)), log_rho, log_k, log_m, sigma, log_rho * log_k, log_rho * sigma)
        ),
    }
    names = {
        "rho_only": ("intercept", "log_rho"),
        "expanded": (
            "intercept", "log_rho", "log_K", "log_m", "sigma",
            "log_rho_x_log_K", "log_rho_x_sigma",
        ),
    }
    output = []
    for model_name, design in designs.items():
        feature_mean = np.zeros(design.shape[1])
        feature_scale = np.ones(design.shape[1])
        if model_name == "expanded":
            feature_mean[1:] = design[:, 1:].mean(axis=0)
            feature_scale[1:] = design[:, 1:].std(axis=0)
            feature_scale[feature_scale == 0] = 1.0
            design = (design - feature_mean) / feature_scale
            design[:, 0] = 1.0

        trials = np.asarray([row["n_replicates"] for row in anomaly_rows], dtype=float)
        successes = np.asarray(
            [row["power_or_type1_rate"] * row["n_replicates"] for row in anomaly_rows],
            dtype=float,
        )

        def objective(beta):
            eta = design @ beta
            likelihood = np.sum(trials * np.logaddexp(0.0, eta) - successes * eta)
            penalty = 0.5 * 1e-4 * np.sum(beta[1:] ** 2)
            return float(likelihood + penalty)

        fit = minimize(objective, np.zeros(design.shape[1]), method="BFGS")
        beta = fit.x
        eta = design @ beta
        fitted_probabilities = expit(eta)
        null_rate = np.clip(successes.sum() / trials.sum(), 1e-9, 1 - 1e-9)
        null_eta = np.log(null_rate / (1 - null_rate))
        null_nll = np.sum(trials * np.logaddexp(0.0, null_eta) - successes * null_eta)
        fitted_nll = objective(beta)
        pseudo_r2 = 1.0 - fitted_nll / null_nll if null_nll else np.nan
        for index, coefficient in enumerate(beta):
            output.append(
                {
                    "model": model_name,
                    "term": names[model_name][index],
                    "coefficient": float(coefficient),
                    "feature_mean": float(feature_mean[index]),
                    "feature_scale": float(feature_scale[index]),
                    "n_conditions": len(anomaly_rows),
                    "converged": bool(fit.success),
                    "pseudo_r2_vs_intercept": float(pseudo_r2),
                    "mean_fitted_power": float(np.mean(fitted_probabilities)),
                }
            )
    return output


def plot_results(output_dir: Path, summary_rows: list[dict], profile_rows: list[dict]) -> None:
    """Salva grafici de potência, escala, H0 e perfis de um caso ilustrativo."""
    if not summary_rows:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_power, ax_scale, ax_width, ax_type1, ax_sigma, ax_jaccard = axes.flat

    anomaly_rows = [
        row for row in summary_rows
        if row["scenario"] == "cluster_anomaly" and row["tie_policy"] == "fractional"
    ]
    for m in sorted({row["m"] for row in anomaly_rows}):
        rows = sorted(
            [
                row for row in anomaly_rows
                if row["m"] == m
                and row["case_id"].startswith("p_sweep_")
                and np.isclose(row["epsilon"], 0.05)
            ],
            key=lambda row: row["p"] if np.isfinite(row["p"]) else -1,
        )
        if rows:
            ax_power.plot([row["p"] for row in rows], [row["power_or_type1_rate"] for row in rows],
                          "o-", label=f"m={m}")
    ax_power.set_xlabel("Fração anômala p")
    ax_power.set_ylabel("Potência empírica (p_JS < 0,05)")
    ax_power.set_ylim(-0.05, 1.05)
    ax_power.set_title("Potência do cluster anômalo")
    if ax_power.lines:
        ax_power.legend(fontsize="small")
    ax_power.grid(True, alpha=0.25)

    for k in sorted({row["K"] for row in anomaly_rows}):
        rows = sorted(
            [
                row for row in anomaly_rows
                if row["K"] == k
                and row["case_id"].startswith("p_sweep_")
                and np.isfinite(row["rho_pm_over_K"])
                and row["rho_pm_over_K"] > 0
            ],
            key=lambda row: row["rho_pm_over_K"],
        )
        if rows:
            ax_scale.scatter(
                [row["rho_pm_over_K"] for row in rows],
                [row["power_or_type1_rate"] for row in rows],
                s=18,
                alpha=0.6,
                label=f"K={k}",
            )
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("ρ = p·m/K")
    ax_scale.set_ylabel("Potência")
    ax_scale.set_ylim(-0.05, 1.05)
    ax_scale.set_title("Transição de detectabilidade")
    if ax_scale.collections:
        ax_scale.legend(fontsize="small")
    ax_scale.grid(True, which="both", alpha=0.25)

    for policy in TIE_POLICIES:
        rows = [
            row for row in summary_rows
            if row["scenario"] == "cluster_anomaly"
            and row["tie_policy"] == policy
            and row["m"] == 512
            and np.isclose(row["p"], 0.02)
            and np.isclose(row["epsilon"], 0.05)
        ]
        rows.sort(key=lambda row: row["K"])
        if rows:
            ax_width.plot([row["K"] for row in rows], [row["sigma_H"] for row in rows],
                          "o-", label=policy)
    ax_width.set_xlabel("K")
    ax_width.set_ylabel("σ_H (ranks)")
    ax_width.set_title("Largura versus K")
    if ax_width.lines:
        ax_width.legend(fontsize="small")
    ax_width.grid(True, alpha=0.25)

    uniform_rows = [
        row for row in summary_rows
        if row["scenario"] == "uniform" and row["tie_policy"] == "fractional"
    ]
    if uniform_rows:
        labels = [f"m={row['m']},K={row['K']}" for row in uniform_rows[:12]]
        rates = [row["power_or_type1_rate"] for row in uniform_rows[:12]]
        ax_type1.bar(np.arange(len(rates)), rates)
        ax_type1.set_xticks(np.arange(len(rates)), labels, rotation=70, ha="right")
    ax_type1.axhline(ALPHA, color="red", linestyle="--", linewidth=1)
    ax_type1.set_ylabel("Taxa de rejeição sob U(0,1)")
    ax_type1.set_title("Controle aproximado do erro tipo I")
    ax_type1.grid(True, axis="y", alpha=0.25)

    sigma_rows = [
        row for row in summary_rows
        if row["scenario"] == "cluster_anomaly"
        and row["case_id"].startswith("sigma_sweep_")
        and row["tie_policy"] == "fractional"
        and row["m"] == 512
        and row["K"] == 20
        and np.isclose(row["epsilon"], 0.05)
    ]
    sigma_rows.sort(key=lambda row: row["sigma"])
    if sigma_rows:
        ax_sigma.plot(
            [row["sigma"] for row in sigma_rows],
            [row["power_or_type1_rate"] for row in sigma_rows],
            "o-",
        )
    ax_sigma.set_xlabel("Desvio-padrão do cluster σ")
    ax_sigma.set_ylabel("Potência")
    ax_sigma.set_ylim(-0.05, 1.05)
    ax_sigma.set_title("Efeito da compactação (m=512, K=20)")
    ax_sigma.grid(True, alpha=0.25)

    jaccard_rows = [
        row for row in summary_rows
        if row["scenario"] == "cluster_anomaly"
        and row["case_id"].startswith("p_sweep_")
        and row["tie_policy"] == "fractional"
        and row["m"] == 512
        and row["K"] == 20
        and np.isclose(row["epsilon"], 0.05)
    ]
    jaccard_rows.sort(key=lambda row: row["p"])
    if jaccard_rows:
        ax_jaccard.plot(
            [row["p"] for row in jaccard_rows],
            [row["selection_region_jaccard"] for row in jaccard_rows],
            "o-",
            label="seleção",
        )
        ax_jaccard.plot(
            [row["p"] for row in jaccard_rows],
            [row["topology_region_jaccard"] for row in jaccard_rows],
            "s--",
            label="topologia",
        )
    ax_jaccard.set_xlabel("Fração anômala p")
    ax_jaccard.set_ylabel("Jaccard com região verdadeira")
    ax_jaccard.set_ylim(-0.05, 1.05)
    ax_jaccard.set_title("Sobreposição da região detectada")
    if ax_jaccard.lines:
        ax_jaccard.legend(fontsize="small")
    ax_jaccard.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "topology_detection_summary.png", dpi=150)
    plt.close(fig)

    # Perfil ilustrativo da anomalia, se estiver presente na execução.
    selected = [
        row for row in profile_rows
        if row["scenario"] == "cluster_anomaly"
        and row["tie_policy"] == "fractional"
        and row["K"] == 20
        and row["m"] == 512
        and np.isclose(row["epsilon"], 0.05)
    ]
    if selected:
        by_case: dict[str, list[dict]] = {}
        for row in selected:
            by_case.setdefault(row["case_id"], []).append(row)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for case_id, rows in list(by_case.items())[:8]:
            rows.sort(key=lambda row: row["rank"])
            axes[0].plot(
                [r["quantile"] for r in rows],
                [r["Pi_hat_K"] for r in rows],
                label=case_id,
            )
            axes[1].plot(
                [r["quantile"] for r in rows],
                [r["T_hat"] for r in rows],
                label=case_id,
            )
        axes[0].set_ylabel("Π̂(i,K): frequência de seleção")
        axes[0].set_title("Persistência de seleção")
        axes[1].set_xlabel("Quantil de rank")
        axes[1].set_ylabel("T̂(i): persistência H0 finita")
        axes[1].set_title("Perfil topológico")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / "topology_rank_profiles.png", dpi=150)
        plt.close(fig)


def plot_beta0_example(output_dir: Path, filtration_rows: list[dict]) -> None:
    """Plota beta0(lambda) de uma amostra uniforme e de uma bimodal."""
    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for scenario, label in (("uniform", "uniforme"), ("bimodal_separated", "bimodal")):
        rows = [
            row for row in filtration_rows
            if row["scenario"] == scenario and row["replicate"] == 0 and row["sample"] == 0
        ]
        if rows:
            rows.sort(key=lambda row: row["threshold"], reverse=True)
            ax.step(
                [row["threshold"] for row in rows],
                [row["beta_0"] for row in rows],
                where="post",
                label=label,
            )
            found = True
    if found:
        ax.set_xlabel("Limiar λ")
        ax.set_ylabel("β₀(λ): número de componentes")
        ax.set_title("Componentes conexas na filtração de supernível")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "topology_beta0_filtration.png", dpi=150)
    plt.close(fig)


def plot_barcode_example(output_dir: Path, barcode_rows: list[dict]) -> None:
    """Plota um diagrama H0 de uma realização bimodal bem separada."""
    selected = [
        row for row in barcode_rows
        if row["scenario"] == "bimodal_separated"
        and row["replicate"] == 0
        and row["sample"] == 0
    ]
    if not selected:
        return
    births = np.asarray([row["birth_level"] for row in selected], dtype=float)
    deaths = np.asarray([row["death_level"] for row in selected], dtype=float)
    persistence = np.asarray([row["persistence"] for row in selected], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    finite = np.asarray([not bool(row["essential"]) for row in selected])
    ax.scatter(
        births[finite], deaths[finite], c=persistence[finite], cmap="viridis",
        s=24 + 8 * persistence[finite], label="classes finitas",
    )
    if np.any(~finite):
        ax.scatter(births[~finite], deaths[~finite], marker="*", s=100,
                   color="crimson", label="classe essencial")
    lim = max(float(np.max(births)), float(np.max(deaths)), 1.0)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="diagonal")
    ax.set_xlabel("Nascimento λ")
    ax.set_ylabel("Fusão λ")
    ax.set_title("Barcode H0: mistura bimodal separada")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "topology_h0_barcode.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Suite curta para validar o fluxo.")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--samples-per-ensemble", type=int, default=30)
    parser.add_argument("--output-dir", default="rank_persistence_topology_results")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--skip-perturbation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        m_values = (64, 128)
        k_values = (5, 10, 20)
        replicate_count = 3
        samples_per_ensemble = 5
    else:
        m_values = FULL_M_VALUES
        k_values = FULL_K_VALUES
        replicate_count = args.replicates
        samples_per_ensemble = args.samples_per_ensemble

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(m_values, args.quick)
    rng = np.random.default_rng(args.seed)

    total_cases = sum(
        len(scenario["m_values"]) * len(scenario["epsilons"])
        for scenario in scenarios
    )
    print(
        f"Casos={total_cases}; réplicas/caso={replicate_count}; "
        f"amostras por ensemble={samples_per_ensemble}; m={m_values}; K={k_values}"
    )

    all_realizations = []
    all_barcodes = []
    all_filtration = []
    all_profiles = []
    all_summaries = []
    completed = 0

    for scenario in scenarios:
        for m in scenario["m_values"]:
            current_k = tuple(k for k in k_values if k <= m)
            for epsilon in scenario["epsilons"]:
                result = run_case(
                    rng,
                    scenario,
                    int(m),
                    float(epsilon),
                    current_k,
                    replicate_count,
                    samples_per_ensemble,
                    perturb=not args.skip_perturbation,
                )
                (
                    realization_rows,
                    barcode_rows,
                    filtration_rows,
                    profile_rows,
                    summary_rows,
                ) = result
                all_realizations.extend(realization_rows)
                all_barcodes.extend(barcode_rows)
                all_filtration.extend(filtration_rows)
                all_profiles.extend(profile_rows)
                all_summaries.extend(summary_rows)
                completed += 1
                print(
                    f"[{completed}/{total_cases}] {scenario['case_id']} "
                    f"m={m}, eps={epsilon:g}"
                )

    add_empirical_null_pvalues(all_realizations)
    all_summaries = aggregate_final(all_realizations, all_summaries)
    save_dict_rows(output_dir / "realizations.csv", all_realizations)
    save_dict_rows(output_dir / "h0_barcodes.csv", all_barcodes)
    save_dict_rows(output_dir / "beta0_filtration.csv", all_filtration)
    save_dict_rows(output_dir / "rank_profiles.csv", all_profiles)
    save_dict_rows(output_dir / "summary.csv", all_summaries)
    model_rows = fit_detectability_models(all_summaries)
    save_dict_rows(output_dir / "detectability_logistic.csv", model_rows)
    plot_results(output_dir, all_summaries, all_profiles)
    plot_barcode_example(output_dir, all_barcodes)
    plot_beta0_example(output_dir, all_filtration)
    print(f"Resultados e gráficos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
