#!/usr/bin/env python3
"""Bootstrap summaries for the fixed-composition ranks/permutations ensembles."""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_groups(path, n_col, metric_columns, normalizers=None):
    groups = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {n_col, *metric_columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        for row in reader:
            try:
                n = int(row[n_col])
                values = [float(row[col]) for col in metric_columns]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid row {row}") from exc
            if not np.all(np.isfinite(values)):
                continue
            groups.setdefault(n, [[] for _ in metric_columns])
            for index, (bucket, value) in enumerate(zip(groups[n], values)):
                normalizer = normalizers[index] if normalizers else None
                bucket.append(value / normalizer(n) if normalizer else value)
    return {n: [np.asarray(v, dtype=float) for v in columns]
            for n, columns in groups.items()}


def bootstrap_moments(values, count, rng, batch_size=128):
    """Return point estimates and percentile bootstrap intervals for SD and skew."""
    if values.size < 3:
        raise ValueError("at least three observations are required per group")
    centered = values - values.mean()
    sd = values.std(ddof=1)
    skew = np.mean(centered ** 3) / np.mean(centered ** 2) ** 1.5
    boot_sd = np.empty(count)
    boot_skew = np.empty(count)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        sample = values[rng.integers(0, values.size, size=(stop - start, values.size))]
        means = sample.mean(axis=1, keepdims=True)
        delta = sample - means
        m2 = np.mean(delta ** 2, axis=1)
        boot_sd[start:stop] = np.sqrt(m2 * values.size / (values.size - 1))
        boot_skew[start:stop] = np.mean(delta ** 3, axis=1) / np.maximum(m2, 1e-300) ** 1.5
    return sd, skew, boot_sd, boot_skew


def ci(values, alpha):
    return np.quantile(values, [alpha / 2, 1 - alpha / 2])


def analyze(input_name, groups, labels, bootstrap_count, alpha, rng, summary_rows, scaling_rows):
    ns = sorted(groups)
    for metric_index, metric in enumerate(labels):
        boot_sd_by_n = {}
        for n in ns:
            values = groups[n][metric_index]
            sd, skew, boot_sd, boot_skew = bootstrap_moments(values, bootstrap_count, rng)
            sd_lo, sd_hi = ci(boot_sd, alpha)
            skew_lo, skew_hi = ci(boot_skew, alpha)
            boot_sd_by_n[n] = boot_sd
            summary_rows.append({
                "source": input_name, "metric": metric, "n": n, "count": values.size,
                "mean": values.mean(), "sd": sd, "sd_ci_low": sd_lo, "sd_ci_high": sd_hi,
                "skewness": skew, "skew_ci_low": skew_lo, "skew_ci_high": skew_hi,
                "bootstrap_replicates": bootstrap_count, "confidence": 1 - alpha,
            })

        if len(ns) >= 2:
            log_n = np.log(np.asarray(ns, dtype=float))
            boot_log_sd = np.column_stack([np.log(boot_sd_by_n[n]) for n in ns])
            center_x = log_n - log_n.mean()
            slope_boot = boot_log_sd @ center_x / np.sum(center_x ** 2)
            point_sd = np.asarray([groups[n][metric_index].std(ddof=1) for n in ns])
            point_slope = np.polyfit(log_n, np.log(point_sd), 1)[0]
            low, high = ci(slope_boot, alpha)
            scaling_rows.append({
                "source": input_name, "metric": metric, "fit": "all_n", "n1": ns[0], "n2": ns[-1],
                "chi": point_slope, "chi_ci_low": low, "chi_ci_high": high,
                "bootstrap_replicates": bootstrap_count, "confidence": 1 - alpha,
            })
            for n1, n2 in zip(ns[:-1], ns[1:]):
                boot_chi = np.log(boot_sd_by_n[n2] / boot_sd_by_n[n1]) / np.log(n2 / n1)
                low, high = ci(boot_chi, alpha)
                scaling_rows.append({
                    "source": input_name, "metric": metric, "fit": "local", "n1": n1, "n2": n2,
                    "chi": np.log(point_sd[ns.index(n2)] / point_sd[ns.index(n1)]) / np.log(n2 / n1),
                    "chi_ci_low": low, "chi_ci_high": high,
                    "bootstrap_replicates": bootstrap_count, "confidence": 1 - alpha,
                })


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=Path, default=Path("results/mechanism/permutations.csv"))
    parser.add_argument("--ranks", type=Path, default=Path("results/mechanism/ranks.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/mechanism/order-summary"))
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260325)
    args = parser.parse_args()
    if args.bootstrap < 100 or not 0 < args.confidence < 1:
        parser.error("--bootstrap must be >=100 and --confidence must be between 0 and 1")
    for path in (args.permutations, args.ranks):
        if not path.is_file():
            parser.error(f"input file not found: {path}")

    rng = np.random.default_rng(args.seed)
    summaries, scalings = [], []
    permutations = read_groups(args.permutations, "n", ["dtw_perm", "diagonal_perm"],
                               normalizers=[lambda n: n - 1, None])
    ranks = read_groups(args.ranks, "n", ["rank_dtw", "raw_dtw", "diag_rank"])
    analyze("permutations", permutations, ["dtw_perm_per_(n-1)", "diagonal_perm"],
            args.bootstrap, 1 - args.confidence, rng, summaries, scalings)
    analyze("ranks", ranks, ["rank_dtw", "raw_dtw", "diag_rank"],
            args.bootstrap, 1 - args.confidence, rng, summaries, scalings)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "moments.csv", summaries,
              ["source", "metric", "n", "count", "mean", "sd", "sd_ci_low", "sd_ci_high",
               "skewness", "skew_ci_low", "skew_ci_high", "bootstrap_replicates", "confidence"])
    write_csv(args.out / "scaling.csv", scalings,
              ["source", "metric", "fit", "n1", "n2", "chi", "chi_ci_low", "chi_ci_high",
               "bootstrap_replicates", "confidence"])
    print(f"Wrote {args.out / 'moments.csv'} and {args.out / 'scaling.csv'}")


if __name__ == "__main__":
    main()
