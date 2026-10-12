#!/usr/bin/env python3

"""Run the controlled raw/Wavelet/STFT/CTM SVM comparison."""

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.data import load_mitdb
from experiments.evaluate import compare_representations
from experiments.representations import extract_representations


DEFAULT_RECORDS = (
    "100",
    "101",
    "103",
    "105",
    "106",
    "108",
    "109",
    "111",
    "112",
    "113",
    "115",
    "116",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
)



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        nargs="+",
        default=DEFAULT_RECORDS,
        help="MIT-BIH record IDs (default: a fixed 17-record subset).",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--window-samples", type=int, default=360)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--ctm-span", type=int, default=32)
    parser.add_argument(
        "--ctm-measure",
        choices=("amplitude", "energy"),
        default="amplitude",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/comparison.csv"),
    )
    return parser.parse_args()



def build_feature_sets(dataset, args):
    extracted = [
        extract_representations(
            signal,
            ctm_span=args.ctm_span,
            ctm_measure=args.ctm_measure,
        )
        for signal in dataset.signals
    ]
    names = tuple(extracted[0])
    return {
        name: np.vstack([window[name] for window in extracted])
        for name in names
    }



def write_results(path: Path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "representation",
        "fold",
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "accuracy",
        "best_params",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: getattr(result, field) for field in fields}
            row["best_params"] = repr(row["best_params"])
            writer.writerow(row)



def main():
    args = parse_args()
    dataset = load_mitdb(
        args.records,
        window_samples=args.window_samples,
        channel=args.channel,
        data_dir=args.data_dir,
    )
    feature_sets = build_feature_sets(dataset, args)
    results = compare_representations(
        feature_sets,
        dataset.labels,
        dataset.groups,
        n_splits=args.outer_folds,
        inner_splits=args.inner_folds,
        random_state=args.seed,
    )
    write_results(args.output, results)
    print(f"Janelas: {len(dataset.labels)}")
    print(f"Registros: {len(np.unique(dataset.groups))}")
    print(f"Resultados: {args.output}")


if __name__ == "__main__":
    main()
