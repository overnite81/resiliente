#!/usr/bin/env python3
"""Small CPU-first ResNet-18 pilot for STFT, CWT scalogram, and CTM."""
from __future__ import annotations
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from experiments.data import load_mitdb
from experiments.image_representations import REPRESENTATIONS, encode_image

DEFAULT_RECORDS = ("100", "101", "103", "105")


def parse_args():
    long_records = (
        "100", "101", "103", "105", "106", "108", "109", "111", "112",
        "113", "115", "116", "118", "119", "121", "122", "123", "124",
    )
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--profile", choices=("pilot", "spot-arm64-long"), default="pilot")
    profile, _ = bootstrap.parse_known_args()
    if profile.profile == "spot-arm64-long":
        defaults = {
            "records": long_records, "epochs": 12,
            "output": Path("experiments/results/resnet_spot_arm64_long.csv"),
            "time_log": Path("experiments/results/resnet_spot_arm64_long_timing.csv"),
        }
    else:
        defaults = {
            "records": DEFAULT_RECORDS, "epochs": 2,
            "output": Path("experiments/results/resnet_pilot.csv"),
            "time_log": Path("experiments/results/resnet_pilot_timing.csv"),
        }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "spot-arm64-long"), default="pilot")
    parser.add_argument("--records", nargs="+", default=defaults["records"])
    parser.add_argument("--test-record", default="105")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--window-samples", type=int, default=360)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--ctm-span", type=int, default=48)
    parser.add_argument("--ctm-measure", choices=("amplitude", "energy"), default="amplitude")
    parser.add_argument("--limit-per-class-record", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--time-log", type=Path, default=defaults["time_log"])
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--resume", action="store_true",
                        help="no perfil longo, pula representações já gravadas nos CSVs")
    return parser.parse_args()


def cap_each_class(dataset, indices, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for record in np.unique(dataset.record_ids[indices]):
        record_indices = indices[dataset.record_ids[indices] == record]
        for label in (0, 1):
            candidates = record_indices[dataset.labels[record_indices] == label]
            if candidates.size > per_class:
                candidates = rng.choice(candidates, size=per_class, replace=False)
            selected.extend(candidates.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def encode_batch(dataset, indices, representation, args):
    started = time.perf_counter()
    images = np.stack([
        encode_image(dataset.signals[i], representation, image_size=args.image_size,
                     ctm_span=args.ctm_span, ctm_measure=args.ctm_measure)
        for i in indices
    ])
    return images, time.perf_counter() - started


def train_and_score(images_train, y_train, images_test, y_test, args):
    import torch
    import torchvision
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    weights = torchvision.models.ResNet18_Weights.DEFAULT if args.weights == "imagenet" else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    counts = np.bincount(y_train, minlength=2)
    class_weights = torch.tensor(len(y_train) / np.maximum(2 * counts, 1), dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def tensor_data(images, labels):
        x = torch.from_numpy(images[:, None, :, :]).float().repeat(1, 3, 1, 1)
        if args.weights == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
        else:
            x = (x - 0.5) / 0.5
        return TensorDataset(x, torch.from_numpy(labels.astype(np.int64)))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(tensor_data(images_train, y_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, generator=generator)
    test_loader = DataLoader(tensor_data(images_test, y_test), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    epoch_seconds = []
    fit_started = time.perf_counter()
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        for batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
        epoch_seconds.append(time.perf_counter() - started)
        print(f"  epoch {epoch + 1}/{args.epochs}: {epoch_seconds[-1]:.1f}s; loss={loss.item():.4f}", flush=True)

    model.eval()
    pred, score, truth = [], [], []
    with torch.no_grad():
        for batch, targets in test_loader:
            logits = model(batch)
            pred.extend(logits.argmax(1).numpy())
            score.extend(torch.softmax(logits, 1)[:, 1].numpy())
            truth.extend(targets.numpy())
    fit_seconds = time.perf_counter() - fit_started
    truth, pred, score = np.asarray(truth), np.asarray(pred), np.asarray(score)
    metrics = {
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "f1": f1_score(truth, pred, zero_division=0),
        "roc_auc": roc_auc_score(truth, score) if np.unique(truth).size == 2 else float("nan"),
        "average_precision": average_precision_score(truth, score) if np.unique(truth).size == 2 else float("nan"),
    }
    return metrics, fit_seconds, epoch_seconds


def write_csv_row(path, row, append):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not append:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def completed_representations(path):
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {row["representation"] for row in csv.DictReader(stream)}


def instance_uptime_seconds():
    """Return Linux host uptime, or NaN on non-Linux systems."""
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return float("nan")


def main():
    args = parse_args()
    if args.test_record not in args.records:
        raise SystemExit("--test-record precisa estar listado em --records.")
    if args.epochs < 1 or args.limit_per_class_record < 1 or args.threads < 1:
        raise SystemExit("epochs, limite por classe e threads devem ser positivos.")
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            "PyTorch e torchvision são necessários; consulte experiments/README.md."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    instance_uptime_start = instance_uptime_seconds()
    torch.set_num_threads(args.threads)
    print(f"Perfil={args.profile}; PyTorch {torch.__version__}; torchvision {torchvision.__version__}; threads={args.threads}")
    load_started = time.perf_counter()
    dataset = load_mitdb(args.records, window_samples=args.window_samples,
                         channel=args.channel, data_dir=args.data_dir)
    load_seconds = time.perf_counter() - load_started

    train_indices = np.flatnonzero(dataset.record_ids != args.test_record)
    test_indices = np.flatnonzero(dataset.record_ids == args.test_record)
    train_indices = cap_each_class(dataset, train_indices, args.limit_per_class_record, args.seed)
    test_indices = cap_each_class(dataset, test_indices, args.limit_per_class_record, args.seed + 1)
    if not len(train_indices) or not len(test_indices):
        raise SystemExit("Sem exemplos após divisão; confira registros/test-record.")
    for name, indices in (("treino", train_indices), ("teste", test_indices)):
        if np.unique(dataset.labels[indices]).size != 2:
            raise SystemExit(f"A divisão de {name} precisa conter as duas classes.")

    print(f"Dados carregados em {load_seconds:.1f}s; treino={len(train_indices)}, teste={len(test_indices)}")
    if args.resume and args.profile != "spot-arm64-long":
        raise SystemExit("--resume só está disponível no perfil spot-arm64-long.")
    if args.resume:
        if args.output.exists() != args.time_log.exists():
            raise SystemExit("Para retomar, os CSVs de resultados e tempos precisam existir juntos.")
        done = completed_representations(args.output) & completed_representations(args.time_log)
        write_first = not args.output.exists()
    else:
        done = set()
        args.output.unlink(missing_ok=True)
        args.time_log.unlink(missing_ok=True)
        write_first = True
    for representation in args.representations:
        if representation in done:
            print(f"\n[{representation}] já registrada; pulando (retomada)", flush=True)
            continue
        print(f"\n[{representation}] codificando imagens", flush=True)
        train_images, train_encode_seconds = encode_batch(dataset, train_indices, representation, args)
        test_images, test_encode_seconds = encode_batch(dataset, test_indices, representation, args)
        started = time.perf_counter()
        metrics, fit_seconds, epoch_seconds = train_and_score(
            train_images, dataset.labels[train_indices],
            test_images, dataset.labels[test_indices], args,
        )
        total_seconds = time.perf_counter() - started
        row = {
            "representation": representation,
            "train_records": ";".join(sorted(set(dataset.record_ids[train_indices]))),
            "test_record": args.test_record,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "image_size": args.image_size,
            "ctm_span": args.ctm_span,
            "ctm_measure": args.ctm_measure,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "limit_per_class_record": args.limit_per_class_record,
            "profile": args.profile,
            "weights": args.weights,
            "seed": args.seed,
            **metrics,
        }
        write_csv_row(args.output, row, append=not write_first)
        timing_row = {
            "representation": representation,
            "data_load_seconds": round(load_seconds, 3),
            "train_encode_seconds": round(train_encode_seconds, 3),
            "test_encode_seconds": round(test_encode_seconds, 3),
            "epoch_seconds_mean": round(float(np.mean(epoch_seconds)), 3),
            "fit_and_inference_seconds": round(fit_seconds, 3),
            "wall_seconds": round(total_seconds + train_encode_seconds + test_encode_seconds, 3),
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "epochs": args.epochs,
            "threads": args.threads,
            "profile": args.profile,
            "seed": args.seed,
            "instance_uptime_start_seconds": round(instance_uptime_start, 1),
            "instance_uptime_end_seconds": round(instance_uptime_seconds(), 1),
        }
        write_csv_row(args.time_log, timing_row, append=not write_first)
        write_first = False
        print("  métricas: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print(f"  encode={train_encode_seconds + test_encode_seconds:.1f}s; total={total_seconds:.1f}s")
        del train_images, test_images

    print(f"\nResultados: {args.output}\nTempos: {args.time_log}")


if __name__ == "__main__":
    main()
