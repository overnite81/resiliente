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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--test-record", default="105")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--window-samples", type=int, default=360)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--ctm-span", type=int, default=48)
    parser.add_argument("--ctm-measure", choices=("amplitude", "energy"), default="amplitude")
    parser.add_argument("--limit-per-class-record", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/resnet_pilot.csv"))
    parser.add_argument("--time-log", type=Path, default=Path("experiments/results/resnet_pilot_timing.csv"))
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
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

    #torch.set_num_threads(args.threads)
    torch.set_num_threads(1)
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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    #torch.set_num_threads(args.threads)
    torch.set_num_threads(1)
    print(f"PyTorch {torch.__version__}; torchvision {torchvision.__version__}; threads={args.threads}")
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
    result_rows, timing_rows = [], []
    for representation in args.representations:
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
            "epochs": args.epochs,
            "weights": args.weights,
            "seed": args.seed,
            **metrics,
        }
        result_rows.append(row)
        timing_rows.append({
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
        })
        print("  métricas: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print(f"  encode={train_encode_seconds + test_encode_seconds:.1f}s; total={total_seconds:.1f}s")
        del train_images, test_images

    write_csv(args.output, result_rows)
    write_csv(args.time_log, timing_rows)
    print(f"\nResultados: {args.output}\nTempos: {args.time_log}")


if __name__ == "__main__":
    main()
