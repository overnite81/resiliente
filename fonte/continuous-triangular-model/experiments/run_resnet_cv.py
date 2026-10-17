#!/usr/bin/env python3
"""Five-fold, record-grouped comparison of image representations and CNN backbones."""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from experiments.data import load_mitdb
from experiments.image_representations import REPRESENTATIONS, encode_image

DEFAULT_RECORDS = (
    "100", "101", "103", "105", "106", "108", "109", "111", "112",
    "113", "115", "116", "118", "119", "121", "122", "123", "124",
)
METRICS = ("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision")
MODELS = ("resnet18", "resnet34", "efficientnet_b0")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit-per-class-record", type=int, default=64)
    parser.add_argument("--window-samples", type=int, default=360)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--ctm-span", type=int, default=48)
    parser.add_argument("--ctm-measure", choices=("amplitude", "energy"), default="amplitude")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS,
                        default=list(REPRESENTATIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS),
                        help="backbones a comparar (padrão: ResNet-18, ResNet-34 e EfficientNet-B0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results/cnn_cv5"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("experiments/results/cnn_cv5_checkpoints"))
    parser.add_argument("--resume", action="store_true",
                        help="retoma folds/épocas a partir dos CSVs e checkpoints existentes")
    return parser.parse_args()


def cap_each_class(dataset, per_class: int, seed: int) -> np.ndarray:
    """Cap each label independently within each record, preserving group split."""
    rng = np.random.default_rng(seed)
    selected = []
    for record in np.unique(dataset.record_ids):
        record_indices = np.flatnonzero(dataset.record_ids == record)
        for label in (0, 1):
            candidates = record_indices[dataset.labels[record_indices] == label]
            if candidates.size > per_class:
                candidates = rng.choice(candidates, size=per_class, replace=False)
            selected.extend(candidates.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def make_grouped_folds(labels, groups, n_splits=5, seed=42):
    """Return stratified folds whose record groups never cross train/validation."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"São necessários ao menos {n_splits} registros; recebidos {len(unique_groups)}.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError(f"Vazamento de registro detectado no fold {fold}.")
        if np.unique(labels[train_idx]).size != 2 or np.unique(labels[valid_idx]).size != 2:
            raise ValueError(
                f"Fold {fold} não contém as duas classes em treino e validação. "
                "Use mais registros ou ajuste --seed/--records."
            )
    return splits


def summarize_folds(rows, confidence=0.95):
    """Fold-level mean, sample SD, and two-sided Student-t confidence interval."""
    summaries = []
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        count = int(values.size)
        if count == 0:
            mean = std = low = high = float("nan")
        else:
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if count > 1 else float("nan")
            if count > 1:
                half_width = float(student_t.ppf((1 + confidence) / 2, count - 1) * std / np.sqrt(count))
                low, high = mean - half_width, mean + half_width
            else:
                low = high = float("nan")
        summaries.append({
            "metric": metric,
            "n_folds": count,
            "mean": mean,
            "std": std,
            "ci95_low": low,
            "ci95_high": high,
        })
    return summaries


def encode_dataset(dataset, indices, representation, args):
    started = time.perf_counter()
    images = np.stack([
        encode_image(
            dataset.signals[index], representation,
            image_size=args.image_size,
            ctm_span=args.ctm_span,
            ctm_measure=args.ctm_measure,
        )
        for index in indices
    ])
    return images, time.perf_counter() - started


def _atomic_torch_save(torch, payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_model(torch, torchvision, model_name, weights):
    """Create a supported torchvision backbone with a two-class output head."""
    weight_enums = {
        "resnet18": torchvision.models.ResNet18_Weights,
        "resnet34": torchvision.models.ResNet34_Weights,
        "efficientnet_b0": torchvision.models.EfficientNet_B0_Weights,
    }
    constructors = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "efficientnet_b0": torchvision.models.efficientnet_b0,
    }
    if model_name not in constructors:
        raise ValueError(f"Backbone não suportado: {model_name}")
    model_weights = weight_enums[model_name].DEFAULT if weights == "imagenet" else None
    model = constructors[model_name](weights=model_weights)
    if model_name.startswith("resnet"):
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    else:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    return model


def fit_fold(images, labels, train_idx, valid_idx, args, checkpoint_path, fold_seed, model_name):
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset, TensorDataset

    torch.set_num_threads(args.threads)
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    random.seed(fold_seed)

    x = torch.from_numpy(images[:, None, :, :]).float().repeat(1, 3, 1, 1)
    if args.weights == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
    else:
        x = (x - 0.5) / 0.5
    tensor_labels = torch.from_numpy(labels.astype(np.int64))
    dataset = TensorDataset(x, tensor_labels)

    model = build_model(torch, torchvision, model_name, args.weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    counts = np.bincount(labels[train_idx], minlength=2)
    class_weights = torch.tensor(len(train_idx) / np.maximum(2 * counts, 1), dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=args.batch_size,
        shuffle=True, num_workers=0, generator=generator,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    completed_epoch = 0
    epoch_seconds = []
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        completed_epoch = int(state["completed_epoch"])
        epoch_seconds = list(state["epoch_seconds"])
        generator.set_state(state["loader_generator"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        print(f"    retomando da época {completed_epoch + 1}", flush=True)

    for epoch in range(completed_epoch, args.epochs):
        started = time.perf_counter()
        model.train()
        last_loss = float("nan")
        for batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
        duration = time.perf_counter() - started
        epoch_seconds.append(duration)
        _atomic_torch_save(torch, {
            "completed_epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_seconds": epoch_seconds,
            "loader_generator": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }, checkpoint_path)
        print(f"    época {epoch + 1}/{args.epochs}: {duration:.1f}s; loss={last_loss:.4f}", flush=True)

    model.eval()
    truth, predicted, scores = [], [], []
    inference_started = time.perf_counter()
    with torch.no_grad():
        for batch, targets in valid_loader:
            logits = model(batch)
            predicted.extend(logits.argmax(1).numpy())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].numpy())
            truth.extend(targets.numpy())
    inference_seconds = time.perf_counter() - inference_started
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    scores = np.asarray(scores)
    metrics = {
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "f1": f1_score(truth, predicted, zero_division=0),
        "roc_auc": roc_auc_score(truth, scores),
        "average_precision": average_precision_score(truth, scores),
    }
    return metrics, epoch_seconds, inference_seconds


def write_rows(path: Path, rows, fields, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_summary(path: Path, fold_rows):
    fields = ("model", "representation", "metric", "n_folds", "mean", "std", "ci95_low", "ci95_high")
    rows = []
    combinations = sorted({(row["model"], row["representation"]) for row in fold_rows})
    for model_name, representation in combinations:
        folds = [
            row for row in fold_rows
            if row["model"] == model_name and row["representation"] == representation
        ]
        for summary in summarize_folds(folds):
            rows.append({"model": model_name, "representation": representation, **summary})
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_rows(temporary, rows, fields)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.limit_per_class_record < 1 or args.threads < 1:
        raise SystemExit("folds deve ser >=2; épocas, limite por classe e threads devem ser positivos.")
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit("PyTorch e torchvision são necessários; consulte experiments/README.md.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "cnn_cv5_folds.csv"
    summary_path = args.output_dir / "cnn_cv5_summary.csv"
    timing_path = args.output_dir / "cnn_cv5_timing.csv"
    if not args.resume:
        for path in (fold_path, summary_path, timing_path):
            path.unlink(missing_ok=True)
        for path in args.checkpoint_dir.glob("*.pt"):
            path.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    print(f"PyTorch {torch.__version__}; torchvision {torchvision.__version__}; threads={args.threads}", flush=True)
    load_started = time.perf_counter()
    dataset = load_mitdb(
        args.records, window_samples=args.window_samples,
        channel=args.channel, data_dir=args.data_dir,
    )
    load_seconds = time.perf_counter() - load_started
    selected = cap_each_class(dataset, args.limit_per_class_record, args.seed)
    labels = dataset.labels[selected]
    groups = dataset.record_ids[selected]
    splits = make_grouped_folds(labels, groups, args.folds, args.seed)
    print(f"Dados carregados em {load_seconds:.1f}s; janelas={len(selected)}; registros={len(np.unique(groups))}; folds={args.folds}", flush=True)

    fold_rows = read_rows(fold_path) if args.resume else []
    timing_rows = read_rows(timing_path) if args.resume else []
    fold_fields = [
        "model", "representation", "fold", "validation_records", "n_train", "n_validation",
        "epochs", "seed", "image_size", "ctm_span", "ctm_measure", "weights",
        "accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision",
    ]
    timing_fields = [
        "model", "representation", "fold", "encode_seconds", "epoch_seconds_mean",
        "fit_epoch_seconds", "inference_seconds", "wall_seconds", "n_train",
        "n_validation", "epochs", "threads", "seed",
    ]

    for representation in args.representations:
        images, encode_seconds = encode_dataset(dataset, selected, representation, args)
        for model_name in args.models:
            for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
                already_done = any(
                    row["model"] == model_name
                    and row["representation"] == representation
                    and int(row["fold"]) == fold_number
                    for row in fold_rows
                )
                checkpoint = args.checkpoint_dir / f"{model_name}_{representation}_fold_{fold_number}.pt"
                if already_done:
                    checkpoint.unlink(missing_ok=True)
                    continue
                fold_seed = args.seed + fold_number
                print(f"\n[{model_name} + {representation}] fold {fold_number}/{args.folds}; seed={fold_seed}", flush=True)
                started = time.perf_counter()
                metrics, epoch_seconds, inference_seconds = fit_fold(
                    images, labels, train_idx, valid_idx, args, checkpoint, fold_seed, model_name,
                )
                wall_seconds = time.perf_counter() - started
                fold_row = {
                    "model": model_name,
                    "representation": representation,
                    "fold": fold_number,
                    "validation_records": ";".join(sorted(set(groups[valid_idx]))),
                    "n_train": len(train_idx),
                    "n_validation": len(valid_idx),
                    "epochs": args.epochs,
                    "seed": fold_seed,
                    "image_size": args.image_size,
                    "ctm_span": args.ctm_span,
                    "ctm_measure": args.ctm_measure,
                    "weights": args.weights,
                    **metrics,
                }
                timing_row = {
                    "model": model_name,
                    "representation": representation,
                    "fold": fold_number,
                    "encode_seconds": round(encode_seconds, 3),
                    "epoch_seconds_mean": round(float(np.mean(epoch_seconds)), 3),
                    "fit_epoch_seconds": round(float(np.sum(epoch_seconds)), 3),
                    "inference_seconds": round(inference_seconds, 3),
                    "wall_seconds": round(wall_seconds, 3),
                    "n_train": len(train_idx),
                    "n_validation": len(valid_idx),
                    "epochs": args.epochs,
                    "threads": args.threads,
                    "seed": fold_seed,
                }
                write_rows(fold_path, [fold_row], fold_fields, append=fold_path.exists())
                write_rows(timing_path, [timing_row], timing_fields, append=timing_path.exists())
                fold_rows.append({key: str(value) for key, value in fold_row.items()})
                timing_rows.append({key: str(value) for key, value in timing_row.items()})
                checkpoint.unlink(missing_ok=True)
                write_summary(summary_path, fold_rows)
                print("  métricas: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)
                print(f"  fold concluído em {wall_seconds / 60:.1f} min", flush=True)
        del images

    write_summary(summary_path, fold_rows)
    print(f"\nFolds: {fold_path}\nResumo média±DP e IC95%: {summary_path}\nTempos: {timing_path}")


if __name__ == "__main__":
    main()
