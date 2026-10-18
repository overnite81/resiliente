#!/usr/bin/env python3
"""Chronological ETT forecasting study: raw/STFT/Wavelet/CTM x SVM/CNNs."""
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pywt
from PIL import Image
from scipy.signal import stft
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from tools.ctm_plot_image_fixed import build_ctm_grid

REPRESENTATIONS = ("raw", "stft", "wavelet", "ctm")
MODELS = ("svm", "resnet18", "swin_tiny")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="ETT CSV, por exemplo ETTh1.csv")
    parser.add_argument("--dataset-name", default=None, help="nome registrado nos artefatos")
    parser.add_argument("--target", default="OT", help="coluna alvo; padrão do ETT é OT")
    parser.add_argument("--sequence-length", type=int, default=336)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--evaluation-stride", type=int, default=96,
                        help="avança este número de timestamps entre janelas de validação/teste")
    parser.add_argument("--max-train-windows", type=int, default=1024,
                        help="subamostra temporalmente uniforme comum a todos os métodos; 0 usa todas")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS,
                        default=list(REPRESENTATIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--svm-image-size", type=int, default=96,
                        help="resolução do mapa achatado pelo SVM; padrão iguala a entrada das CNNs")
    parser.add_argument("--encoding-samples", type=int, default=128)
    parser.add_argument("--ctm-span", type=int, default=32)
    parser.add_argument("--wavelet-scales", type=int, default=32)
    parser.add_argument("--ctm-measure", choices=("amplitude", "energy"), default="amplitude")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/ett_study"))
    return parser.parse_args()


def load_ett_csv(path: Path, target_column: str):
    """Read an ETT CSV in file order; all non-date numeric columns are inputs."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        date_column = next((name for name in reader.fieldnames if name.lower() == "date"), None)
        if date_column is None:
            raise ValueError("O CSV precisa ter uma coluna 'date'.")
        feature_columns = [name for name in reader.fieldnames if name != date_column]
        if target_column not in feature_columns:
            raise ValueError(f"Coluna alvo {target_column!r} não encontrada; disponíveis: {feature_columns}")
        dates, rows = [], []
        for line_number, row in enumerate(reader, start=2):
            dates.append(row[date_column])
            try:
                rows.append([float(row[name]) for name in feature_columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Valor não numérico na linha {line_number} do CSV.") from exc
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 10 or not np.isfinite(values).all():
        raise ValueError("O CSV precisa conter ao menos 10 linhas numéricas finitas.")
    if any(left > right for left, right in zip(dates, dates[1:])):
        raise ValueError("As linhas do CSV devem estar em ordem cronológica crescente.")
    return dates, feature_columns, values, feature_columns.index(target_column)


def chronological_origins(n_rows, sequence_length, prediction_length, train_ratio, validation_ratio, evaluation_stride):
    if sequence_length < 8 or prediction_length < 1 or evaluation_stride < 1:
        raise ValueError("Comprimentos de janela e stride devem ser positivos; sequence-length deve ser >=8.")
    if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and train_ratio + validation_ratio < 1):
        raise ValueError("As proporções de treino/validação devem ser positivas e somar menos que 1.")
    train_end = int(n_rows * train_ratio)
    validation_end = int(n_rows * (train_ratio + validation_ratio))
    first_testable = sequence_length + prediction_length
    if train_end < first_testable or validation_end - train_end < prediction_length or n_rows - validation_end < prediction_length:
        raise ValueError("O dataset é curto demais para as divisões e horizontes solicitados.")
    train = np.arange(0, train_end - sequence_length - prediction_length + 1)
    # Validação/teste podem usar histórico imediatamente anterior ao limite, mas seus alvos
    # permanecem inteiramente dentro do período correspondente.
    validation = np.arange(
        max(0, train_end - sequence_length),
        validation_end - sequence_length - prediction_length + 1,
        evaluation_stride,
    )
    test = np.arange(
        max(0, validation_end - sequence_length),
        n_rows - sequence_length - prediction_length + 1,
        evaluation_stride,
    )
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Uma das divisões não gerou janelas; reduza horizonte ou comprimento de sequência.")
    return {"train": train, "validation": validation, "test": test}, train_end, validation_end


def even_subsample(indices, maximum):
    if maximum <= 0 or len(indices) <= maximum:
        return np.asarray(indices, dtype=np.int64)
    positions = np.linspace(0, len(indices) - 1, num=maximum).round().astype(np.int64)
    return np.asarray(indices, dtype=np.int64)[positions]


def resize_map(values, size):
    image = Image.fromarray(np.asarray(values, dtype=np.float32))
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def encode_window(window, representation, *, image_size, encoding_samples, ctm_span, wavelet_scales, ctm_measure):
    """Encode a [time, variables] normalized input window as one grayscale map."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[0] < 8 or window.shape[1] < 1:
        raise ValueError("window deve ter forma [tempo, variáveis] e ao menos 8 timestamps.")
    positions = np.linspace(0, window.shape[0] - 1, num=encoding_samples)
    signals = [np.interp(positions, np.arange(window.shape[0]), window[:, col])
               for col in range(window.shape[1])]
    if representation == "raw":
        stacked = np.stack(signals)
    else:
        maps = []
        for signal in signals:
            if representation == "stft":
                nperseg = min(32, signal.size)
                _, _, coefficients = stft(
                    signal, nperseg=nperseg, noverlap=min(24, nperseg - 1),
                    boundary=None, padded=False,
                )
                feature_map = np.log1p(np.abs(coefficients))
            elif representation == "wavelet":
                scales = np.arange(1, min(wavelet_scales, signal.size - 1) + 1)
                coefficients, _ = pywt.cwt(signal, scales, "morl")
                feature_map = np.log1p(np.abs(coefficients))
            elif representation == "ctm":
                values = signal if ctm_measure == "amplitude" else signal**2
                points = list(zip(np.arange(signal.size, dtype=np.float64), values))
                grid = build_ctm_grid(points, ctm_span)
                feature_map = np.zeros((ctm_span, ctm_span), dtype=np.float32)
                for row_index, grid_row in enumerate(grid):
                    for col_index, value in enumerate(grid_row):
                        if value is not None:
                            feature_map[row_index, col_index] = value
            else:
                raise ValueError(f"Representação desconhecida: {representation}")
            maps.append(np.asarray(feature_map, dtype=np.float32))
        stacked = np.concatenate(maps, axis=0)
    return resize_map(stacked, image_size)


class PixelStandardizer:
    """Per-pixel train-only normalization; preserves absolute levels across windows."""
    def fit(self, images):
        self.mean = np.mean(images, axis=0, dtype=np.float64).astype(np.float32)
        self.scale = np.std(images, axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, images):
        return np.clip((images - self.mean) / self.scale, -8.0, 8.0).astype(np.float32)


def materialize_images(values, origins, sequence_length, representation, args, image_size):
    return np.stack([
        encode_window(
            values[start:start + sequence_length], representation,
            image_size=image_size,
            encoding_samples=args.encoding_samples,
            ctm_span=args.ctm_span,
            wavelet_scales=args.wavelet_scales,
            ctm_measure=args.ctm_measure,
        )
        for start in origins
    ])


def materialize_targets(values, origins, sequence_length, prediction_length, target_index):
    return np.stack([
        values[start + sequence_length:start + sequence_length + prediction_length, target_index]
        for start in origins
    ]).astype(np.float32)


def score_forecast(actual, predicted):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def build_vision_model(torch, torchvision, name, prediction_length):
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        original = model.conv1
        model.conv1 = torch.nn.Conv2d(
            1, original.out_channels, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding, bias=False,
        )
        model.fc = torch.nn.Linear(model.fc.in_features, prediction_length)
    elif name == "swin_tiny":
        model = torchvision.models.swin_t(weights=None)
        original = model.features[0][0]
        model.features[0][0] = torch.nn.Conv2d(
            1, original.out_channels, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding, bias=original.bias is not None,
        )
        model.head = torch.nn.Linear(model.head.in_features, prediction_length)
    else:
        raise ValueError(f"Modelo visual desconhecido: {name}")
    return model


def predict_batches(model, images, batch_size, torch):
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(images[:, None]).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def fit_vision_model(name, train_images, train_targets, validation_images, validation_targets,
                     test_images, args, model_seed):
    import torch
    import torchvision
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(args.threads)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    random.seed(model_seed)
    model = build_vision_model(torch, torchvision, name, args.prediction_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    train_data = TensorDataset(
        torch.from_numpy(train_images[:, None]).float(), torch.from_numpy(train_targets).float()
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_y = torch.from_numpy(validation_targets).float()
    best_state, best_loss, best_epoch, stale_epochs = None, float("inf"), 0, 0

    for epoch in range(args.epochs):
        model.train()
        for batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        validation_prediction = predict_batches(model, validation_images, args.batch_size, torch)
        validation_loss = float(criterion(
            torch.from_numpy(validation_prediction), validation_y
        ))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
        print(f"    {name} época {epoch + 1}/{args.epochs}; val_mse_norm={validation_loss:.5f}", flush=True)
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Nenhum checkpoint de validação foi produzido para {name}.")
    model.load_state_dict(best_state)
    model.eval()
    validation_prediction = predict_batches(model, validation_images, args.batch_size, torch)
    test_prediction = predict_batches(model, test_images, args.batch_size, torch)
    return validation_prediction, test_prediction, best_epoch


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def persist_results(output_dir, metric_rows, delta_rows):
    write_csv(output_dir / "ett_metrics.csv", metric_rows, (
        "dataset", "model", "representation", "split", "n_windows", "best_epoch",
        "fit_seconds", "mae", "mse", "rmse", "r2",
    ))
    write_csv(output_dir / "ett_deltas.csv", delta_rows, (
        "dataset", "representation", "cnn", "split", "metric",
        "svm_value", "cnn_value", "delta_cnn_minus_svm",
    ))


def main():
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.threads < 1:
        raise SystemExit("epochs, patience, batch-size e threads devem ser positivos.")
    if args.image_size < 32 or args.svm_image_size < 4 or args.encoding_samples < 8:
        raise SystemExit("image-size >=32, svm-image-size >=4 e encoding-samples >=8 são necessários.")
    if args.max_train_windows < 0:
        raise SystemExit("max-train-windows não pode ser negativo.")

    dates, feature_names, raw_values, target_index = load_ett_csv(args.csv, args.target)
    windows, train_end, validation_end = chronological_origins(
        len(raw_values), args.sequence_length, args.prediction_length,
        args.train_ratio, args.validation_ratio, args.evaluation_stride,
    )
    windows["train"] = even_subsample(windows["train"], args.max_train_windows)

    # Normalization is fitted only on the chronological training segment.
    feature_mean = raw_values[:train_end].mean(axis=0)
    feature_scale = raw_values[:train_end].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    values = ((raw_values - feature_mean) / feature_scale).astype(np.float32)
    target_mean = float(feature_mean[target_index])
    target_scale = float(feature_scale[target_index])
    targets = {
        split: materialize_targets(values, origins, args.sequence_length,
                                   args.prediction_length, target_index)
        for split, origins in windows.items()
    }
    # CNN/SVM optimize normalized targets; metrics are computed after inverse scaling.
    dataset_name = args.dataset_name or args.csv.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, delta_rows = [], []
    metric_names = ("mae", "mse", "rmse", "r2")
    persist_results(args.output_dir, metric_rows, delta_rows)

    print(f"Dataset={dataset_name}; rows={len(values)}; features={len(feature_names)}; target={args.target}", flush=True)
    print(f"Chronological split endpoints: train={train_end}, validation={validation_end}; windows={ {k: len(v) for k, v in windows.items()} }", flush=True)

    for representation in args.representations:
        print(f"\nRepresentação: {representation}", flush=True)
        cnn_images = {
            split: materialize_images(values, origins, args.sequence_length,
                                      representation, args, args.image_size)
            for split, origins in windows.items()
        }
        cnn_scaler = PixelStandardizer().fit(cnn_images["train"])
        cnn_images = {split: cnn_scaler.transform(images) for split, images in cnn_images.items()}
        small_images = {
            split: np.stack([resize_map(image, args.svm_image_size) for image in images])
            for split, images in cnn_images.items()
        }
        svm_scaler = PixelStandardizer().fit(small_images["train"])
        small_images = {split: svm_scaler.transform(images) for split, images in small_images.items()}
        normalized_targets = {split: target / target_scale for split, target in targets.items()}

        svm_predictions = {}
        if "svm" in args.models:
            started = time.perf_counter()
            svm_model = make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(LinearSVR(C=1.0, dual="auto", max_iter=5000, random_state=args.seed)),
            )
            svm_model.fit(
                small_images["train"].reshape(len(small_images["train"]), -1),
                normalized_targets["train"],
            )
            elapsed = time.perf_counter() - started
            for split in ("validation", "test"):
                prediction = svm_model.predict(small_images[split].reshape(len(small_images[split]), -1))
                actual_units = targets[split] * target_scale + target_mean
                predicted_units = prediction * target_scale + target_mean
                scores = score_forecast(actual_units, predicted_units)
                svm_predictions[split] = scores
                metric_rows.append({
                    "dataset": dataset_name, "model": "svm", "representation": representation,
                    "split": split, "n_windows": len(prediction), "best_epoch": "",
                    "fit_seconds": round(elapsed, 3), **scores,
                })
            persist_results(args.output_dir, metric_rows, delta_rows)

        for model_name in (name for name in args.models if name != "svm"):
            started = time.perf_counter()
            validation_prediction, test_prediction, best_epoch = fit_vision_model(
                model_name,
                cnn_images["train"], normalized_targets["train"],
                cnn_images["validation"], normalized_targets["validation"],
                cnn_images["test"], args, args.seed,
            )
            elapsed = time.perf_counter() - started
            for split, prediction in (("validation", validation_prediction), ("test", test_prediction)):
                actual_units = targets[split] * target_scale + target_mean
                predicted_units = prediction * target_scale + target_mean
                scores = score_forecast(actual_units, predicted_units)
                metric_rows.append({
                    "dataset": dataset_name, "model": model_name, "representation": representation,
                    "split": split, "n_windows": len(prediction), "best_epoch": best_epoch,
                    "fit_seconds": round(elapsed, 3), **scores,
                })
                if split in svm_predictions:
                    for metric in metric_names:
                        delta_rows.append({
                            "dataset": dataset_name,
                            "representation": representation,
                            "cnn": model_name,
                            "split": split,
                            "metric": metric,
                            "svm_value": svm_predictions[split][metric],
                            "cnn_value": scores[metric],
                            "delta_cnn_minus_svm": scores[metric] - svm_predictions[split][metric],
                        })
            persist_results(args.output_dir, metric_rows, delta_rows)
        del cnn_images, small_images

    metrics_path = args.output_dir / "ett_metrics.csv"
    deltas_path = args.output_dir / "ett_deltas.csv"
    persist_results(args.output_dir, metric_rows, delta_rows)
    config = {
        "dataset": dataset_name,
        "csv": str(args.csv),
        "target": args.target,
        "features": feature_names,
        "rows": len(raw_values),
        "train_end_index": train_end,
        "validation_end_index": validation_end,
        "window_counts": {key: int(len(value)) for key, value in windows.items()},
        "target_train_mean": target_mean,
        "target_train_std": target_scale,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "metrics_note": "delta_cnn_minus_svm is CNN metric minus SVM metric; negative is favorable for MAE/MSE/RMSE, positive for R2.",
        "split_note": "Chronological 60/20/20 by default; validation/test inputs may include preceding history, targets stay within their split.",
    }
    with (args.output_dir / "ett_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    print(f"Métricas: {metrics_path}\nDeltas CNN-SVM: {deltas_path}")


if __name__ == "__main__":
    main()
