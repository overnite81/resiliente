"""Grouped SVM evaluation for ECG representation comparisons."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class FoldResult:
    representation: str
    fold: int
    roc_auc: float
    average_precision: float
    sensitivity: float
    specificity: float
    f1: float
    balanced_accuracy: float
    accuracy: float
    best_params: dict



def _safe_metric(metric, *args) -> float:
    try:
        return float(metric(*args))
    except ValueError:
        return float("nan")



def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.0).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else float("nan")
    return {
        "roc_auc": _safe_metric(roc_auc_score, labels, scores),
        "average_precision": _safe_metric(average_precision_score, labels, scores),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }



def _make_search(inner_splits: int, random_state: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=False, random_state=random_state)),
        ]
    )
    parameter_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__gamma": ["scale", 0.001, 0.01],
        "classifier__class_weight": [None, "balanced"],
    }
    return GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="roc_auc",
        cv=GroupKFold(n_splits=inner_splits),
        refit=True,
        n_jobs=1,
        error_score="raise",
    )



def evaluate_representation(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    representation: str,
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> list[FoldResult]:
    """Evaluate one representation using nested grouped cross-validation."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups)
    if features.ndim != 2:
        raise ValueError("features deve ter formato (n_janelas, n_caracteristicas).")
    if len(features) != len(labels) or len(labels) != len(groups):
        raise ValueError("features, labels e groups devem ter o mesmo tamanho.")

    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("n_splits deve estar entre 2 e o número de grupos.")
    if inner_splits < 2:
        raise ValueError("inner_splits deve ser pelo menos 2.")

    outer = GroupKFold(n_splits=n_splits)
    results = []
    for fold, (train_index, test_index) in enumerate(
        outer.split(features, labels, groups),
        start=1,
    ):
        train_groups = np.unique(groups[train_index])
        current_inner_splits = min(inner_splits, len(train_groups))
        if current_inner_splits < 2:
            raise ValueError("Não há grupos suficientes para validação interna.")

        search = _make_search(current_inner_splits, random_state)
        search.fit(features[train_index], labels[train_index], groups=groups[train_index])
        scores = search.best_estimator_.decision_function(features[test_index])
        metrics = _classification_metrics(labels[test_index], scores)
        results.append(
            FoldResult(
                representation=representation,
                fold=fold,
                best_params=search.best_params_,
                **metrics,
            )
        )

    return results



def compare_representations(
    feature_sets: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> list[FoldResult]:
    """Run the identical grouped protocol for every representation."""
    if not feature_sets:
        raise ValueError("feature_sets não pode ser vazio.")
    results = []
    for representation, features in feature_sets.items():
        results.extend(
            evaluate_representation(
                features,
                labels,
                groups,
                representation=representation,
                n_splits=n_splits,
                inner_splits=inner_splits,
                random_state=random_state,
            )
        )
    return results
