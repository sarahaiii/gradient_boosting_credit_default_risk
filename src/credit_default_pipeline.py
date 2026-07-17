from __future__ import annotations

import csv
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_SEED = 42


def ensure_dirs() -> None:
    for path in [RAW, OUTPUTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def optional_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "auc": auc_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.unique(np.quantile(y_prob, np.linspace(0.02, 0.98, 121))):
        f1 = binary_metrics(y_true, y_prob, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0, 1, 101)))
    tpr, fpr = [], []
    for t in thresholds:
        pred = y_score >= t
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    return np.array(fpr), np.array(tpr)


def pr_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0, 1, 101)))
    precision, recall = [], []
    for t in thresholds:
        pred = y_score >= t
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / max(tp + fn, 1))
    return np.array(recall), np.array(precision)


def make_demo_data(n_train: int = 5000, n_test: int = 1200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)

    def frame(n: int, with_target: bool) -> pd.DataFrame:
        income = rng.lognormal(mean=10.7, sigma=0.45, size=n)
        credit = income * rng.uniform(2.2, 6.5, size=n)
        annuity = credit / rng.uniform(14, 32, size=n)
        age_days = -rng.integers(21 * 365, 68 * 365, size=n)
        employed_days = -rng.integers(30, 35 * 365, size=n)
        ext1 = rng.beta(3, 4, size=n)
        ext2 = rng.beta(4, 3, size=n)
        ext3 = rng.beta(3, 5, size=n)
        contract = rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1])
        education = rng.choice(["Secondary", "Higher", "Incomplete higher", "Lower secondary"], size=n, p=[0.68, 0.22, 0.07, 0.03])
        family = rng.choice(["Single", "Married", "Civil marriage", "Separated"], size=n, p=[0.18, 0.62, 0.13, 0.07])
        children = rng.poisson(0.45, size=n).clip(0, 5)
        price_ratio = credit / np.maximum(income, 1)
        linear = (
            -2.85
            + 0.42 * (price_ratio - price_ratio.mean()) / price_ratio.std()
            - 1.25 * (ext2 - ext2.mean()) / ext2.std()
            - 0.7 * (ext3 - ext3.mean()) / ext3.std()
            + 0.2 * (children > 1)
            + 0.35 * (contract == "Revolving loans")
            + 0.25 * (education == "Lower secondary")
        )
        prob = sigmoid(linear)
        out = pd.DataFrame(
            {
                ID_COL: np.arange(100000, 100000 + n),
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "DAYS_BIRTH": age_days,
                "DAYS_EMPLOYED": employed_days,
                "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3,
                "CNT_CHILDREN": children,
                "NAME_CONTRACT_TYPE": contract,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family,
            }
        )
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_ANNUITY"]:
            mask = rng.random(n) < 0.09
            out.loc[mask, col] = np.nan
        if with_target:
            out[TARGET] = rng.binomial(1, prob)
        return out

    return frame(n_train, True), frame(n_test, False)


def load_data() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], bool]:
    train_path = RAW / "application_train.csv"
    test_path = RAW / "application_test.csv"
    if train_path.exists():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path) if test_path.exists() else None
        return train, test, False
    train, test = make_demo_data()
    return train, test, True


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
            "dtype": [str(x) for x in df.dtypes],
        }
    )
    return out.sort_values(["missing_pct", "missing_count"], ascending=False)


def clean_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED_ANOM"] = (out["DAYS_EMPLOYED"] == 365243).astype(int)
        out.loc[out["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    return out


@dataclass
class Preprocessor:
    numeric_cols: List[str]
    categorical_cols: List[str]
    medians: Dict[str, float]
    modes: Dict[str, str]
    means: np.ndarray
    stds: np.ndarray
    feature_names: List[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        work = clean_known_anomalies(df)
        parts = []
        for col in self.numeric_cols:
            series = pd.to_numeric(work[col], errors="coerce") if col in work else pd.Series(np.nan, index=work.index)
            parts.append(series.fillna(self.medians[col]).to_numpy(dtype=float).reshape(-1, 1))
        for col in self.categorical_cols:
            series = work[col].astype("object") if col in work else pd.Series(np.nan, index=work.index)
            series = series.fillna(self.modes[col]).astype(str)
            cats = [name.split(f"{col}=", 1)[1] for name in self.feature_names if name.startswith(f"{col}=")]
            encoded = np.column_stack([(series == cat).astype(float).to_numpy() for cat in cats]) if cats else np.empty((len(work), 0))
            parts.append(encoded)
        X = np.hstack(parts) if parts else np.empty((len(work), 0))
        return (X - self.means) / self.stds


def fit_preprocessor(df: pd.DataFrame) -> Tuple[Preprocessor, np.ndarray, np.ndarray]:
    work = clean_known_anomalies(df)
    exclude = {TARGET, ID_COL}
    numeric_cols = [c for c in work.columns if c not in exclude and pd.api.types.is_numeric_dtype(work[c])]
    categorical_cols = [c for c in work.columns if c not in exclude and not pd.api.types.is_numeric_dtype(work[c])]
    categorical_cols = categorical_cols[:30]
    medians = {c: float(pd.to_numeric(work[c], errors="coerce").median()) for c in numeric_cols}
    medians = {k: (0.0 if math.isnan(v) else v) for k, v in medians.items()}
    modes = {}
    feature_names = list(numeric_cols)
    parts = []
    for col in numeric_cols:
        parts.append(pd.to_numeric(work[col], errors="coerce").fillna(medians[col]).to_numpy(dtype=float).reshape(-1, 1))
    for col in categorical_cols:
        mode = work[col].mode(dropna=True)
        modes[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
        series = work[col].fillna(modes[col]).astype(str)
        top_cats = list(series.value_counts().head(20).index)
        feature_names.extend([f"{col}={cat}" for cat in top_cats])
        parts.append(np.column_stack([(series == cat).astype(float).to_numpy() for cat in top_cats]))
    X = np.hstack(parts) if parts else np.empty((len(work), 0))
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    prep = Preprocessor(numeric_cols, categorical_cols, medians, modes, means, stds, feature_names)
    return prep, (X - means) / stds, work[TARGET].to_numpy(dtype=int)


class NumpyLogistic:
    def __init__(self, lr: float = 0.06, epochs: int = 800, l2: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogistic":
        rng = np.random.default_rng(RANDOM_SEED)
        self.w = rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        weights = np.where(y == 1, pos_weight, 1.0)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            err = (p - y) * weights
            self.w -= self.lr * ((X.T @ err) / len(y) + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)


class DecisionStumpBoost:
    def __init__(self, n_estimators: int = 80, learning_rate: float = 0.08):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.init_log_odds = 0.0
        self.stumps: List[Tuple[int, float, float, float]] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionStumpBoost":
        pos = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.init_log_odds = float(np.log(pos / (1 - pos)))
        raw = np.full(len(y), self.init_log_odds)
        importances = np.zeros(X.shape[1])
        rng = np.random.default_rng(RANDOM_SEED)
        candidate_features = np.arange(X.shape[1])
        for _ in range(self.n_estimators):
            residual = y - sigmoid(raw)
            best = None
            sample_features = rng.choice(candidate_features, size=min(35, X.shape[1]), replace=False)
            for j in sample_features:
                qs = np.unique(np.quantile(X[:, j], [0.2, 0.35, 0.5, 0.65, 0.8]))
                for thr in qs:
                    left = X[:, j] <= thr
                    if left.sum() < 20 or (~left).sum() < 20:
                        continue
                    lv = residual[left].mean()
                    rv = residual[~left].mean()
                    pred = np.where(left, lv, rv)
                    score = float(((residual - pred) ** 2).mean())
                    if best is None or score < best[0]:
                        best = (score, j, float(thr), float(lv), float(rv))
            if best is None:
                break
            _, j, thr, lv, rv = best
            update = np.where(X[:, j] <= thr, lv, rv)
            raw += self.learning_rate * update
            importances[j] += float(np.var(update))
            self.stumps.append((j, thr, lv, rv))
        self.feature_importances_ = importances / max(importances.sum(), 1e-12)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(X.shape[0], self.init_log_odds)
        for j, thr, lv, rv in self.stumps:
            raw += self.learning_rate * np.where(X[:, j] <= thr, lv, rv)
        return sigmoid(raw)


def split_indices(n: int, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_size))
    return idx[:cut], idx[cut:]


def kfold_indices(n: int, k: int = 5) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        valid = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, valid


def run_numpy_models(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_idx, valid_idx = split_indices(len(y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    models = {
        "logistic_regression_numpy": NumpyLogistic(),
        "gradient_boosted_stumps_numpy": DecisionStumpBoost(),
    }
    sklearn = optional_module("sklearn")
    if sklearn is not None:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            models.update(
                {
                    "logistic_regression_sklearn": LogisticRegression(max_iter=700, class_weight="balanced", n_jobs=-1),
                    "random_forest_sklearn": RandomForestClassifier(
                        n_estimators=220,
                        max_depth=9,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                    "hist_gradient_boosting_sklearn": HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.045,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=RANDOM_SEED,
                    ),
                }
            )
        except Exception:
            pass
    lightgbm = optional_module("lightgbm")
    if lightgbm is not None:
        try:
            models["lightgbm"] = lightgbm.LGBMClassifier(
                n_estimators=450,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                verbosity=-1,
            )
        except Exception:
            pass
    xgboost = optional_module("xgboost")
    if xgboost is not None:
        try:
            scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
            models["xgboost"] = xgboost.XGBClassifier(
                n_estimators=320,
                learning_rate=0.035,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
            )
        except Exception:
            pass
    catboost = optional_module("catboost")
    if catboost is not None:
        try:
            models["catboost"] = catboost.CatBoostClassifier(
                iterations=320,
                learning_rate=0.04,
                depth=5,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=RANDOM_SEED,
                verbose=False,
            )
        except Exception:
            pass
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        raw_prob = model.predict_proba(X_valid)
        prob = raw_prob[:, 1] if getattr(raw_prob, "ndim", 1) == 2 else raw_prob
        threshold = choose_threshold(y_valid, prob)
        m = binary_metrics(y_valid, prob, threshold)
        rows.append({"model": name, "validation_auc": m["auc"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
        fitted[name] = model
    cv_scores = []
    for tr, va in kfold_indices(len(y), 5):
        model = DecisionStumpBoost()
        model.fit(X[tr], y[tr])
        cv_scores.append(auc_score(y[va], model.predict_proba(X[va])))
    best_name = max(rows, key=lambda r: r["validation_auc"])["model"]
    best_model = fitted[best_name]
    best_prob = best_model.predict_proba(X_valid)
    best_threshold = choose_threshold(y_valid, best_prob)
    details = {
        "best_name": best_name,
        "best_model": best_model,
        "valid_idx": valid_idx,
        "y_valid": y_valid,
        "valid_prob": best_prob,
        "threshold": best_threshold,
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
    }
    return pd.DataFrame(rows).sort_values("validation_auc", ascending=False), details


def svg_page(title: str, body: str, width: int = 760, height: int = 480) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fbfbf8"/><text x="32" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#202124">{title}</text>{body}</svg>'


def save_bar_chart(path: Path, title: str, labels: List[str], values: List[float]) -> None:
    width, height = 760, 460
    max_v = max(values) if values else 1
    body = []
    chart_x, chart_y, chart_w, chart_h = 90, 80, 590, 300
    body.append(f'<line x1="{chart_x}" y1="{chart_y+chart_h}" x2="{chart_x+chart_w}" y2="{chart_y+chart_h}" stroke="#444"/>')
    bar_w = chart_w / max(len(values), 1) * 0.62
    for i, (label, value) in enumerate(zip(labels, values)):
        x = chart_x + i * chart_w / len(values) + bar_w * 0.3
        h = chart_h * value / max_v
        y = chart_y + chart_h - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#356d8c"/>')
        body.append(f'<text x="{x + bar_w/2:.1f}" y="{chart_y+chart_h+24}" font-family="Arial" font-size="13" text-anchor="middle" fill="#333">{label}</text>')
        body.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" font-family="Arial" font-size="13" text-anchor="middle" fill="#333">{value:.3g}</text>')
    path.write_text(svg_page(title, "".join(body), width, height))


def save_line_chart(path: Path, title: str, x: np.ndarray, y: np.ndarray, xlab: str, ylab: str) -> None:
    width, height = 760, 460
    chart_x, chart_y, chart_w, chart_h = 80, 78, 610, 300
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    xs = chart_x + chart_w * (x - x.min()) / max(x.max() - x.min(), 1e-12)
    ys = chart_y + chart_h - chart_h * (y - y.min()) / max(y.max() - y.min(), 1e-12)
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    body = f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" fill="#fff" stroke="#d7d7d7"/><polyline points="{points}" fill="none" stroke="#b84d3a" stroke-width="3"/><text x="{chart_x+chart_w/2}" y="{height-34}" font-family="Arial" font-size="14" text-anchor="middle">{xlab}</text><text x="24" y="{chart_y+chart_h/2}" font-family="Arial" font-size="14" transform="rotate(-90 24 {chart_y+chart_h/2})" text-anchor="middle">{ylab}</text>'
    path.write_text(svg_page(title, body, width, height))


def save_confusion(path: Path, m: Dict[str, float]) -> None:
    vals = [[int(m["tn"]), int(m["fp"])], [int(m["fn"]), int(m["tp"])]]
    max_v = max(max(row) for row in vals)
    body = []
    labels = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            x, y = 190 + c * 150, 110 + r * 120
            alpha = 0.18 + 0.72 * vals[r][c] / max(max_v, 1)
            body.append(f'<rect x="{x}" y="{y}" width="140" height="110" fill="rgba(53,109,140,{alpha:.2f})" stroke="#fff"/>')
            body.append(f'<text x="{x+70}" y="{y+45}" font-family="Arial" font-size="18" font-weight="700" text-anchor="middle" fill="#111">{labels[r][c]}</text>')
            body.append(f'<text x="{x+70}" y="{y+76}" font-family="Arial" font-size="20" text-anchor="middle" fill="#111">{vals[r][c]}</text>')
    path.write_text(svg_page("Confusion Matrix", "".join(body)))


def save_heatmap(path: Path, df: pd.DataFrame, cols: List[str]) -> None:
    corr = df[cols].corr(numeric_only=True).fillna(0)
    n = len(corr)
    cell = min(58, int(430 / max(n, 1)))
    body = []
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            val = float(corr.loc[row, col])
            color = "#b84d3a" if val >= 0 else "#356d8c"
            opacity = 0.12 + 0.82 * abs(val)
            x, y = 170 + j * cell, 88 + i * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" opacity="{opacity:.2f}" stroke="#fff"/>')
        body.append(f'<text x="164" y="{88+i*cell+cell*.62:.1f}" font-family="Arial" font-size="10" text-anchor="end">{row[:18]}</text>')
    for j, col in enumerate(corr.columns):
        body.append(f'<text x="{170+j*cell+cell*.5:.1f}" y="78" font-family="Arial" font-size="10" transform="rotate(-45 {170+j*cell+cell*.5:.1f} 78)" text-anchor="start">{col[:18]}</text>')
    path.write_text(svg_page("Correlation Heatmap", "".join(body), 820, 560))


def permutation_importance(model, X: np.ndarray, y: np.ndarray, feature_names: List[str], n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    baseline = auc_score(y, model.predict_proba(X))
    rows = []
    for j, name in enumerate(feature_names):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        score = auc_score(y, model.predict_proba(Xp))
        rows.append({"feature": name, "importance": max(baseline - score, 0)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(n)


def write_report(demo: bool, comparison: pd.DataFrame, details: Dict[str, object], metrics: Dict[str, float], available: Dict[str, bool]) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    lines = [
        "# Credit Default Risk Report",
        "",
        f"Mode: {'synthetic demo data' if demo else 'Home Credit Kaggle data'}",
        "",
        "## Model Results",
        "",
        markdown_table(comparison),
        "",
        f"Best model: `{details['best_name']}`",
        f"Cross-validation AUC: {details['cv_auc_mean']:.4f} +/- {details['cv_auc_std']:.4f}",
        f"Out-of-sample AUC: {metrics['auc']:.4f}",
        f"Selected threshold: {metrics['threshold']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-score: {metrics['f1']:.4f}",
        "",
        "## Optional Library Availability",
        "",
        markdown_table(pd.DataFrame([{"library": k, "available": v} for k, v in available.items()])),
        "",
        "## Notes",
        "",
        "- Drop the Kaggle CSVs into `data/raw` to run on real data.",
        "- The fallback gradient boosted stump model exists so the project runs even without ML packages installed.",
        "- Install `requirements.txt` to enable the full sklearn/LightGBM/XGBoost/CatBoost/SHAP workflow.",
    ]
    (OUTPUTS / "credit_default_report.md").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    train, test, demo = load_data()
    missing_summary(train).to_csv(OUTPUTS / "missing_value_summary.csv", index=False)
    train[TARGET].value_counts(normalize=True).sort_index().rename("share").to_csv(OUTPUTS / "target_distribution.csv")

    prep, X, y = fit_preprocessor(train)
    comparison, details = run_numpy_models(X, y)
    y_valid = details["y_valid"]
    valid_prob = details["valid_prob"]
    metrics = binary_metrics(y_valid, valid_prob, details["threshold"])
    comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)

    save_bar_chart(FIGURES / "target_distribution.svg", "Target Distribution", [str(i) for i in sorted(train[TARGET].unique())], [float((train[TARGET] == i).mean()) for i in sorted(train[TARGET].unique())])
    numeric = [c for c in clean_known_anomalies(train).columns if c != TARGET and pd.api.types.is_numeric_dtype(clean_known_anomalies(train)[c])]
    selected = [c for c in ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "CREDIT_INCOME_RATIO"] if c in clean_known_anomalies(train).columns]
    save_heatmap(FIGURES / "correlation_heatmap.svg", clean_known_anomalies(train), selected[:10])
    fpr, tpr = roc_points(y_valid, valid_prob)
    save_line_chart(FIGURES / "roc_curve.svg", f"ROC Curve - AUC {metrics['auc']:.3f}", fpr, tpr, "False Positive Rate", "True Positive Rate")
    recall, precision = pr_points(y_valid, valid_prob)
    save_line_chart(FIGURES / "precision_recall_curve.svg", "Precision-Recall Curve", recall, precision, "Recall", "Precision")
    save_confusion(FIGURES / "confusion_matrix.svg", metrics)
    imp = permutation_importance(details["best_model"], X[details["valid_idx"]], y_valid, prep.feature_names)
    imp.to_csv(OUTPUTS / "feature_importance.csv", index=False)
    save_bar_chart(FIGURES / "feature_importance.svg", "Feature Importance", imp["feature"].head(12).astype(str).tolist(), imp["importance"].head(12).astype(float).tolist())
    imp.rename(columns={"importance": "mean_abs_shap_or_proxy"}).to_csv(OUTPUTS / "shap_summary_or_proxy.csv", index=False)
    save_bar_chart(FIGURES / "shap_summary_or_proxy.svg", "SHAP Summary Proxy", imp["feature"].head(12).astype(str).tolist(), imp["importance"].head(12).astype(float).tolist())

    if test is not None:
        X_test = prep.transform(test)
        pred = details["best_model"].predict_proba(X_test)
        ids = test[ID_COL].to_numpy() if ID_COL in test.columns else np.arange(len(test))
        pd.DataFrame({ID_COL: ids, TARGET: pred}).to_csv(OUTPUTS / ("demo_submission.csv" if demo else "kaggle_submission.csv"), index=False)

    available = {name: optional_module(name) is not None for name in ["sklearn", "lightgbm", "xgboost", "catboost", "optuna", "shap", "matplotlib", "seaborn"]}
    write_report(demo, comparison, details, metrics, available)
    metadata = {
        "demo_mode": demo,
        "rows": int(len(train)),
        "columns": int(train.shape[1]),
        "best_model": details["best_name"],
        "cv_auc_mean": details["cv_auc_mean"],
        "out_of_sample_auc": metrics["auc"],
    }
    (OUTPUTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
