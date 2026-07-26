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



def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            pieces = [str(x) for x in col if x not in ("", None)]
            flat.append(prefix + "_" + "_".join(pieces).upper())
        else:
            flat.append(str(col))
    df.columns = flat
    return df.reset_index()


def aggregate_bureau_features() -> Optional[pd.DataFrame]:
    path = RAW / "bureau.csv"
    if not path.exists():
        return None
    usecols = [
        ID_COL,
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_TYPE",
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY",
        "CNT_CREDIT_PROLONG",
    ]
    bureau = pd.read_csv(path, usecols=lambda c: c in usecols)
    bureau["BUREAU_HAS_DEBT"] = (bureau.get("AMT_CREDIT_SUM_DEBT", 0).fillna(0) > 0).astype(int)
    bureau["BUREAU_HAS_OVERDUE"] = (bureau.get("AMT_CREDIT_SUM_OVERDUE", 0).fillna(0) > 0).astype(int)
    if {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"}.issubset(bureau.columns):
        bureau["BUREAU_DEBT_CREDIT_RATIO_RAW"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    num_cols = [c for c in bureau.columns if c != ID_COL and pd.api.types.is_numeric_dtype(bureau[c])]
    agg = bureau.groupby(ID_COL)[num_cols].agg(["count", "mean", "max", "min", "sum"])
    agg = flatten_columns(agg, "BUREAU")
    if "CREDIT_ACTIVE" in bureau.columns:
        active = pd.crosstab(bureau[ID_COL], bureau["CREDIT_ACTIVE"], normalize="index").add_prefix("BUREAU_ACTIVE_SHARE_").reset_index()
        agg = agg.merge(active, on=ID_COL, how="left")
    if "CREDIT_TYPE" in bureau.columns:
        top_types = bureau["CREDIT_TYPE"].value_counts().head(6).index
        type_work = bureau.loc[bureau["CREDIT_TYPE"].isin(top_types), [ID_COL, "CREDIT_TYPE"]]
        ctype = pd.crosstab(type_work[ID_COL], type_work["CREDIT_TYPE"], normalize="index").add_prefix("BUREAU_TYPE_SHARE_").reset_index()
        agg = agg.merge(ctype, on=ID_COL, how="left")
    return agg


def aggregate_previous_application_features() -> Optional[pd.DataFrame]:
    path = RAW / "previous_application.csv"
    if not path.exists():
        return None
    usecols = [
        ID_COL,
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "HOUR_APPR_PROCESS_START",
    ]
    prev = pd.read_csv(path, usecols=lambda c: c in usecols)
    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(prev.columns):
        prev["PREV_CREDIT_APPLICATION_RATIO_RAW"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(prev.columns):
        prev["PREV_ANNUITY_CREDIT_RATIO_RAW"] = prev["AMT_ANNUITY"] / prev["AMT_CREDIT"].replace(0, np.nan)
    num_cols = [c for c in prev.columns if c != ID_COL and pd.api.types.is_numeric_dtype(prev[c])]
    agg = prev.groupby(ID_COL)[num_cols].agg(["count", "mean", "max", "min", "sum"])
    agg = flatten_columns(agg, "PREV")
    if "NAME_CONTRACT_STATUS" in prev.columns:
        status = pd.crosstab(prev[ID_COL], prev["NAME_CONTRACT_STATUS"], normalize="index").add_prefix("PREV_STATUS_SHARE_").reset_index()
        agg = agg.merge(status, on=ID_COL, how="left")
    if "NAME_CONTRACT_TYPE" in prev.columns:
        ctype = pd.crosstab(prev[ID_COL], prev["NAME_CONTRACT_TYPE"], normalize="index").add_prefix("PREV_TYPE_SHARE_").reset_index()
        agg = agg.merge(ctype, on=ID_COL, how="left")
    return agg



def aggregate_installment_features() -> Optional[pd.DataFrame]:
    path = RAW / "installments_payments.csv"
    if not path.exists():
        return None
    usecols = [
        ID_COL,
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    inst = pd.read_csv(path, usecols=lambda c: c in usecols)
    inst["INSTAL_DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["INSTAL_IS_LATE"] = (inst["INSTAL_DAYS_LATE"] > 0).astype(int)
    inst["INSTAL_PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["INSTAL_UNDERPAID"] = (inst["INSTAL_PAYMENT_DIFF"] > 0).astype(int)
    inst["INSTAL_PAYMENT_RATIO_RAW"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    num_cols = [c for c in inst.columns if c != ID_COL and pd.api.types.is_numeric_dtype(inst[c])]
    agg = inst.groupby(ID_COL)[num_cols].agg(["count", "mean", "max", "min", "sum"])
    return flatten_columns(agg, "INSTAL")

def add_secondary_table_features(train: pd.DataFrame, test: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    added = []
    train_out = train.copy()
    test_out = test.copy() if test is not None else None
    for name, builder in [("bureau", aggregate_bureau_features), ("previous_application", aggregate_previous_application_features), ("installments_payments", aggregate_installment_features)]:
        features = builder()
        if features is None:
            continue
        added.append(name)
        train_out = train_out.merge(features, on=ID_COL, how="left")
        if test_out is not None:
            test_out = test_out.merge(features, on=ID_COL, how="left")
    return train_out, test_out, added

def load_data() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], bool]:
    train_path = RAW / "application_train.csv"
    test_path = RAW / "application_test.csv"
    if train_path.exists():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path) if test_path.exists() else None
        train, test, added = add_secondary_table_features(train, test)
        (OUTPUTS / "secondary_tables_used.json").write_text(json.dumps({"tables": added}, indent=2))
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
    out["MISSING_VALUE_COUNT"] = out.isna().sum(axis=1)
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED_ANOM"] = (out["DAYS_EMPLOYED"] == 365243).astype(int)
        out.loc[out["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(out.columns):
        out["EMPLOYED_AGE_RATIO"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"].replace(0, np.nan)
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["CREDIT_TERM"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"].replace(0, np.nan)
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(out.columns):
        out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"].replace(0, np.nan)
    if {"AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["GOODS_INCOME_RATIO"] = out["AMT_GOODS_PRICE"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["INCOME_PER_PERSON"] = out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"].replace(0, np.nan)
    if {"AMT_CREDIT", "CNT_CHILDREN"}.issubset(out.columns):
        out["CREDIT_PER_CHILD"] = out["AMT_CREDIT"] / (1 + out["CNT_CHILDREN"])
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
        out["EXT_SOURCE_STD"] = out[ext_cols].std(axis=1)
        out["EXT_SOURCE_MIN"] = out[ext_cols].min(axis=1)
        out["EXT_SOURCE_MAX"] = out[ext_cols].max(axis=1)
        if len(ext_cols) == 3:
            out["EXT_SOURCE_PRODUCT"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
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
    def __init__(self, lr: float = 0.06, epochs: int = 800, l2: float = 0.01, class_weight: str = "balanced"):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogistic":
        rng = np.random.default_rng(RANDOM_SEED)
        self.w = rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1) if self.class_weight == "balanced" else 1.0
        weights = np.where(y == 1, pos_weight, 1.0)
        weights = weights / weights.mean()
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            err = (p - y) * weights
            self.w -= self.lr * ((X.T @ err) / len(y) + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)


class DecisionStumpBoost:
    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.06, max_features: int = 60, class_weight: str = "balanced"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.class_weight = class_weight
        self.init_log_odds = 0.0
        self.stumps: List[Tuple[int, float, float, float]] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionStumpBoost":
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1) if self.class_weight == "balanced" else 1.0
        weights = np.where(y == 1, pos_weight, 1.0).astype(float)
        weights = weights / weights.mean()
        pos = np.clip(np.average(y, weights=weights), 1e-4, 1 - 1e-4)
        self.init_log_odds = float(np.log(pos / (1 - pos)))
        raw = np.full(len(y), self.init_log_odds)
        importances = np.zeros(X.shape[1])
        rng = np.random.default_rng(RANDOM_SEED)
        candidate_features = np.arange(X.shape[1])
        for _ in range(self.n_estimators):
            residual = y - sigmoid(raw)
            weighted_residual = residual * weights
            best = None
            sample_features = rng.choice(candidate_features, size=min(self.max_features, X.shape[1]), replace=False)
            for j in sample_features:
                qs = np.unique(np.quantile(X[:, j], [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]))
                for thr in qs:
                    left = X[:, j] <= thr
                    if left.sum() < 30 or (~left).sum() < 30:
                        continue
                    lv = weighted_residual[left].sum() / max(weights[left].sum(), 1e-12)
                    rv = weighted_residual[~left].sum() / max(weights[~left].sum(), 1e-12)
                    pred = np.where(left, lv, rv)
                    score = float(np.average((residual - pred) ** 2, weights=weights))
                    if best is None or score < best[0]:
                        best = (score, j, float(thr), float(lv), float(rv))
            if best is None:
                break
            _, j, thr, lv, rv = best
            update = np.where(X[:, j] <= thr, lv, rv)
            raw += self.learning_rate * update
            importances[j] += float(np.average(update ** 2, weights=weights))
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


def tune_numpy_models(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> Tuple[Dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(25000, len(y_train))
    sample_idx = rng.choice(len(y_train), size=sample_size, replace=False)
    Xt, yt = X_train[sample_idx], y_train[sample_idx]
    candidates = [
        ("logistic_regression_numpy", NumpyLogistic(lr=0.05, epochs=600, l2=0.003)),
        ("logistic_regression_numpy", NumpyLogistic(lr=0.06, epochs=800, l2=0.01)),
        ("logistic_regression_numpy", NumpyLogistic(lr=0.04, epochs=900, l2=0.03)),
        ("gradient_boosted_stumps_numpy", DecisionStumpBoost(n_estimators=90, learning_rate=0.05, max_features=45)),
        ("gradient_boosted_stumps_numpy", DecisionStumpBoost(n_estimators=130, learning_rate=0.035, max_features=60)),
    ]
    best: Dict[str, Tuple[float, object, Dict[str, object]]] = {}
    rows = []
    for family, model in candidates:
        model.fit(Xt, yt)
        prob = model.predict_proba(X_valid)
        score = auc_score(y_valid, prob)
        params = {k: v for k, v in vars(model).items() if k in {"lr", "epochs", "l2", "n_estimators", "learning_rate", "max_features", "class_weight"}}
        rows.append({"model_family": family, "validation_auc": score, "params": json.dumps(params)})
        if family not in best or score > best[family][0]:
            best[family] = (score, model, params)
    tuned_models = {family: item[1] for family, item in best.items()}
    return tuned_models, pd.DataFrame(rows).sort_values("validation_auc", ascending=False)


def run_numpy_models(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_idx, valid_idx = split_indices(len(y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    models, tuning_results = tune_numpy_models(X_train, y_train, X_valid, y_valid)
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
    cv_sample_size = min(120000, len(y))
    rng = np.random.default_rng(RANDOM_SEED)
    cv_base_idx = rng.choice(len(y), size=cv_sample_size, replace=False) if cv_sample_size < len(y) else np.arange(len(y))
    X_cv, y_cv = X[cv_base_idx], y[cv_base_idx]
    for tr, va in kfold_indices(len(y_cv), 5):
        model = NumpyLogistic(lr=0.06, epochs=550, l2=0.01)
        model.fit(X_cv[tr], y_cv[tr])
        cv_scores.append(auc_score(y_cv[va], model.predict_proba(X_cv[va])))
    best_name = max(rows, key=lambda r: r["validation_auc"])["model"]
    best_model = fitted[best_name]
    raw_best_prob = best_model.predict_proba(X_valid)
    best_prob = raw_best_prob[:, 1] if getattr(raw_best_prob, "ndim", 1) == 2 else raw_best_prob
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
        "tuning_results": tuning_results,
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
    if len(y) > 8000:
        idx = rng.choice(len(y), size=8000, replace=False)
        X = X[idx]
        y = y[idx]
    raw_baseline_prob = model.predict_proba(X)
    baseline_prob = raw_baseline_prob[:, 1] if getattr(raw_baseline_prob, "ndim", 1) == 2 else raw_baseline_prob
    baseline = auc_score(y, baseline_prob)
    rows = []
    for j, name in enumerate(feature_names):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        raw_score_prob = model.predict_proba(Xp)
        score_prob = raw_score_prob[:, 1] if getattr(raw_score_prob, "ndim", 1) == 2 else raw_score_prob
        score = auc_score(y, score_prob)
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
        "- The real Home Credit Kaggle files are loaded from `data/raw` when present.",
        "- Class weights are used to handle the imbalanced default target.",
        "- Feature engineering includes credit/income ratios, annuity ratios, employment-age ratios, external-source aggregates, missing-value counts, bureau history aggregates, previous-application aggregates, and installment payment behavior aggregates.",
        "- Hyperparameter tuning results are saved to `outputs/hyperparameter_tuning_results.csv`.",
        "- scikit-learn and CatBoost models are used when installed; the NumPy stump model remains as a fallback.",
        "- On macOS, LightGBM and XGBoost may require the native `libomp` runtime before they can import.",
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
    if "tuning_results" in details:
        details["tuning_results"].to_csv(OUTPUTS / "hyperparameter_tuning_results.csv", index=False)

    save_bar_chart(FIGURES / "target_distribution.svg", "Target Distribution", [str(i) for i in sorted(train[TARGET].unique())], [float((train[TARGET] == i).mean()) for i in sorted(train[TARGET].unique())])
    cleaned_train = clean_known_anomalies(train)
    selected = [c for c in ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "CREDIT_INCOME_RATIO"] if c in cleaned_train.columns]
    save_heatmap(FIGURES / "correlation_heatmap.svg", cleaned_train, selected[:10])
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
        raw_pred = details["best_model"].predict_proba(X_test)
        pred = raw_pred[:, 1] if getattr(raw_pred, "ndim", 1) == 2 else raw_pred
        ids = test[ID_COL].to_numpy() if ID_COL in test.columns else np.arange(len(test))
        pd.DataFrame({ID_COL: ids, TARGET: pred}).to_csv(OUTPUTS / ("demo_submission.csv" if demo else "kaggle_submission.csv"), index=False)

    available = {name: optional_module(name) is not None for name in ["sklearn", "lightgbm", "xgboost", "catboost", "optuna", "shap", "matplotlib", "seaborn"]}
    write_report(demo, comparison, details, metrics, available)
    metadata = {
        "demo_mode": demo,
        "rows": int(len(train)),
        "columns": int(train.shape[1]),
        "secondary_tables_used": json.loads((OUTPUTS / "secondary_tables_used.json").read_text()).get("tables", []) if (OUTPUTS / "secondary_tables_used.json").exists() else [],
        "best_model": details["best_name"],
        "cv_auc_mean": details["cv_auc_mean"],
        "out_of_sample_auc": metrics["auc"],
    }
    (OUTPUTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
