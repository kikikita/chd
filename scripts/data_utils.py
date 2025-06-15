import os
import re
import glob
import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import optuna
import lightgbm as lgb
import xgboost as xgb

# ----------------------------- constants -----------------------------
DEVICE_COL = "ip_address"
TIME_COL = "timestamp"
TARGET_COL = "target"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------- utils --------------------------------

def make_unique_columns(columns: List[str]) -> List[str]:
    seen = {}
    result = []
    for col in columns:
        clean_col = re.sub(r"[^a-zA-Zа-яА-Я0-9_]", "_", col)
        clean_col = re.sub(r"_+", "_", clean_col).strip("_")
        if clean_col in seen:
            seen[clean_col] += 1
            clean_col = f"{clean_col}_{seen[clean_col]}"
        else:
            seen[clean_col] = 1
        result.append(clean_col)
    return result


def load_incidents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["incident_time"] = pd.to_datetime(df["incident_time"], errors="coerce")
    df = df.dropna(subset=["incident_time"])
    return df


def load_info(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates(subset=["item_id"])
    df = df.rename(columns={"name": "metric_name", "ipaddress": "ip_address"})
    return df[["item_id", "metric_name", "ip_address"]]


def build_ip_incident_map(df: pd.DataFrame) -> Dict[str, List[pd.Timestamp]]:
    mapping: Dict[str, List[pd.Timestamp]] = {}
    for ip, sub in df.groupby(DEVICE_COL):
        times = sorted(sub["incident_time"].tolist())
        mapping[ip] = times
    return mapping


def label_target(df: pd.DataFrame, ip_map: Dict[str, List[pd.Timestamp]], time_lag: pd.Timedelta) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = 0
    for ip, times in ip_map.items():
        if ip not in df[DEVICE_COL].unique():
            continue
        for t in times:
            start = t - time_lag
            mask = (
                (df[DEVICE_COL] == ip)
                & (df[TIME_COL] >= start)
                & (df[TIME_COL] <= t)
            )
            df.loc[mask, TARGET_COL] = 1
    return df


def process_file(
    file_path: str,
    map_item_to_name: Dict[str, str],
    map_item_to_ip: Dict[str, str],
    ip_inc_map: Dict[str, List[pd.Timestamp]],
    time_lag: pd.Timedelta,
    resample_freq: str = "30T",
    label: bool = True,
) -> pd.DataFrame:
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        tmp = pd.read_csv(f)
    tmp = tmp[tmp["itemId"] != "itemId"]
    tmp.columns = ["item_id", "unix_ts", "value"]
    tmp[TIME_COL] = pd.to_datetime(tmp["unix_ts"], unit="s", errors="coerce")
    tmp = tmp.dropna(subset=[TIME_COL, "value"])
    if tmp.empty:
        return pd.DataFrame()
    tmp[TIME_COL] = tmp[TIME_COL].dt.floor(resample_freq)
    tmp["metric_name"] = tmp["item_id"].map(map_item_to_name)
    tmp[DEVICE_COL] = tmp["item_id"].map(map_item_to_ip)
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
    tmp = tmp.dropna(subset=["metric_name", DEVICE_COL])
    if tmp.empty:
        return pd.DataFrame()
    df_pivot = (
        tmp.groupby([TIME_COL, DEVICE_COL, "metric_name"]) ["value"].mean().unstack(fill_value=np.nan).reset_index()
    )
    if df_pivot.empty:
        return pd.DataFrame()
    frames = []
    for ip in df_pivot[DEVICE_COL].unique():
        df_ip = df_pivot[df_pivot[DEVICE_COL] == ip].copy()
        df_ip = df_ip.sort_values(TIME_COL).set_index(TIME_COL).ffill()
        full_idx = pd.date_range(df_ip.index.min(), df_ip.index.max(), freq=resample_freq)
        df_ip = df_ip.reindex(full_idx).ffill()
        df_ip[DEVICE_COL] = ip
        df_ip[TIME_COL] = df_ip.index
        frames.append(df_ip.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if label and ip_inc_map:
        result = label_target(result, ip_inc_map, time_lag)
    else:
        result[TARGET_COL] = np.nan
    return result


def save_processed_part(part_df: pd.DataFrame, src_path: str, out_dir: Path, fmt: str = "csv") -> Path:
    month_tag = Path(src_path).parent.name
    device_tag = Path(src_path).stem
    fname = f"{month_tag}__{device_tag}.{fmt}"
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        part_df.to_csv(out_path, index=False)
    else:
        part_df.to_parquet(out_path, index=False)
    return out_path


def process_all(
    metrics_glob: str,
    incidents_path: Optional[Path],
    info_path: Path,
    out_dir: Path,
    time_lag_hours: int = 24,
    save_format: str = "csv",
    is_test: bool = False,
) -> pd.DataFrame:
    if not is_test and incidents_path and incidents_path.exists():
        inc_df = load_incidents(incidents_path)
        ip_map = build_ip_incident_map(inc_df)
    else:
        ip_map = {}
    info_df = load_info(info_path)
    map_item_to_name = dict(zip(info_df["item_id"], info_df["metric_name"]))
    map_item_to_ip = dict(zip(info_df["item_id"], info_df[DEVICE_COL]))
    files = glob.glob(metrics_glob)
    all_parts = []
    tlag = pd.Timedelta(hours=time_lag_hours)
    for fp in tqdm(files):
        df_dev = process_file(fp, map_item_to_name, map_item_to_ip, ip_map, tlag, label=not is_test)
        if df_dev.empty:
            continue
        save_processed_part(df_dev, fp, out_dir, fmt=save_format)
        all_parts.append(df_dev)
    if not all_parts:
        raise ValueError("No data processed. Check paths")
    combined = pd.concat(all_parts, ignore_index=True)
    return combined


def add_window_features(df: pd.DataFrame, windows: List[int] = [6, 12]) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in [DEVICE_COL, TIME_COL, TARGET_COL]]
    df = df.sort_values([DEVICE_COL, TIME_COL])
    new_features = []
    grouped = df.groupby(DEVICE_COL)
    for w in windows:
        roll = grouped[num_cols].rolling(w, min_periods=1)
        stats = roll.agg(["mean", "std"])
        stats.columns = [f"{col}_r{w}_{stat}" for col, stat in stats.columns]
        stats = stats.reset_index(level=0, drop=True)
        new_features.append(stats)
    new_df = pd.concat(new_features, axis=1)
    out = pd.concat([df.reset_index(drop=True), new_df], axis=1)
    return out


def prepare_dataset(
    df: pd.DataFrame,
    feature_list_path: Optional[Path] = None,
    rename_map_path: Optional[Path] = None,
    is_test: bool = False,
) -> pd.DataFrame:
    if is_test:
        if rename_map_path is None or not rename_map_path.exists():
            raise FileNotFoundError("rename_map_path not found")
        rename_map = json.loads(rename_map_path.read_text(encoding="utf-8"))
        raw_features = list(rename_map.keys())
        for col in raw_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[[DEVICE_COL, TIME_COL, TARGET_COL] + raw_features]
        df = df.rename(columns=rename_map)
    else:
        raw_features = [c for c in df.columns if c not in (DEVICE_COL, TIME_COL, TARGET_COL)]
        clean_features = make_unique_columns(raw_features)
        rename_map = dict(zip(raw_features, clean_features))
        if feature_list_path:
            feature_list_path.write_text(json.dumps(raw_features, indent=2, ensure_ascii=False), encoding="utf-8")
        if rename_map_path:
            rename_map_path.write_text(json.dumps(rename_map, indent=2, ensure_ascii=False), encoding="utf-8")
        df = df[[DEVICE_COL, TIME_COL, TARGET_COL] + raw_features].rename(columns=rename_map)
    df = add_window_features(df)
    return df


def calc_hours_to_fail(df: pd.DataFrame, inc_df: pd.DataFrame, max_lag_hours: int = 24) -> pd.Series:
    out = np.full(len(df), np.nan, dtype="float64")
    max_ns = max_lag_hours * 3_600_000_000_000
    ts_ns = df[TIME_COL].astype("datetime64[ns]").view("int64").to_numpy()
    inc_map = (
        inc_df.groupby(DEVICE_COL)["incident_time"].apply(lambda s: np.sort(s.astype("datetime64[ns]").view("int64"))).to_dict()
    )
    dev_arr = df[DEVICE_COL].to_numpy()
    for ip, inc_ns in inc_map.items():
        idx = np.flatnonzero(dev_arr == ip)
        if idx.size == 0:
            continue
        pos = np.searchsorted(inc_ns, ts_ns[idx], side="left")
        valid = pos < len(inc_ns)
        future = idx[valid]
        delta_ns = inc_ns[pos[valid]] - ts_ns[future]
        mask = delta_ns <= max_ns
        out[future[mask]] = (delta_ns[mask] // 3_600_000_000_000).astype(float)
    return pd.Series(out, index=df.index, name="hours_to_fail")


def update_target_by_htf(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    res = df.copy()
    res[TARGET_COL] = (
        res["hours_to_fail"].notna() & (res["hours_to_fail"] <= lag_hours)
    ).astype(int)
    return res


def split_dataset(df: pd.DataFrame, by_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    thr = pd.to_datetime(by_date)
    train = df[df[TIME_COL] <= thr]
    test = df[df[TIME_COL] > thr]
    return train.reset_index(drop=True), test.reset_index(drop=True)


class OptunaEarlyStop:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        score = trial.value
        direction = study.direction
        improved = (
            (direction == optuna.study.StudyDirection.MAXIMIZE and (self.best is None or score - self.best > self.min_delta)) or
            (direction == optuna.study.StudyDirection.MINIMIZE and (self.best is None or self.best - score > self.min_delta))
        )
        if improved:
            self.best = score
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            study.stop()


class ModelTrainer:
    def __init__(self, model_type: str = "lgbm", n_trials: int = 40, cv_splits: int = 3, random_state: int = 42, timeseries_split: bool = False):
        assert model_type in ("lgbm", "xgb")
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.timeseries_split = timeseries_split
        self.random_state = random_state
        self.best_params_: Dict = {}
        self.best_threshold_: float = 0.5
        self.model_ = None

    def _lgb_objective(self, trial, X, y, groups, feature_cols):
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256, step=8),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "verbosity": -1,
            "verbose": -1,
            "early_stopping_round": 100,
            "force_col_wise": True,
            "seed": self.random_state,
        }
        if self.timeseries_split:
            splitter = TimeSeriesSplit(n_splits=self.cv_splits)
            splits = splitter.split(X)
        else:
            splitter = GroupKFold(n_splits=self.cv_splits)
            splits = splitter.split(X, y, groups=groups)
        aucs = []
        for tr_idx, val_idx in splits:
            lgb_train = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_cols)
            lgb_val = lgb.Dataset(X[val_idx], label=y[val_idx], reference=lgb_train)
            gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_val], valid_names=["valid"])
            proba = gbm.predict(X[val_idx], num_iteration=gbm.best_iteration)
            aucs.append(roc_auc_score(y[val_idx], proba))
        return np.mean(aucs)

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str], df_test: Optional[pd.DataFrame] = None, device: str = ""):
        if self.timeseries_split:
            df_train = df_train.sort_values(TIME_COL).reset_index(drop=True)
        X = df_train[feature_cols].values
        y = df_train[TARGET_COL].values.astype(int)
        groups = df_train[DEVICE_COL].values
        study = optuna.create_study(direction="maximize")
        callback = OptunaEarlyStop()
        objective = lambda trial: self._lgb_objective(trial, X, y, groups, feature_cols)
        study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])
        self.best_params_ = study.best_params
        self.best_params_.update({"objective": "binary", "metric": "auc", "feature_pre_filter": False, "verbosity": -1, "seed": self.random_state})
        lgb_train = lgb.Dataset(X, label=y, feature_name=feature_cols)
        valid_sets = None
        valid_names = None
        if df_test is not None:
            X_val = df_test[feature_cols].values
            y_val = df_test[TARGET_COL].values.astype(int)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            valid_sets = [lgb_val]
            valid_names = ["valid"]
        self.model_ = lgb.train(self.best_params_, lgb_train, num_boost_round=4000, valid_sets=valid_sets, valid_names=valid_names)
        if df_test is not None:
            y_test = df_test[TARGET_COL].values.astype(int)
            proba_test = self.predict_proba(df_test[feature_cols])
            best_f1, best_thr = -1, 0.5
            for thr in np.linspace(0.05, 0.95, 19):
                f1 = f1_score(y_test, (proba_test >= thr).astype(int))
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            self.best_threshold_ = best_thr
        model_name = f"best_{self.model_type}_clf_{device}"
        self.model_.save_model(MODEL_DIR / f"{model_name}.txt")
        with open(MODEL_DIR / f"{model_name}_params.json", "w") as f:
            json.dump(self.best_params_, f, indent=2)
        with open(MODEL_DIR / f"{model_name}_thr.txt", "w") as f:
            f.write(str(self.best_threshold_))
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def predict(self, X):
        return (self.predict_proba(X) >= self.best_threshold_).astype(int)

    def evaluate(self, df: pd.DataFrame, feature_cols: List[str], tag: str = "Hold-out") -> Dict:
        y_true = df[TARGET_COL].values.astype(int)
        proba = self.predict_proba(df[feature_cols])
        pred = (proba >= self.best_threshold_).astype(int)
        roc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        print(f"\n=== {tag} | thr={self.best_threshold_:.2f} ===")
        print(f"ROC-AUC={roc:.4f} | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
        print(classification_report(y_true, pred, digits=3))
        return {"roc_auc": roc, "f1": f1, "precision": prec, "recall": rec}
