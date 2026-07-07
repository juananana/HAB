import argparse
import copy
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_DATA", str(ROOT / "data" / "mne"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import mne
import numpy as np
import pandas as pd
import torch
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_pilot_experiment import (
    EEGNetLite,
    HABPrototype,
    bandpower_features,
    brier_score,
    count_params_m,
    dataframe_to_markdown,
    expected_calibration_error,
    make_loaders,
    predict_torch,
    selective_metrics,
    set_seed,
    train_torch_model,
)


DATA_DIR = ROOT / "data" / "mne"
DEFAULT_RESULT_DIR = ROOT / "results" / "full_physionet_loso"


def parse_args():
    parser = argparse.ArgumentParser(description="Full PhysioNet EEGBCI LOSO experiment for HAB.")
    parser.add_argument("--config", default=str(ROOT / "code" / "config_full_physionet.json"))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--only-test-subjects", default="", help="Comma-separated subject ids for controlled partial runs.")
    parser.add_argument("--force", action="store_true", help="Re-run completed method/fold/seed rows.")
    return parser.parse_args()


def normalize_subjects(value):
    if value == "all":
        return list(range(1, 110))
    return [int(v) for v in value]


def config_hash(config):
    tracked = {
        "dataset": config["dataset"],
        "runs": config["runs"],
        "classes": config["classes"],
        "tmin": config["tmin"],
        "tmax": config["tmax"],
        "bandpass": config["bandpass"],
        "resample_hz": config["resample_hz"],
    }
    blob = json.dumps(tracked, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def append_row(path, row):
    if path.exists():
        columns = list(pd.read_csv(path, nrows=0).columns)
        for key in row:
            if key not in columns:
                columns.append(key)
        if set(columns) != set(pd.read_csv(path, nrows=0).columns):
            existing = pd.read_csv(path)
            existing = existing.reindex(columns=columns)
            existing.to_csv(path, index=False)
        df = pd.DataFrame([row]).reindex(columns=columns)
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)


def read_completed(metrics_path):
    if not metrics_path.exists():
        return set()
    df = pd.read_csv(metrics_path)
    required = {"seed", "test_subject", "method", "status"}
    if not required.issubset(df.columns):
        return set()
    done = df[df["status"] == "ok"]
    return set(zip(done["seed"].astype(int), done["test_subject"].astype(int), done["method"].astype(str)))


def load_or_build_dataset(config, result_dir):
    mne.set_log_level("WARNING")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = result_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"physionet_eegbci_{config_hash(config)}.npz"
    log_path = cache_dir / f"physionet_eegbci_{config_hash(config)}_load_log.csv"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["x"], data["y"], data["groups"], pd.DataFrame(data["load_log"].tolist())

    eegbci.base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    xs, ys, groups, logs = [], [], [], []
    event_id = {"T1": 0, "T2": 1}
    subjects = normalize_subjects(config["subjects"])

    for subject in subjects:
        started = time.time()
        try:
            files = eegbci.load_data(subject, config["runs"], path=str(DATA_DIR), update_path=False, verbose=False)
            raws = [mne.io.read_raw_edf(path, preload=True, verbose=False) for path in files]
            raw = mne.concatenate_raws(raws)
            eegbci.standardize(raw)
            raw.pick_types(eeg=True)
            raw.filter(config["bandpass"][0], config["bandpass"][1], fir_design="firwin", verbose=False)
            raw.resample(config["resample_hz"], npad="auto", verbose=False)
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=config["tmin"],
                tmax=config["tmax"],
                baseline=None,
                preload=True,
                verbose=False,
            )
            x_sub = epochs.get_data(copy=True).astype(np.float32)
            y_sub = epochs.events[:, -1].astype(np.int64)
            xs.append(x_sub)
            ys.append(y_sub)
            groups.extend([subject] * len(y_sub))
            logs.append({
                "subject": subject,
                "status": "ok",
                "n_epochs": len(y_sub),
                "n_channels": x_sub.shape[1],
                "n_samples": x_sub.shape[2],
                "seconds": time.time() - started,
                "message": "",
            })
        except Exception as exc:
            logs.append({
                "subject": subject,
                "status": "failed",
                "n_epochs": 0,
                "n_channels": 0,
                "n_samples": 0,
                "seconds": time.time() - started,
                "message": repr(exc),
            })
            print(f"[WARN] subject {subject} failed: {exc}", flush=True)

    if not xs:
        raise RuntimeError("No subject data could be loaded.")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    groups = np.array(groups, dtype=np.int64)
    load_log = pd.DataFrame(logs)
    load_log.to_csv(log_path, index=False)
    np.savez_compressed(cache_path, x=x, y=y, groups=groups, load_log=load_log.to_dict("records"))
    return x, y, groups, load_log


def subject_split(groups, test_subject, seed, val_fraction):
    unique = np.array(sorted(set(groups.tolist())))
    train_pool = unique[unique != test_subject]
    rng = np.random.default_rng(seed + int(test_subject) * 1009)
    shuffled = train_pool.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_subjects = np.array(sorted(shuffled[:n_val]))
    train_subjects = np.array(sorted(shuffled[n_val:]))
    train_idx = np.where(np.isin(groups, train_subjects))[0]
    val_idx = np.where(np.isin(groups, val_subjects))[0]
    test_idx = np.where(groups == test_subject)[0]
    return train_idx, val_idx, test_idx, train_subjects, val_subjects


def standardize_epochs(x_train, x_val, x_test):
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std


def apply_temperature(probs, temperature):
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / max(float(temperature), 1e-6)
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(val_probs, y_val):
    candidates = np.concatenate([
        np.linspace(0.5, 1.5, 21),
        np.linspace(1.6, 5.0, 18),
    ])
    best_t, best_loss = 1.0, float("inf")
    for temp in candidates:
        calibrated = apply_temperature(val_probs, temp)
        loss = log_loss(y_val, calibrated, labels=[0, 1])
        if loss < best_loss:
            best_t, best_loss = float(temp), float(loss)
    return best_t


def evaluate_method(name, probs, y, seed, test_subject, train_subjects, val_subjects, params_m=0.0, latency_ms=0.0, raw_probs=None, temperature=1.0):
    raw_probs = probs if raw_probs is None else raw_probs
    pred = probs.argmax(axis=1)
    return {
        "status": "ok",
        "seed": seed,
        "test_subject": int(test_subject),
        "method": name,
        "n_train_subjects": len(train_subjects),
        "n_val_subjects": len(val_subjects),
        "val_subjects": " ".join(str(int(s)) for s in val_subjects),
        "n_test": len(y),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "kappa": float(cohen_kappa_score(y, pred)),
        "nll": float(log_loss(y, probs, labels=[0, 1])),
        "ece": expected_calibration_error(probs, y),
        "brier": brier_score(probs, y),
        "raw_nll": float(log_loss(y, raw_probs, labels=[0, 1])),
        "raw_ece": expected_calibration_error(raw_probs, y),
        "raw_brier": brier_score(raw_probs, y),
        "temperature": float(temperature),
        "params_m": float(params_m),
        "latency_ms": float(latency_ms),
        "error": "",
    }


def graph_summary(cog_adj):
    tri = np.matmul(cog_adj, cog_adj)
    return {
        "clustering_proxy": float(np.mean(tri * cog_adj)),
        "adjacency_l1_mean": float(np.mean(np.abs(cog_adj))),
        "edge_entropy": float(np.mean(-np.sum(cog_adj * np.log(cog_adj + 1e-8), axis=-1))),
        "mean_max_edge": float(np.mean(np.max(cog_adj, axis=-1))),
    }


@torch.no_grad()
def predict_hab_states(model, loader):
    device = next(model.parameters()).device
    model.eval()
    probs, ys, cogs, next_preds, cog_adjs = [], [], [], [], []
    start = time.perf_counter()
    n = 0
    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device)
        out = model(xb)
        probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
        ys.append(yb.numpy())
        cogs.append(out["cog"].cpu().numpy())
        next_preds.append(out["next_pred"].cpu().numpy())
        cog_adjs.append(out["cog_adj"].cpu().numpy())
        n += len(yb)
    elapsed = time.perf_counter() - start
    return {
        "probs": np.concatenate(probs),
        "y": np.concatenate(ys),
        "cog": np.concatenate(cogs),
        "next_pred": np.concatenate(next_preds),
        "cog_adj": np.concatenate(cog_adjs),
        "latency_ms": elapsed * 1000 / max(1, n),
    }


def method_config(config, method, channels):
    cfg = copy.deepcopy(config)
    if method == "HAB-NoWorldModel":
        cfg["hab"]["lambda_wm"] = 0.0
    elif method == "HAB-NoSmallWorld":
        cfg["hab"]["lambda_sw"] = 0.0
    elif method == "HAB-NoAlign":
        cfg["hab"]["lambda_align"] = 0.0
    elif method == "HAB-DenseGraph":
        cfg["hab"]["top_k"] = channels
        cfg["hab"]["cognitive_top_k"] = None
    return cfg


def run_csp_lda(config, x_train, y_train, x_test):
    n_components = min(config["csp"]["n_components"], x_train.shape[1])
    csp = CSP(n_components=n_components, reg=config["csp"]["reg"], log=True, norm_trace=False)
    clf = Pipeline([("csp", csp), ("lda", LinearDiscriminantAnalysis())])
    clf.fit(x_train, y_train)
    return clf


def run_torch_method(method, config, train, val, test, result_dir, seed, test_subject, meta=None):
    x_train, _ = train
    channels, samples = x_train.shape[1], x_train.shape[2]

    if method == "EEGNet-Lite":
        train_loader, val_loader, test_loader = make_loaders(train, val, test, config["batch_size"], shuffle_train=True)
        model = EEGNetLite(channels, samples)
        model, history = train_torch_model(model, train_loader, val_loader, config, is_hab=False)
        pred = predict_torch(model, test_loader)
        history_name = f"history_seed{seed}_sub{int(test_subject):03d}_{method}.csv"
        (result_dir / "histories").mkdir(parents=True, exist_ok=True)
        history.to_csv(result_dir / "histories" / history_name, index=False)
        val_pred = predict_torch(model, val_loader)
        return pred, val_pred, count_params_m(model), history

    hab_cfg = method_config(config, method, channels)
    train_loader, val_loader, test_loader = make_loaders(
        train,
        val,
        test,
        hab_cfg["batch_size"],
        train_meta=meta["train"] if meta else None,
        val_meta=meta["val"] if meta else None,
        test_meta=meta["test"] if meta else None,
        shuffle_train=False,
    )
    model = HABPrototype(
        channels,
        samples,
        hidden_dim=hab_cfg["hab"]["hidden_dim"],
        cognitive_units=hab_cfg["hab"]["cognitive_units"],
        top_k=hab_cfg["hab"]["top_k"],
        cognitive_top_k=hab_cfg["hab"].get("cognitive_top_k"),
    )
    model, history = train_torch_model(model, train_loader, val_loader, hab_cfg, is_hab=True)
    pred = predict_hab_states(model, test_loader)
    val_pred = predict_hab_states(model, val_loader)
    history_name = f"history_seed{seed}_sub{int(test_subject):03d}_{method}.csv"
    (result_dir / "histories").mkdir(parents=True, exist_ok=True)
    history.to_csv(result_dir / "histories" / history_name, index=False)
    return pred, val_pred, count_params_m(model), history


def write_summary(result_dir, config, load_log):
    metrics_path = result_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    ok = metrics[metrics["status"] == "ok"].copy()
    summary_lines = [
        "# Full PhysioNet EEGBCI LOSO Experiment",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Host: {socket.gethostname()}",
        "",
        "## Dataset Load Summary",
        dataframe_to_markdown(load_log.groupby("status", as_index=False).agg(subjects=("subject", "count"), epochs=("n_epochs", "sum"))),
        "",
        "## Completed Rows",
        f"{len(ok)} successful method/fold/seed rows.",
        "",
    ]

    if len(ok):
        aggregate = ok.groupby("method", as_index=False).agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            kappa_mean=("kappa", "mean"),
            nll_mean=("nll", "mean"),
            ece_mean=("ece", "mean"),
            brier_mean=("brier", "mean"),
            raw_nll_mean=("raw_nll", "mean"),
            raw_ece_mean=("raw_ece", "mean"),
            raw_brier_mean=("raw_brier", "mean"),
            temperature_mean=("temperature", "mean"),
            latency_ms_mean=("latency_ms", "mean"),
            rows=("acc", "count"),
        ).sort_values("acc_mean", ascending=False)
        aggregate.to_csv(result_dir / "aggregate_metrics.csv", index=False)
        summary_lines.extend(["## Aggregate Metrics", dataframe_to_markdown(aggregate), ""])

    risk_path = result_dir / "risk_metrics.csv"
    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        if len(risk):
            risk_agg = risk.groupby(["method", "target_coverage"], as_index=False).agg(
                actual_coverage_mean=("actual_coverage", "mean"),
                selective_acc_mean=("selective_acc", "mean"),
                selective_risk_mean=("selective_risk", "mean"),
                expected_cost_mean=("expected_cost", "mean"),
                confirmation_rate_mean=("confirmation_rate", "mean"),
                rows=("method", "count"),
            )
            risk_agg.to_csv(result_dir / "aggregate_risk_metrics.csv", index=False)
            summary_lines.extend(["## Aggregate Risk Metrics", dataframe_to_markdown(risk_agg), ""])

    wm_path = result_dir / "world_model_metrics.csv"
    if wm_path.exists():
        wm = pd.read_csv(wm_path)
        if len(wm):
            wm_agg = wm.groupby("method", as_index=False).agg(
                hab_next_state_mse_mean=("hab_next_state_mse", "mean"),
                last_state_copy_mse_mean=("last_state_copy_mse", "mean"),
                rows=("method", "count"),
            )
            wm_agg.to_csv(result_dir / "aggregate_world_model_metrics.csv", index=False)
            summary_lines.extend(["## Aggregate World-Model Proxy", dataframe_to_markdown(wm_agg), ""])

    (result_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    methods = []
    if config.get("run_ablations", False):
        method_candidates = list(config["methods"]) + list(config.get("ablation_methods", []))
    else:
        method_candidates = list(config["methods"])
    for method in method_candidates:
        if method not in methods:
            methods.append(method)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_meta = {
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "mne_version": mne.__version__,
    }
    (result_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    x, y, groups, load_log = load_or_build_dataset(config, result_dir)
    subjects_loaded = sorted(set(groups.tolist()))
    configured_tests = normalize_subjects(config["test_subjects"])
    test_subjects = [s for s in configured_tests if s in subjects_loaded]
    if args.only_test_subjects.strip():
        wanted = {int(s.strip()) for s in args.only_test_subjects.split(",") if s.strip()}
        test_subjects = [s for s in test_subjects if s in wanted]

    metrics_path = result_dir / "metrics.csv"
    risk_path = result_dir / "risk_metrics.csv"
    wm_path = result_dir / "world_model_metrics.csv"
    graph_path = result_dir / "graph_metrics.csv"
    completed = set() if args.force else read_completed(metrics_path)

    print(f"[INFO] loaded epochs={len(y)} subjects={len(subjects_loaded)} methods={methods}", flush=True)
    print(f"[INFO] test_subjects={len(test_subjects)} seeds={config['random_seeds']}", flush=True)

    for seed in config["random_seeds"]:
        set_seed(int(seed))
        for test_subject in test_subjects:
            train_idx, val_idx, test_idx, train_subjects, val_subjects = subject_split(
                groups,
                test_subject,
                int(seed),
                float(config["validation_subject_fraction"]),
            )
            if len(test_idx) == 0 or len(train_idx) == 0 or len(val_idx) == 0:
                continue

            x_train_raw, y_train = x[train_idx], y[train_idx]
            x_val_raw, y_val = x[val_idx], y[val_idx]
            x_test_raw, y_test = x[test_idx], y[test_idx]
            x_train, x_val, x_test = standardize_epochs(x_train_raw, x_val_raw, x_test_raw)
            train = (x_train, y_train)
            val = (x_val, y_val)
            test = (x_test, y_test)
            meta = {
                "train": (train_idx, groups[train_idx]),
                "val": (val_idx, groups[val_idx]),
                "test": (test_idx, groups[test_idx]),
            }

            for method in methods:
                key = (int(seed), int(test_subject), method)
                if key in completed:
                    continue
                started = time.time()
                print(f"[RUN] seed={seed} test_subject={test_subject} method={method}", flush=True)
                try:
                    if method == "Bandpower-LDA":
                        clf = Pipeline([("scale", StandardScaler()), ("lda", LinearDiscriminantAnalysis())])
                        clf.fit(bandpower_features(x_train, config["resample_hz"]), y_train)
                        val_raw_probs = clf.predict_proba(bandpower_features(x_val, config["resample_hz"]))
                        raw_probs = clf.predict_proba(bandpower_features(x_test, config["resample_hz"]))
                        temperature = fit_temperature(val_raw_probs, y_val)
                        probs = apply_temperature(raw_probs, temperature)
                        row = evaluate_method(
                            method,
                            probs,
                            y_test,
                            seed,
                            test_subject,
                            train_subjects,
                            val_subjects,
                            raw_probs=raw_probs,
                            temperature=temperature,
                        )
                    elif method == "CSP-LDA":
                        clf = run_csp_lda(config, x_train, y_train, x_test)
                        val_raw_probs = clf.predict_proba(x_val)
                        raw_probs = clf.predict_proba(x_test)
                        temperature = fit_temperature(val_raw_probs, y_val)
                        probs = apply_temperature(raw_probs, temperature)
                        row = evaluate_method(
                            method,
                            probs,
                            y_test,
                            seed,
                            test_subject,
                            train_subjects,
                            val_subjects,
                            raw_probs=raw_probs,
                            temperature=temperature,
                        )
                    elif method == "EEGNet-Lite" or method.startswith("HAB-"):
                        pred, val_pred, params_m, _ = run_torch_method(method, config, train, val, test, result_dir, seed, test_subject, meta=meta)
                        raw_probs = pred["probs"]
                        temperature = fit_temperature(val_pred["probs"], val_pred["y"])
                        probs = apply_temperature(raw_probs, temperature)
                        row = evaluate_method(
                            method,
                            probs,
                            pred["y"],
                            seed,
                            test_subject,
                            train_subjects,
                            val_subjects,
                            params_m=params_m,
                            latency_ms=pred["latency_ms"],
                            raw_probs=raw_probs,
                            temperature=temperature,
                        )
                        if method.startswith("HAB-"):
                            if len(pred["cog"]) > 1:
                                wm_row = {
                                    "seed": seed,
                                    "test_subject": int(test_subject),
                                    "method": method,
                                    "hab_next_state_mse": float(np.mean((pred["next_pred"][:-1] - pred["cog"][1:]) ** 2)),
                                    "last_state_copy_mse": float(np.mean((pred["cog"][:-1] - pred["cog"][1:]) ** 2)),
                                }
                                append_row(wm_path, wm_row)
                            graph_row = {"seed": seed, "test_subject": int(test_subject), "method": method}
                            graph_row.update(graph_summary(pred["cog_adj"]))
                            append_row(graph_path, graph_row)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    row["seconds"] = time.time() - started
                    append_row(metrics_path, row)
                    for target in config["risk"]["target_coverages"]:
                        risk_row = selective_metrics(
                            probs,
                            y_test,
                            float(target),
                            float(config["risk"]["error_cost"]),
                            float(config["risk"]["confirmation_cost"]),
                        )
                        risk_row.update({"seed": seed, "test_subject": int(test_subject), "method": method})
                        append_row(risk_path, risk_row)
                    completed.add(key)
                    write_summary(result_dir, config, load_log)
                except Exception as exc:
                    fail_row = {
                        "status": "failed",
                        "seed": seed,
                        "test_subject": int(test_subject),
                        "method": method,
                        "n_train_subjects": len(train_subjects),
                        "n_val_subjects": len(val_subjects),
                        "val_subjects": " ".join(str(int(s)) for s in val_subjects),
                        "n_test": len(y_test),
                        "acc": math.nan,
                        "macro_f1": math.nan,
                        "kappa": math.nan,
                        "nll": math.nan,
                        "ece": math.nan,
                        "brier": math.nan,
                        "raw_nll": math.nan,
                        "raw_ece": math.nan,
                        "raw_brier": math.nan,
                        "temperature": math.nan,
                        "params_m": math.nan,
                        "latency_ms": math.nan,
                        "seconds": time.time() - started,
                        "error": repr(exc),
                    }
                    append_row(metrics_path, fail_row)
                    print(f"[ERROR] seed={seed} test_subject={test_subject} method={method}: {exc}", flush=True)
                    write_summary(result_dir, config, load_log)

    write_summary(result_dir, config, load_log)
    print(f"[DONE] summary: {result_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
