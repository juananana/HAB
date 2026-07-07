import json
import math
import os
import random
import time
from dataclasses import dataclass
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
import torch.nn as nn
import torch.nn.functional as F
from mne.datasets import eegbci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


RESULT_DIR = ROOT / "results" / "pilot_physionet"
DATA_DIR = ROOT / "data" / "mne"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def selective_metrics(probs: np.ndarray, y: np.ndarray, target_coverage: float, error_cost: float, confirm_cost: float):
    uncertainty = 1.0 - probs.max(axis=1)
    if target_coverage >= 0.999:
        threshold = np.inf
    else:
        threshold = float(np.quantile(uncertainty, target_coverage))
    execute = uncertainty <= threshold
    coverage = float(execute.mean())
    pred = probs.argmax(axis=1)
    if execute.any():
        sel_acc = float(accuracy_score(y[execute], pred[execute]))
        sel_risk = float(np.mean(pred[execute] != y[execute]))
    else:
        sel_acc = math.nan
        sel_risk = math.nan
    expected_cost = float((error_cost * ((pred != y) & execute).sum() + confirm_cost * (~execute).sum()) / len(y))
    return {
        "target_coverage": target_coverage,
        "actual_coverage": coverage,
        "selective_acc": sel_acc,
        "selective_risk": sel_risk,
        "expected_cost": expected_cost,
        "confirmation_rate": float((~execute).mean()),
        "threshold": threshold,
    }


def load_physionet(config: dict):
    mne.set_log_level("WARNING")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eegbci.base_url = "https://physionet.org/files/eegmmidb/1.0.0/"

    xs, ys, groups = [], [], []
    event_id = {"T1": 0, "T2": 1}
    for subject in config["subjects"]:
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
        x = epochs.get_data(copy=True).astype(np.float32)
        y = epochs.events[:, -1].astype(np.int64)
        xs.append(x)
        ys.append(y)
        groups.extend([subject] * len(y))
    return np.concatenate(xs), np.concatenate(ys), np.array(groups)


def split_data(x, y, groups, config):
    test_subject = config["test_subject"]
    train_mask = groups != test_subject
    test_mask = groups == test_subject
    train_indices = np.where(train_mask)[0]
    rng = np.random.default_rng(config["random_seed"])
    rng.shuffle(train_indices)
    val_size = max(1, int(len(train_indices) * config["validation_fraction"]))
    val_idx = train_indices[:val_size]
    tr_idx = train_indices[val_size:]
    te_idx = np.where(test_mask)[0]
    return (x[tr_idx], y[tr_idx]), (x[val_idx], y[val_idx]), (x[te_idx], y[te_idx])


def flatten_features(x):
    return x.reshape(x.shape[0], -1)


def bandpower_features(x, sfreq):
    freqs = np.fft.rfftfreq(x.shape[-1], d=1 / sfreq)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    bands = [(8, 12), (12, 18), (18, 26), (26, 30)]
    feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(spec[:, :, mask].mean(axis=-1) + 1e-8))
    return np.concatenate(feats, axis=1)


def evaluate_probs(name, probs, y, params_m=0.0, latency_ms=0.0):
    pred = probs.argmax(axis=1)
    return {
        "method": name,
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "kappa": float(cohen_kappa_score(y, pred)),
        "nll": float(log_loss(y, probs, labels=[0, 1])),
        "ece": expected_calibration_error(probs, y),
        "brier": brier_score(probs, y),
        "params_m": float(params_m),
        "latency_ms": float(latency_ms),
    }


class EEGNetLite(nn.Module):
    def __init__(self, channels, samples, classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, kernel_size=(1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            out = self.net(torch.zeros(1, 1, channels, samples))
        self.head = nn.Linear(int(np.prod(out.shape[1:])), classes)

    def forward(self, x):
        z = self.net(x.unsqueeze(1))
        return self.head(z.flatten(1))


class HABPrototype(nn.Module):
    def __init__(self, channels, samples, hidden_dim=48, cognitive_units=6, top_k=8, cognitive_top_k=None, classes=2):
        super().__init__()
        self.top_k = min(top_k, channels)
        self.cognitive_top_k = cognitive_top_k if cognitive_top_k is None else min(cognitive_top_k, cognitive_units)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=15, padding=7, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.node_proj = nn.Linear(samples, hidden_dim)
        self.query = nn.Parameter(torch.randn(cognitive_units, hidden_dim) * 0.02)
        self.cog_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))
        self.unc_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.next_head = nn.Linear(hidden_dim, hidden_dim)
        self.align = nn.Linear(hidden_dim, hidden_dim)

    def neural_graph(self, x):
        temporal = self.temporal(x)
        pooled = temporal.mean(dim=-1)
        node_seed = self.node_proj(x)
        sim = torch.softmax(torch.matmul(node_seed, node_seed.transpose(1, 2)) / math.sqrt(node_seed.shape[-1]), dim=-1)
        if self.top_k < sim.shape[-1]:
            values, idx = torch.topk(sim, self.top_k, dim=-1)
            sparse = torch.zeros_like(sim).scatter_(-1, idx, values)
            sim = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        graph_context = torch.matmul(sim, node_seed).mean(dim=1)
        h = pooled + graph_context
        return h, sim

    def cognitive_units(self, h):
        q = self.query.unsqueeze(0).expand(h.shape[0], -1, -1)
        h_expand = h.unsqueeze(1).expand_as(q)
        units = self.cog_update(q.reshape(-1, q.shape[-1]), h_expand.reshape(-1, h.shape[-1]))
        units = units.reshape(h.shape[0], q.shape[1], q.shape[2])
        cog_adj = torch.softmax(torch.matmul(units, units.transpose(1, 2)) / math.sqrt(units.shape[-1]), dim=-1)
        if self.cognitive_top_k is not None and self.cognitive_top_k < cog_adj.shape[-1]:
            values, idx = torch.topk(cog_adj, self.cognitive_top_k, dim=-1)
            sparse = torch.zeros_like(cog_adj).scatter_(-1, idx, values)
            cog_adj = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        cog = torch.matmul(cog_adj, units).mean(dim=1)
        return cog, units, cog_adj

    def forward(self, x):
        h, neural_adj = self.neural_graph(x)
        cog, units, cog_adj = self.cognitive_units(h)
        logits = self.classifier(cog)
        uncertainty = F.softplus(self.unc_head(cog)).squeeze(-1)
        next_pred = self.next_head(cog)
        aligned = self.align(cog)
        return {
            "logits": logits,
            "uncertainty": uncertainty,
            "next_pred": next_pred,
            "cog": cog,
            "aligned": aligned,
            "neural": h,
            "neural_adj": neural_adj,
            "cog_adj": cog_adj,
        }


def make_loaders(train, val, test, batch_size, train_meta=None, val_meta=None, test_meta=None, shuffle_train=True):
    def ds(pair, meta=None):
        x, y = pair
        tensors = [torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)]
        if meta is not None:
            indices, groups = meta
            tensors.extend([
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(groups, dtype=torch.long),
            ])
        return TensorDataset(*tensors)

    return (
        DataLoader(ds(train, train_meta), batch_size=batch_size, shuffle=shuffle_train),
        DataLoader(ds(val, val_meta), batch_size=batch_size),
        DataLoader(ds(test, test_meta), batch_size=batch_size),
    )


def small_world_loss(adj):
    tri = torch.matmul(adj, adj)
    clustering_proxy = (tri * adj).mean()
    sparsity = adj.abs().mean()
    entropy = -(adj * (adj + 1e-8).log()).sum(dim=-1).mean()
    return -clustering_proxy + 0.05 * sparsity + 0.001 * entropy


def train_torch_model(model, train_loader, val_loader, config, is_hab=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(config["max_epochs"]):
        aux_scale = 1.0
        if is_hab:
            warmup = int(config.get("hab", {}).get("aux_warmup_epochs", 0))
            ramp = max(1, int(config.get("hab", {}).get("aux_ramp_epochs", 1)))
            if epoch < warmup:
                aux_scale = 0.0
            elif epoch < warmup + ramp:
                aux_scale = float(epoch - warmup + 1) / float(ramp)
        model.train()
        for batch in train_loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            out = model(xb)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, yb)
            if is_hab:
                wm = torch.zeros((), device=device)
                if len(batch) >= 4 and len(yb) > 1:
                    sample_idx = batch[2].to(device)
                    sample_group = batch[3].to(device)
                    order = torch.argsort(sample_idx)
                    idx_sorted = sample_idx[order]
                    group_sorted = sample_group[order]
                    pair_mask = (idx_sorted[1:] == idx_sorted[:-1] + 1) & (group_sorted[1:] == group_sorted[:-1])
                    if pair_mask.any():
                        next_sorted = out["next_pred"][order]
                        cog_sorted = out["cog"][order]
                        wm = F.mse_loss(next_sorted[:-1][pair_mask], cog_sorted[1:][pair_mask].detach())
                align = F.mse_loss(out["aligned"], out["neural"].detach())
                sw = small_world_loss(out["cog_adj"])
                loss = loss + aux_scale * (
                    config["hab"]["lambda_wm"] * wm
                    + config["hab"]["lambda_align"] * align
                    + config["hab"]["lambda_sw"] * sw
                )
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_loss, val_acc = eval_loss_acc(model, val_loader, device)
        history.append({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


@torch.no_grad()
def eval_loss_acc(model, loader, device):
    model.eval()
    losses, preds, ys = [], [], []
    for batch in loader:
        xb, yb = batch[0].to(device), batch[1].to(device)
        out = model(xb)
        logits = out["logits"] if isinstance(out, dict) else out
        losses.append(F.cross_entropy(logits, yb).item() * len(yb))
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        ys.extend(yb.cpu().numpy())
    return float(np.sum(losses) / len(ys)), float(accuracy_score(ys, preds))


@torch.no_grad()
def predict_torch(model, loader):
    device = next(model.parameters()).device
    model.eval()
    probs, ys, cogs, next_preds = [], [], [], []
    start = time.perf_counter()
    n = 0
    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device)
        out = model(xb)
        logits = out["logits"] if isinstance(out, dict) else out
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(yb.numpy())
        if isinstance(out, dict):
            cogs.append(out["cog"].cpu().numpy())
            next_preds.append(out["next_pred"].cpu().numpy())
        n += len(yb)
    elapsed = time.perf_counter() - start
    result = {
        "probs": np.concatenate(probs),
        "y": np.concatenate(ys),
        "latency_ms": elapsed * 1000 / max(1, n),
    }
    if cogs:
        result["cog"] = np.concatenate(cogs)
        result["next_pred"] = np.concatenate(next_preds)
    return result


def count_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def run():
    config = json.loads((ROOT / "code" / "config_pilot.json").read_text(encoding="utf-8"))
    set_seed(config["random_seed"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    x, y, groups = load_physionet(config)
    (train, val, test) = split_data(x, y, groups, config)
    x_train, y_train = train
    x_val, y_val = val
    x_test, y_test = test

    sfreq = config["resample_hz"]
    metrics = []

    lda = Pipeline([("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])
    lda.fit(bandpower_features(x_train, sfreq), y_train)
    lda_probs = lda.predict_proba(bandpower_features(x_test, sfreq))
    metrics.append(evaluate_probs("Bandpower-LDA", lda_probs, y_test))

    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.2, random_state=config["random_seed"])),
    ])
    logreg.fit(flatten_features(x_train), y_train)
    lr_probs = logreg.predict_proba(flatten_features(x_test))
    metrics.append(evaluate_probs("Raw-LogReg", lr_probs, y_test))

    train_loader, val_loader, test_loader = make_loaders(train, val, test, config["batch_size"])
    channels, samples = x.shape[1], x.shape[2]

    eegnet = EEGNetLite(channels, samples)
    eegnet, eeg_hist = train_torch_model(eegnet, train_loader, val_loader, config, is_hab=False)
    eeg_pred = predict_torch(eegnet, test_loader)
    metrics.append(evaluate_probs("EEGNet-Lite", eeg_pred["probs"], eeg_pred["y"], count_params_m(eegnet), eeg_pred["latency_ms"]))
    eeg_hist.to_csv(RESULT_DIR / "eegnet_training_history.csv", index=False)

    hab = HABPrototype(
        channels,
        samples,
        hidden_dim=config["hab"]["hidden_dim"],
        cognitive_units=config["hab"]["cognitive_units"],
        top_k=config["hab"]["top_k"],
        cognitive_top_k=config["hab"].get("cognitive_top_k"),
    )
    hab, hab_hist = train_torch_model(hab, train_loader, val_loader, config, is_hab=True)
    hab_pred = predict_torch(hab, test_loader)
    metrics.append(evaluate_probs("HAB-Prototype", hab_pred["probs"], hab_pred["y"], count_params_m(hab), hab_pred["latency_ms"]))
    hab_hist.to_csv(RESULT_DIR / "hab_training_history.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULT_DIR / "main_metrics.csv", index=False)

    risk_rows = []
    for target in config["risk"]["target_coverages"]:
        row = selective_metrics(
            hab_pred["probs"],
            hab_pred["y"],
            target,
            config["risk"]["error_cost"],
            config["risk"]["confirmation_cost"],
        )
        row["method"] = "HAB-Prototype"
        risk_rows.append(row)
    pd.DataFrame(risk_rows).to_csv(RESULT_DIR / "risk_metrics.csv", index=False)

    if "cog" in hab_pred and len(hab_pred["cog"]) > 1:
        wm_err = np.mean((hab_pred["next_pred"][:-1] - hab_pred["cog"][1:]) ** 2)
    else:
        wm_err = np.nan
    wm_df = pd.DataFrame([
        {"method": "Last-state Copy", "representation_mse": np.nan, "next_step_acc": np.nan, "note": "Not evaluated in pilot"},
        {"method": "HAB-Prototype", "representation_mse": float(wm_err), "next_step_acc": np.nan, "note": "Cognitive latent next-state proxy"},
    ])
    wm_df.to_csv(RESULT_DIR / "world_model_metrics.csv", index=False)

    summary = [
        "# Pilot Experiment Results",
        "",
        "Scope: PhysioNet EEGBCI pilot, subjects 1-5, leave-subject-5-out, left/right motor imagery runs 4/8/12.",
        "",
        "## Main metrics",
        dataframe_to_markdown(metrics_df),
        "",
        "## Risk metrics",
        dataframe_to_markdown(pd.DataFrame(risk_rows)),
        "",
        "## World model proxy",
        dataframe_to_markdown(wm_df),
    ]
    (RESULT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print((RESULT_DIR / "summary.md").resolve())


if __name__ == "__main__":
    run()
