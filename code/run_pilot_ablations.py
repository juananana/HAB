import copy
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from run_pilot_experiment import (
    HABPrototype,
    brier_score,
    count_params_m,
    evaluate_probs,
    expected_calibration_error,
    load_physionet,
    make_loaders,
    selective_metrics,
    set_seed,
    split_data,
    train_torch_model,
    dataframe_to_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "pilot_physionet_ablation"


@torch.no_grad()
def predict_hab_with_states(model, loader):
    device = next(model.parameters()).device
    model.eval()
    probs, ys, cogs, next_preds, cog_adjs = [], [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
        ys.append(yb.numpy())
        cogs.append(out["cog"].cpu().numpy())
        next_preds.append(out["next_pred"].cpu().numpy())
        cog_adjs.append(out["cog_adj"].cpu().numpy())
    return {
        "probs": np.concatenate(probs),
        "y": np.concatenate(ys),
        "cog": np.concatenate(cogs),
        "next_pred": np.concatenate(next_preds),
        "cog_adj": np.concatenate(cog_adjs),
    }


def graph_summary(cog_adj):
    tri = np.matmul(cog_adj, cog_adj)
    clustering_proxy = float(np.mean(tri * cog_adj))
    sparsity = float(np.mean(np.abs(cog_adj)))
    entropy = float(np.mean(-np.sum(cog_adj * np.log(cog_adj + 1e-8), axis=-1)))
    max_edge = float(np.mean(np.max(cog_adj, axis=-1)))
    return {
        "clustering_proxy": clustering_proxy,
        "adjacency_l1_mean": sparsity,
        "edge_entropy": entropy,
        "mean_max_edge": max_edge,
    }


def world_model_summary(cog, next_pred):
    if len(cog) < 2:
        return {"hab_next_state_mse": math.nan, "last_state_copy_mse": math.nan}
    return {
        "hab_next_state_mse": float(np.mean((next_pred[:-1] - cog[1:]) ** 2)),
        "last_state_copy_mse": float(np.mean((cog[:-1] - cog[1:]) ** 2)),
    }


def run_variant(name, config, train, val, test, channels, samples, variant_updates):
    variant_config = copy.deepcopy(config)
    for key_path, value in variant_updates.items():
        cursor = variant_config
        parts = key_path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value

    set_seed(config["random_seed"])
    train_loader, val_loader, test_loader = make_loaders(train, val, test, variant_config["batch_size"])
    model = HABPrototype(
        channels,
        samples,
        hidden_dim=variant_config["hab"]["hidden_dim"],
        cognitive_units=variant_config["hab"]["cognitive_units"],
        top_k=variant_config["hab"]["top_k"],
    )
    model, history = train_torch_model(model, train_loader, val_loader, variant_config, is_hab=True)
    pred = predict_hab_with_states(model, test_loader)

    metrics = evaluate_probs(name, pred["probs"], pred["y"], count_params_m(model), 0.0)
    metrics["best_val_acc"] = float(history["val_acc"].max()) if len(history) else math.nan
    metrics["epochs"] = int(history["epoch"].max()) if len(history) else 0

    risk_rows = []
    for target in variant_config["risk"]["target_coverages"]:
        row = selective_metrics(
            pred["probs"],
            pred["y"],
            target,
            variant_config["risk"]["error_cost"],
            variant_config["risk"]["confirmation_cost"],
        )
        row["method"] = name
        risk_rows.append(row)

    wm = world_model_summary(pred["cog"], pred["next_pred"])
    wm["method"] = name
    graph = graph_summary(pred["cog_adj"])
    graph["method"] = name

    history.to_csv(RESULT_DIR / f"{name.replace(' ', '_').replace('/', 'no_')}_history.csv", index=False)
    return metrics, risk_rows, wm, graph


def run():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "code" / "config_pilot.json").read_text(encoding="utf-8"))
    (RESULT_DIR / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    set_seed(config["random_seed"])
    x, y, groups = load_physionet(config)
    train, val, test = split_data(x, y, groups, config)
    channels, samples = x.shape[1], x.shape[2]

    variants = [
        ("Full HAB", {}),
        ("w/o World Model", {"hab.lambda_wm": 0.0}),
        ("w/o Small-World Loss", {"hab.lambda_sw": 0.0}),
        ("w/o Neural-Cognitive Align", {"hab.lambda_align": 0.0}),
        ("Dense Neural Graph", {"hab.top_k": channels}),
    ]

    metrics_rows, risk_rows, wm_rows, graph_rows = [], [], [], []
    for name, updates in variants:
        metrics, risks, wm, graph = run_variant(name, config, train, val, test, channels, samples, updates)
        metrics_rows.append(metrics)
        risk_rows.extend(risks)
        wm_rows.append(wm)
        graph_rows.append(graph)

    metrics_df = pd.DataFrame(metrics_rows)
    risk_df = pd.DataFrame(risk_rows)
    wm_df = pd.DataFrame(wm_rows)
    graph_df = pd.DataFrame(graph_rows)

    metrics_df.to_csv(RESULT_DIR / "ablation_metrics.csv", index=False)
    risk_df.to_csv(RESULT_DIR / "ablation_risk_metrics.csv", index=False)
    wm_df.to_csv(RESULT_DIR / "ablation_world_model_metrics.csv", index=False)
    graph_df.to_csv(RESULT_DIR / "ablation_graph_metrics.csv", index=False)

    summary = [
        "# Pilot HAB Ablation Results",
        "",
        "Scope: PhysioNet EEGBCI pilot, subjects 1-5, leave-subject-5-out, left/right motor imagery runs 4/8/12.",
        "These numbers are preliminary and should be reported only as pilot evidence.",
        "",
        "## Decoding and calibration",
        dataframe_to_markdown(metrics_df),
        "",
        "## Selective decision simulation",
        dataframe_to_markdown(risk_df),
        "",
        "## World-model proxy",
        dataframe_to_markdown(wm_df),
        "",
        "## Cognitive-graph proxy metrics",
        dataframe_to_markdown(graph_df),
    ]
    (RESULT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print((RESULT_DIR / "summary.md").resolve())


if __name__ == "__main__":
    run()
