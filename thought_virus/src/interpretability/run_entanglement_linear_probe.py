import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_regression(X, y, groups, n_splits=5):
    uniq = np.unique(groups)
    n_splits = max(2, min(n_splits, len(uniq)))
    cv = GroupKFold(n_splits=n_splits)
    y_true, y_pred = [], []
    for tr, te in cv.split(X, y, groups=groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        y_true.append(y[te])
        y_pred.append(p)
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    r2 = float(r2_score(yt, yp))
    rho = float(spearmanr(yt, yp).correlation)
    return r2, rho


def cv_classification(X, y_bin, groups, n_splits=5):
    uniq = np.unique(groups)
    n_splits = max(2, min(n_splits, len(uniq)))
    cv = GroupKFold(n_splits=n_splits)
    y_true, y_score = [], []
    for tr, te in cv.split(X, y_bin, groups=groups):
        if len(np.unique(y_bin[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y_bin[tr])
        s = clf.predict_proba(X[te])[:, 1]
        y_true.append(y_bin[te])
        y_score.append(s)
    if not y_true:
        return np.nan, np.nan
    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    if len(np.unique(yt)) < 2:
        return np.nan, np.nan
    auc = float(roc_auc_score(yt, ys))
    auprc = float(average_precision_score(yt, ys))
    return auc, auprc


def concept_holdout_regression(X, y, concepts):
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    for c in np.unique(concepts):
        te = concepts == c
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    m = ~np.isnan(preds)
    if m.sum() < 5:
        return np.nan, np.nan
    r2 = float(r2_score(y[m], preds[m]))
    rho = float(spearmanr(y[m], preds[m]).correlation)
    return r2, rho


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on entanglement activations")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ds = Path(args.dataset_dir).resolve()
    out_dir = Path(args.output).resolve() if args.output else ds
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = torch.load(ds / "features.pt", map_location="cpu")
    labels = pd.read_csv(ds / "labels.csv")

    numbers = feat["numbers"]
    hook_names = feat["hook_names"]
    features = feat["features"].numpy()

    number_to_idx = {n: i for i, n in enumerate(numbers)}
    labels = labels[labels["number"].astype(str).str.zfill(3).isin(number_to_idx)].copy()
    labels["number"] = labels["number"].astype(int).astype(str).str.zfill(3)
    labels["number_idx"] = labels["number"].map(number_to_idx)

    y_metrics = [
        "delta_vs_remove_injection_span",
        "delta_vs_replace_with_control_number",
        "delta_vs_first_chunk_only",
    ]

    rows = []
    n_layers = features.shape[2]

    for y_name in y_metrics:
        y = labels[y_name].to_numpy(dtype=float)
        groups = labels["number"].to_numpy()
        concepts = labels["concept"].to_numpy()
        thr = float(np.quantile(y, 0.75))
        y_bin = (y >= thr).astype(int)

        for hi, hname in enumerate(hook_names):
            for li in range(n_layers):
                X = features[labels["number_idx"].to_numpy(), hi, li, :]

                r2, rho = cv_regression(X, y, groups)
                auc, auprc = cv_classification(X, y_bin, groups)
                hr2, hrho = concept_holdout_regression(X, y, concepts)

                rows.append(
                    {
                        "target": y_name,
                        "hook": hname,
                        "layer": li + 1,
                        "cv_group_number_r2": r2,
                        "cv_group_number_spearman": rho,
                        "cv_group_number_auc": auc,
                        "cv_group_number_auprc": auprc,
                        "holdout_concept_r2": hr2,
                        "holdout_concept_spearman": hrho,
                        "n_samples": int(len(y)),
                    }
                )

    res = pd.DataFrame(rows)
    out_csv = out_dir / "probe_results_by_layer.csv"
    res.to_csv(out_csv, index=False)

    best = (
        res.sort_values("cv_group_number_spearman", ascending=False)
        .groupby("target", as_index=False)
        .first()
    )
    best_csv = out_dir / "probe_best_configs.csv"
    best.to_csv(best_csv, index=False)

    lines = ["# Entanglement Linear Probe Summary", ""]
    for _, r in best.iterrows():
        lines.append(f"## {r['target']}")
        lines.append(f"- best hook/layer (cv spearman): {r['hook']} / L{int(r['layer'])}")
        lines.append(f"- cv r2: {r['cv_group_number_r2']:.4f}")
        lines.append(f"- cv spearman: {r['cv_group_number_spearman']:.4f}")
        lines.append(f"- cv auc: {r['cv_group_number_auc']:.4f}")
        lines.append(f"- holdout concept r2: {r['holdout_concept_r2']:.4f}")
        lines.append(f"- holdout concept spearman: {r['holdout_concept_spearman']:.4f}")
        lines.append("")

    md = out_dir / "probe_summary.md"
    md.write_text("\n".join(lines))

    print(out_csv)
    print(best_csv)
    print(md)
    print(f"rows={len(res)}")


if __name__ == "__main__":
    main()
