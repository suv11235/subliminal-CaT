import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    b = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def make_group_folds(groups, n_splits=5):
    uniq = np.array(sorted(set(groups)))
    n_splits = max(2, min(n_splits, len(uniq)))
    folds = np.array_split(uniq, n_splits)
    out = []
    groups = np.asarray(groups)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        out.append((tr, te))
    return out


def fit_ridge(X, y, alpha=1.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd

    y_mean = float(y.mean())
    yc = y - y_mean

    # Dual ridge: solve in sample space (n x n), much faster when d >> n.
    K = Xs @ Xs.T
    K.flat[:: K.shape[0] + 1] += alpha
    a = np.linalg.solve(K, yc)
    w = Xs.T @ a
    b = y_mean
    return mu, sd, w, b


def predict_ridge(X, params):
    mu, sd, w, b = params
    Xs = (X - mu) / sd
    return Xs @ w + b


def cv_regression(X, y, groups, n_splits=5):
    y_true, y_pred = [], []
    for tr_mask, te_mask in make_group_folds(groups, n_splits=n_splits):
        params = fit_ridge(X[tr_mask], y[tr_mask], alpha=1.0)
        p = predict_ridge(X[te_mask], params)
        y_true.append(y[te_mask])
        y_pred.append(p)
    if not y_true:
        return np.nan, np.nan
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    return r2_score(yt, yp), spearman_corr(yt, yp)


def concept_holdout_regression(X, y, concepts):
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    for c in np.unique(concepts):
        te = concepts == c
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        params = fit_ridge(X[tr], y[tr], alpha=1.0)
        preds[te] = predict_ridge(X[te], params)
    m = ~np.isnan(preds)
    if m.sum() < 5:
        return np.nan, np.nan
    return r2_score(y[m], preds[m]), spearman_corr(y[m], preds[m])


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on entanglement activations (numpy ridge)")
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

    idx = labels["number_idx"].to_numpy()
    groups = labels["number"].to_numpy()
    concepts = labels["concept"].to_numpy()

    for y_name in y_metrics:
        y = labels[y_name].to_numpy(dtype=float)

        for hi, hname in enumerate(hook_names):
            for li in range(n_layers):
                X = features[idx, hi, li, :]

                r2, rho = cv_regression(X, y, groups)
                hr2, hrho = concept_holdout_regression(X, y, concepts)

                rows.append(
                    {
                        "target": y_name,
                        "hook": hname,
                        "layer": li + 1,
                        "cv_group_number_r2": r2,
                        "cv_group_number_spearman": rho,
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
