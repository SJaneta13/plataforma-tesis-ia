"""Compara los tres modelos sobre la misma matriz, conservando las corridas originales."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel
from threadpoolctl import threadpool_limits

from src.ordinal_inputs import select_ordinal_inputs

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "f1_weighted", "f1_macro", "mae_ordinal", "kappa_qw", "auc_ovr")


def common_design(preprocessor, train, test):
    encoded = preprocessor.fit_transform(train)
    centered = encoded - encoded.mean(axis=0)
    _, upper, pivots = qr(centered, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    tolerance = np.finfo(encoded.dtype).eps * max(encoded.shape) * diagonal.max()
    keep = np.sort(pivots[:np.count_nonzero(diagonal > tolerance)])
    return encoded[:, keep], preprocessor.transform(test)[:, keep]



def ordinal_loss(params, model):
    """Negative log likelihood and gradient for OrderedModel's threshold parametrization."""
    n_features = model.k_vars
    labels = model.endog
    cutoffs = model.transform_threshold_params(params)
    score = model.exog @ params[:n_features]
    lower = cutoffs[labels] - score
    upper = cutoffs[labels + 1] - score
    probability = model.prob(lower, upper) + 1e-20
    low_density = model.pdf(lower) / probability
    high_density = model.pdf(upper) / probability
    slope_gradient = model.exog.T @ (low_density - high_density)
    cutoff_gradient = (
        np.bincount(labels + 1, weights=high_density, minlength=len(cutoffs))
        - np.bincount(labels, weights=low_density, minlength=len(cutoffs))
    )[1:-1]
    threshold_gradient = np.cumsum(cutoff_gradient[::-1])[::-1]
    threshold_gradient[1:] *= np.exp(params[n_features + 1:])
    return -np.log(probability).sum(), -np.r_[slope_gradient, threshold_gradient]



def compare_run(run_dir, data, digest):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["input_sha256"] != digest:
        raise ValueError(f"La base difiere de la corrida {run_dir.name}")
    run_id = manifest["run_id"]
    classifiers = {
        name: joblib.load(run_dir / "models" / f"{name}_{run_id}.joblib")
        for name in ("random_forest", "svm_rbf")
    }
    features = classifiers["random_forest"].feature_names_in_.tolist()
    assert features == classifiers["svm_rbf"].feature_names_in_.tolist()
    assert len(features) == manifest["model_features"]
    ordinal = select_ordinal_inputs(data, sensitivity=manifest["sensitivity"])
    audit = {
        "corrida": run_dir.name,
        "rf_svm_originales": features,
        "ordinal_originales": ordinal.columns.tolist(),
        "solo_rf_svm": sorted(set(features) - set(ordinal)),
        "solo_ordinal": sorted(set(ordinal) - set(features)),
    }
    sklearn_history = pd.read_csv(run_dir / "tables" / f"metrics_by_seed_{run_id}.csv")
    ordinal_history = pd.read_csv(run_dir / "ordinal" / "metrics_by_seed.csv")
    target = data[manifest["target_column"]].astype(int)
    classes = sorted(target.unique())
    rows = []
    for seed in manifest["seeds"]:
        train, test, y_train, y_test = train_test_split(
            data[features], target, test_size=manifest["test_size"], stratify=target, random_state=seed
        )
        x_train, x_test = common_design(clone(classifiers["random_forest"]["prep"]), train, test)
        original_ordinal = ordinal_history.loc[ordinal_history["seed"].eq(seed)].iloc[0]
        link = original_ordinal["distribution"]
        model = OrderedModel(y_train.to_numpy(), x_train, distr=link)
        fit = minimize(ordinal_loss, model.start_params, args=(model,), jac=True,
                       method="L-BFGS-B", options={"maxiter": 1500, "ftol": 1e-10})
        if not fit.success:
            raise RuntimeError(f"Ordinal sin convergencia: {run_dir.name}, semilla {seed}")

        probabilities = np.asarray(model.predict(fit.x, exog=x_test))
        predictions = {"ordinal_logit": (np.asarray(classes)[probabilities.argmax(axis=1)], probabilities, original_ordinal)}
        for name, pipeline in classifiers.items():
            original = sklearn_history.loc[sklearn_history["model"].eq(name) & sklearn_history["seed"].eq(seed)].iloc[0]
            params = {key.removeprefix("clf__"): val for key, val in json.loads(original["best_params"]).items()}
            estimator = clone(pipeline["clf"]).set_params(random_state=seed, **params)
            estimator.fit(x_train, y_train)
            predictions[name] = (estimator.predict(x_test), estimator.predict_proba(x_test), original)

        for name, (prediction, proba, original) in predictions.items():
            values = (
                accuracy_score(y_test, prediction),
                f1_score(y_test, prediction, average="weighted", zero_division=0),
                f1_score(y_test, prediction, average="macro", zero_division=0),
                mean_absolute_error(y_test, prediction),
                cohen_kappa_score(y_test, prediction, weights="quadratic"),
                roc_auc_score(y_test, proba, multi_class="ovr", average="weighted", labels=classes),
            )
            rows.append({
                "run": run_dir.name, "levels": len(classes), "mode": manifest["analysis_mode"],
                "seed": seed, "model": name, "ordinal_link": link if name == "ordinal_logit" else "",
                "raw_features": len(features), "common_design_columns": x_train.shape[1],
                **{f"{key}_original": float(original[key]) for key in METRICS},
                **{f"{key}_common": val for key, val in zip(METRICS, values)},
            })
        print(f"{run_dir.name} · semilla {seed}", flush=True)
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation" / "common_matrix")
    args = parser.parse_args()
    registry = json.loads((ROOT / "outputs" / "run_registry.json").read_text(encoding="utf-8"))
    source = ROOT / "data" / "processed" / "encuestas" / "model_ready_no_text.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = pd.read_csv(source)
    rows, audits = [], []
    with threadpool_limits(limits=1):
        for run_name in registry.values():
            result, audit = compare_run(ROOT / "outputs" / "runs" / run_name, data, digest)
            rows.extend(result)
            audits.append(audit)
            print(run_name, len(result), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "metrics_by_seed.csv", index=False)
    metric_columns = [f"{metric}_{version}" for metric in METRICS for version in ("original", "common")]
    summary = result.groupby(["levels", "mode", "model"])[metric_columns].mean()
    summary.to_csv(args.output / "comparison.csv")
    (args.output / "predictors.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
