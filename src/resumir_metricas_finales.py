from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs" / "runs"
REGISTRY = ROOT / "outputs" / "run_registry.json"
OUT_DIR = ROOT / "outputs" / "latest" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "svm_rbf": "SVM-RBF",
    "ordinal_logit": "Modelo ordinal",
    "log_reg": "Regresión logística nominal",
    "hist_gradient_boosting": "Gradient Boosting",
    "voting_ensemble": "Voting Ensemble",
    "baseline_majority": "Baseline mayoría",
    "baseline_stratified": "Baseline estratificado",
}

THESIS_MODELS = {
    "random_forest",
    "svm_rbf",
    "ordinal_logit",
}

def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontró {pattern} en {folder}")
    return files[-1]

def load_registry() -> dict:
    if not REGISTRY.exists():
        raise FileNotFoundError(f"No existe: {REGISTRY}")
    return json.loads(REGISTRY.read_text(encoding="utf-8"))

def normalize_general(df: pd.DataFrame, config: str, mode: str, run_name: str) -> pd.DataFrame:
    out = df.copy()
    out["Configuración"] = config
    out["Modo"] = mode
    out["Corrida"] = run_name
    out["Modelo_codigo"] = out["model"].astype(str)
    out["Modelo"] = out["Modelo_codigo"].map(MODEL_LABELS).fillna(out["Modelo_codigo"])

    rename = {
        "accuracy_mean": "Accuracy",
        "accuracy_std": "Accuracy_std",
        "f1_weighted_mean": "F1_ponderado",
        "f1_weighted_std": "F1_ponderado_std",
        "f1_macro_mean": "F1_macro",
        "f1_macro_std": "F1_macro_std",
        "mae_ordinal_mean": "MAE_ordinal",
        "mae_ordinal_std": "MAE_ordinal_std",
        "kappa_qw_mean": "QWK",
        "kappa_qw_std": "QWK_std",
        "auc_ovr_mean": "AUC_OvR",
        "auc_ovr_std": "AUC_OvR_std",
        "n_total": "n",
        "n_features": "Predictores",
    }
    out = out.rename(columns=rename)
    return out

def normalize_ordinal(df: pd.DataFrame, config: str, mode: str, run_name: str) -> pd.DataFrame:
    out = df.copy()
    out["Configuración"] = config
    out["Modo"] = mode
    out["Corrida"] = run_name
    out["Modelo_codigo"] = "ordinal_logit"
    out["Modelo"] = "Modelo ordinal"

    rename = {
        "accuracy_mean": "Accuracy",
        "accuracy_std": "Accuracy_std",
        "f1_weighted_mean": "F1_ponderado",
        "f1_weighted_std": "F1_ponderado_std",
        "f1_macro_mean": "F1_macro",
        "f1_macro_std": "F1_macro_std",
        "mae_ordinal_mean": "MAE_ordinal",
        "mae_ordinal_std": "MAE_ordinal_std",
        "kappa_qw_mean": "QWK",
        "kappa_qw_std": "QWK_std",
        "auc_ovr_mean": "AUC_OvR",
        "auc_ovr_std": "AUC_OvR_std",
        "n_total": "n",
        "n_features": "Predictores",
    }
    out = out.rename(columns=rename)
    return out

def main() -> None:
    registry = load_registry()

    specs = [
        ("target5_complete", "5 niveles", "Completo"),
        ("target5_sensitivity", "5 niveles", "Sensibilidad"),
        ("target3_complete", "3 niveles", "Completo"),
        ("target3_sensitivity", "3 niveles", "Sensibilidad"),
    ]

    frames = []

    for key, config, mode in specs:
        run_name = registry.get(key)
        if not run_name:
            raise KeyError(f"Falta {key} en run_registry.json")

        run_dir = RUNS_DIR / run_name
        general_path = latest_file(run_dir / "tables", "metrics_run_*.csv")
        general = pd.read_csv(general_path, encoding="utf-8-sig")
        frames.append(normalize_general(general, config, mode, run_name))

        ordinal_path = run_dir / "ordinal" / "metrics.csv"
        if not ordinal_path.exists():
            raise FileNotFoundError(f"No existe: {ordinal_path}")
        ordinal = pd.read_csv(ordinal_path, encoding="utf-8-sig")
        frames.append(normalize_ordinal(ordinal, config, mode, run_name))

    all_models = pd.concat(frames, ignore_index=True, sort=False)

    columns = [
        "Configuración",
        "Modo",
        "Modelo",
        "Modelo_codigo",
        "Accuracy",
        "Accuracy_std",
        "F1_ponderado",
        "F1_ponderado_std",
        "F1_macro",
        "F1_macro_std",
        "MAE_ordinal",
        "MAE_ordinal_std",
        "QWK",
        "QWK_std",
        "AUC_OvR",
        "AUC_OvR_std",
        "n",
        "Predictores",
        "Corrida",
    ]

    for col in columns:
        if col not in all_models.columns:
            all_models[col] = pd.NA

    all_models = all_models[columns]
    all_models.to_csv(
        OUT_DIR / "comparacion_final_todos_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    thesis = all_models[
        all_models["Modelo_codigo"].isin(THESIS_MODELS)
    ].copy()

    thesis.to_csv(
        OUT_DIR / "comparacion_final_modelos_tesis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_cols = [
        "Configuración",
        "Modo",
        "Modelo",
        "Accuracy",
        "F1_ponderado",
        "MAE_ordinal",
        "QWK",
        "AUC_OvR",
    ]

    print("\nCOMPARACIÓN FINAL — MODELOS DE TESIS\n")
    print(
        thesis[display_cols]
        .sort_values(["Configuración", "Modo", "Modelo"])
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nARCHIVOS GENERADOS")
    print(OUT_DIR / "comparacion_final_todos_modelos.csv")
    print(OUT_DIR / "comparacion_final_modelos_tesis.csv")

if __name__ == "__main__":
    main()
