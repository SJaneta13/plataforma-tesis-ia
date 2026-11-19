"""Ejecuta preparación, modelado y evaluación de las cuatro configuraciones."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "outputs" / "run_registry.json"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    run("-m", "src.01_prepare_data")
    run("-m", "src.02_feature_engineering")
    run("-m", "src.02b_descriptive_tables")
    run("-m", "src.evaluation.h1_spearman")

    for target in ("5", "3"):
        for mode in ("complete", "sensitivity"):
            options = ["--target", target]
            if mode == "sensitivity":
                options.append("--sensitivity")

            run("-m", "src.03_modeling", *options)
            registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
            run_dir = registry[f"target{target}_{mode}"]
            run("-m", "src.03a_ordinal_logit", *options, "--run-dir", run_dir)

    run("-m", "src.resumir_metricas_finales")
    run("-m", "src.07_dashboard_artifacts.build_model_dashboard_artifacts")


if __name__ == "__main__":
    main()
