"""Run the six pipeline stages with one command (multi-platform equivalent of ``make all``).

Usage (repository root, environment installed)::

    python scripts/run_pipeline.py            # stages 1 to 4 and 6 offline; stage 5 builds images
    python scripts/run_pipeline.py --no-docker
    python scripts/run_pipeline.py --backend json   # without an MLflow installation

Each stage is a subprocess with the same interpreter; the first failure stops
the run with a non-zero exit code. Docker steps are skipped when the daemon is
not reachable and ``--no-docker`` was not given, with an explicit message.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404  (fixed argument lists, no shell)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES: list[tuple[str, list[str]]] = [
    ("1. Coleta e pré-processamento: fixture sintética", ["scripts/make_synthetic_fixture.py"]),
    (
        "1. Coleta e pré-processamento: gate de qualidade (fixture)",
        ["-m", "src.data.quality", "--fixture"],
    ),
    ("1. Coleta e pré-processamento: gate de qualidade (extrato real)", ["-m", "src.data.quality"]),
    ("2 e 3. Treino, validação, rastreamento e promoção", ["-m", "src.models.train", "--all"]),
    ("4. Testes automatizados", ["-m", "pytest", "-q"]),
    ("6. Monitoramento: cenário baseline", ["scripts/baseline_scenario.py", "--no-materialize"]),
    ("6. Monitoramento: cenário simulado", ["scripts/simulate_drift.py", "--no-materialize"]),
]

DOCKER_STAGES: list[tuple[str, list[str]]] = [
    (
        "5. Implantação: imagem da API",
        ["docker", "build", "-f", "docker/Dockerfile.api", "-t", "hypertension-api:latest", "."],
    ),
    (
        "5. Implantação: imagem da interface",
        ["docker", "build", "-f", "docker/Dockerfile.app", "-t", "hypertension-app:latest", "."],
    ),
]


def _run(title: str, argv: list[str]) -> None:
    print(f"\n=== {title}\n$ {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)  # noqa: S603  # nosec B603


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):  # Windows consoles redirected to file default to cp1252
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the whole pipeline.")
    parser.add_argument("--no-docker", action="store_true", help="skip the image builds")
    parser.add_argument("--backend", choices=["mlflow", "json"], default="mlflow")
    args = parser.parse_args(argv)
    try:
        for title, command in STAGES:
            extra = ["--backend", args.backend] if command[:2] == ["-m", "src.models.train"] else []
            _run(title, [sys.executable, *command, *extra])
        if not args.no_docker:
            if shutil.which("docker") is None:
                print("\n=== 5. Implantação: docker não encontrado; etapa pulada (--no-docker)")
            else:
                for title, command in DOCKER_STAGES:
                    _run(title, command)
    except subprocess.CalledProcessError as error:
        print(f"\nFALHA: comando terminou com código {error.returncode}", file=sys.stderr)
        return error.returncode
    print(
        "\nPipeline concluído. Suba a pilha com: docker compose -f docker/docker-compose.yml up -d"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
