"""Launch the shared benchmark using rule-chain-depth's dataset and runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def add_benchmark_parser(commands, normal_repo: Path) -> None:
    parser = commands.add_parser(
        "benchmark",
        help="Run the shared 40-question rule-chain-depth benchmark",
        description=("Run normal RAG and full context on the shared 40-question dataset. "
                     "rule-chain-depth supplies the runner, .env, data, and artifacts."),
        epilog=("Inherited environment settings override rule-chain-depth/.env. "
                "Explicit relative file paths are resolved from your current directory. "
                "Use --prepare-only to validate and save a plan without model calls."),
    )
    parser.add_argument("--rule-chain-depth-repo", type=Path,
                        default=normal_repo.parent / "rule-chain-depth",
                        help="Shared checkout (default: sibling rule-chain-depth)")
    parser.add_argument("--systems", nargs="+", choices=("pln_rag", "rag", "full_context"),
                        default=["rag", "full_context"],
                        help="Systems to run (default: rag full_context)")
    parser.add_argument("--dataset", choices=("benchmark", "pilot"), default="benchmark",
                        help="Shared dataset: benchmark has 40 questions; pilot has 16")
    parser.add_argument("--worlds", type=Path, help="Override the shared worlds JSON file")
    parser.add_argument("--expected", type=Path, help="Override the shared annotations file")
    parser.add_argument("--world-limit", type=int,
                        help="Select the first N worlds; omitted means the full dataset")
    parser.add_argument("--pln-runtime", choices=("docker", "local"), default="docker")
    parser.add_argument("--container", help="Existing PLN-RAG container used to select its image")
    parser.add_argument("--repo", type=Path, help="PLN-RAG checkout for the local PLN runtime")
    parser.add_argument("--python", type=Path, help="Python interpreter for the local PLN runtime")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--ollama-url")
    parser.add_argument("--run-id", help="Unique name under rule-chain-depth/artifacts/benchmark")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate and save a plan without model or embedding calls")


def run_benchmark(args: argparse.Namespace, normal_repo: Path) -> int:
    root = args.rule_chain_depth_repo.expanduser().resolve()
    if not all((root / "b1" / name).is_file() for name in ("__main__.py", "comparison.py")):
        print("Shared runner not found. Set --rule-chain-depth-repo to your rule-chain-depth checkout.",
              file=sys.stderr)
        return 1

    command = [sys.executable, "-B", "-m", "b1", "benchmark",
               "--normal-rag-repo", str(normal_repo.resolve()),
               "--systems", *args.systems, "--dataset", args.dataset]
    # The child runs from the shared checkout. Preserve the caller's meaning of
    # explicit paths, including a virtualenv interpreter's symlink location.
    for name in ("worlds", "expected", "repo", "python"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name, str(value.expanduser().absolute())]
    for name in ("world_limit", "pln_runtime", "container", "qdrant_url", "ollama_url", "run_id"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if args.prepare_only:
        command.append("--prepare-only")

    # Let b1 own configuration, validation, request budgets, and result writing.
    # The subprocess inherits the environment and streams its normal output.
    try:
        return subprocess.run(command, cwd=root, check=False).returncode
    except OSError:
        print("Could not launch the shared runner. Check the checkout and Python interpreter.",
              file=sys.stderr)
        return 1
