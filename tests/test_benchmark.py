"""Exercise the shared CLI without contacting model or embedding services."""

from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from normal_rag.cli import ROOT, main

SHARED = ROOT.parent / "rule-chain-depth"


class BenchmarkLaunchTests(unittest.TestCase):
    def test_missing_shared_checkout_explains_how_to_select_it(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()) as error:
            with patch("normal_rag.cli.load_settings", side_effect=AssertionError("Pilot config was read")):
                status = main(["benchmark", "--rule-chain-depth-repo", folder, "--prepare-only"])
        self.assertEqual(status, 1)
        self.assertIn("--rule-chain-depth-repo", error.getvalue())

    def test_runtime_options_preserve_interpreter_symlink_and_child_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            shared = Path(folder) / "shared checkout"
            (shared / "b1").mkdir(parents=True)
            for name in ("__main__.py", "comparison.py"):
                (shared / "b1" / name).touch()
            interpreter = Path(folder) / "venv python"
            interpreter.symlink_to(sys.executable)
            relative_python = os.path.relpath(interpreter, Path.cwd())
            argv = ["benchmark", "--rule-chain-depth-repo", str(shared),
                    "--systems", "pln_rag", "rag", "full_context", "--pln-runtime", "local",
                    "--repo", folder, "--python", relative_python, "--container", "example-container",
                    "--qdrant-url", "http://127.0.0.1:6333", "--run-id", "trial", "--prepare-only"]
            for child_status in (0, 1, 2):
                with self.subTest(child_status=child_status), \
                     patch("normal_rag.benchmark.subprocess.run", return_value=Mock(returncode=child_status)) as launch:
                    self.assertEqual(main(argv), child_status)
                command = launch.call_args.args[0]
                self.assertEqual(launch.call_args.kwargs["cwd"], shared)
                self.assertEqual(command[command.index("--normal-rag-repo") + 1], str(ROOT))
                forwarded_python = Path(command[command.index("--python") + 1])
                self.assertTrue(forwarded_python.is_absolute())
                self.assertTrue(forwarded_python.is_symlink())
                self.assertEqual(forwarded_python.resolve(), Path(sys.executable).resolve())
                self.assertEqual(command[command.index("--pln-runtime") + 1], "local")
                self.assertEqual(command[command.index("--repo") + 1], folder)
                self.assertEqual(command[command.index("--container") + 1], "example-container")
                self.assertIn("--prepare-only", command)


@unittest.skipUnless((SHARED / "b1/comparison.py").is_file(),
                     "Shared CLI integration tests require the sibling rule-chain-depth checkout")
class SharedBenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.normal = self.workspace / "normal-rag"
        self.shared = self.workspace / "rule-chain-depth"
        self.caller = self.workspace / "unrelated working directory"
        self.caller.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(ROOT / "normal_rag", self.normal / "normal_rag", ignore=ignore)
        for name in ("b1", "data"):
            shutil.copytree(SHARED / name, self.shared / name, ignore=ignore)
        # The benchmark must not read this pilot configuration.
        (self.normal / ".env").write_text("This is deliberately invalid pilot configuration.\n")
        (self.shared / ".env").write_text(
            "OPENAI_API_KEY=offline-test-key\n"
            "OPENAI_MODEL=openrouter/example/configured\n"
            "OPENROUTER_PROVIDER=example\n"
            "B1_MAX_LLM_CALLS=123\n"
            "B1_LLM_MIN_INTERVAL=0\n"
        )

    def run_cli(self, *args, environment=None):
        # No real credentials or inherited model settings enter these tests.
        env = {"PYTHONPATH": str(self.normal), "PYTHONDONTWRITEBYTECODE": "1",
               **(environment or {})}
        return subprocess.run([sys.executable, "-B", "-m", "normal_rag", "benchmark", *args],
                              cwd=self.caller, env=env, capture_output=True, text=True,
                              check=False, timeout=30)

    def test_default_preparation_uses_40_questions_shared_config_and_active_baseline(self):
        result = self.run_cli("--prepare-only", "--run-id", "default-plan")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.shared / "artifacts/benchmark/default-plan"
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["stage"], "prepared_only")
        self.assertEqual(manifest["systems"], ["rag", "full_context"])
        self.assertEqual(manifest["questions_per_system"], 40)
        self.assertEqual(len(manifest["selected_world_ids"]), 20)
        self.assertTrue(manifest["full_dataset_selected"])
        self.assertEqual(manifest["baseline_settings"]["model"], "example/configured")
        oracle = json.loads((output / "oracle.json").read_text())
        self.assertEqual(oracle["question_count"], 40)
        source = self.normal / "normal_rag/core.py"
        self.assertEqual(manifest["normal_rag_source_sha256"]["core.py"],
                         hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertFalse((self.normal / "artifacts").exists())
        self.assertFalse((output / "summary.json").exists())
        self.assertFalse((output / "records.json").exists())
        for file in output.rglob("*"):
            if file.is_file():
                self.assertNotIn("offline-test-key", file.read_text())

    def test_shared_pilot_paths_and_environment_overrides_from_another_directory(self):
        relocated = self.workspace / "shared checkout elsewhere"
        self.shared.rename(relocated)
        self.shared = relocated
        inputs = self.caller / "inputs with spaces"
        inputs.mkdir()
        for name in ("worlds.json", "hand_expected.json"):
            shutil.copy2(self.shared / "data/pilot" / name, inputs / name)
        result = self.run_cli(
            "--rule-chain-depth-repo", os.path.relpath(self.shared, self.caller),
            "--systems", "rag", "--dataset", "pilot", "--world-limit", "3",
            "--worlds", "inputs with spaces/worlds.json",
            "--expected", "inputs with spaces/hand_expected.json",
            "--ollama-url", "http://127.0.0.1:19999/api/embeddings",
            "--prepare-only", "--run-id", "pilot-plan",
            environment={"OPENAI_MODEL": "openrouter/example/inherited"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.shared / "artifacts/benchmark/pilot-plan/manifest.json").read_text())
        self.assertEqual(manifest["dataset_kind"], "pilot")
        self.assertEqual(manifest["systems"], ["rag"])
        self.assertEqual(manifest["questions_per_system"], 6)
        self.assertEqual(len(manifest["selected_world_ids"]), 3)
        self.assertFalse(manifest["full_dataset_selected"])
        self.assertEqual(manifest["baseline_settings"]["model"], "example/inherited")
        self.assertEqual(manifest["baseline_settings"]["ollama_url"], "http://127.0.0.1:19999/api/embeddings")

    def test_shared_validation_failure_reaches_the_callers_exit_status(self):
        result = self.run_cli("--world-limit", "21", "--prepare-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("world-limit must be between 1 and 20", result.stderr)
        self.assertFalse((self.shared / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()
