from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

from .config import load_settings
from .core import UNMEASURED, add_proof_coverage, answer_question, strict_json, validate_sentences

ROOT = Path(__file__).resolve().parent.parent


def read_pilot(path: Path) -> list[dict]:
    payload = strict_json(path.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {"worlds"}:
        raise ValueError("Pilot input must contain only worlds; keep answer keys in a separate file")
    worlds = payload["worlds"]
    if not isinstance(worlds, list) or len(worlds) != 3:
        raise ValueError("This review-stage CLI requires exactly three worlds, one question per world")
    world_ids, question_ids = set(), set()
    for world in worlds:
        if not isinstance(world, dict) or set(world) != {"id", "sentences", "question"}:
            raise ValueError("Each world must contain only id, sentences, and question")
        if not isinstance(world["id"], str) or not world["id"].strip() or world["id"] in world_ids:
            raise ValueError("World IDs must be unique, nonempty strings")
        world_ids.add(world["id"])
        validate_sentences(world["sentences"])
        question = world["question"]
        if not isinstance(question, dict) or set(question) != {"id", "text"}:
            raise ValueError("Each question must contain only id and text")
        if any(not isinstance(question[k], str) or not question[k].strip() for k in ("id", "text")):
            raise ValueError("Question ID and text must be nonempty strings")
        if question["id"] in question_ids:
            raise ValueError("Question IDs must be unique across worlds")
        question_ids.add(question["id"])
    return worlds


def read_key(path: Path | None, worlds: list[dict]) -> dict:
    if path is None:
        return {}
    key = strict_json(path.read_bytes())
    available = {w["question"]["id"]: {s["id"] for s in w["sentences"]} for w in worlds}
    if not isinstance(key, dict) or not set(key) <= available.keys():
        raise ValueError("Answer key must map pilot question IDs to annotations")
    for qid, row in key.items():
        if not isinstance(row, dict) or not set(row) <= {"expected_label", "required_ids", "note"}:
            raise ValueError("Answer key annotations must use expected_label, required_ids, or note")
        if row.get("expected_label") not in (None, "SUPPORTED", "NOT_ESTABLISHED"):
            raise ValueError("Invalid expected label in answer key")
        required = row.get("required_ids")
        if required is not None:
            if (not isinstance(required, list) or not required
                    or any(not isinstance(s, str) for s in required)
                    or len(required) != len(set(required)) or not set(required) <= available[qid]):
                raise ValueError("required_ids must be null or a nonempty set of known sentence IDs")
            if row.get("expected_label") == "NOT_ESTABLISHED":
                raise ValueError("A no-proof case must use null for required_ids")
    return key


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_review(records: list[dict], output: Path) -> None:
    lines = ["# Three-world baseline pilot", "",
             "Live calls only. Review these outputs before enabling benchmark integration.", "",
             "Full context skips retrieval; its retrieved list is empty. Context lists show exactly what the model saw.", ""]
    for index, record in enumerate(records, 1):
        lines += [f"## {record['world_id']} / {record['question_id']} / {record['mode']}", "",
                  record["question"], "",
                  f"Label: {record['label']}; reported model: {record['model']}; provider: {record['provider']}.", "",
                  "Sentences shown to the model, in prompt order:", ""]
        scores = {r["id"]: r["score"] for r in record["retrieved"]}
        for sentence in record["context"]:
            score = f" (cosine {scores[sentence['id']]:.8f})" if sentence["id"] in scores else ""
            lines.append(f"- {sentence['id']}{score}: {sentence['text']}")
        raw = record["raw_model_output"]
        response_path = f"{index:02d}-{record['mode']}/response.body"
        if (output / response_path).exists():
            lines += ["", f"[Untouched API response, including any error]({response_path})"]
        lines += ["", f"[Untouched model output]({index:02d}-{record['mode']}/raw_model_output.txt)"
                  if isinstance(raw, str) else "No text completion was returned.", "",
                  "````text", raw if isinstance(raw, str) else repr(raw), "````", "",
                  "Record:", "", "````json", json.dumps(record, ensure_ascii=False, indent=2), "````", ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")


def pilot(worlds_path: Path, key_path: Path | None, mode: str, output: Path) -> int:
    settings = load_settings(ROOT / ".env")
    worlds = read_pilot(worlds_path)
    key = read_key(key_path, worlds)
    modes = ("rag", "full_context") if mode == "both" else (mode,)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "worlds.json", {"worlds": worlds})
    manifest = {"stage": "three_world_review_only", "created_at": datetime.now(timezone.utc).isoformat(),
                "settings": settings.public(), "modes": modes,
                "worlds_sha256": hashlib.sha256(worlds_path.read_bytes()).hexdigest(),
                "answer_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest() if key_path else None,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "normal_rag").glob("*.py"))}}
    write_json(output / "manifest.json", manifest)
    if key_path:
        write_json(output / "answer_key.json", key)
    records = []
    for world in worlds:
        for selected_mode in modes:
            q = world["question"]
            print(f"Running {world['id']} / {q['id']} / {selected_mode}", flush=True)
            answer = answer_question(world["sentences"], q["text"], selected_mode, settings)
            record = answer.record
            record.update(world_id=world["id"], question_id=q["id"], question=q["text"])
            # Evaluation data enters only after the model call has completed.
            annotation = key.get(q["id"], {})
            add_proof_coverage(record, annotation.get("required_ids"))
            if annotation.get("expected_label"):
                record["expected_label"] = annotation["expected_label"]
                record["correct"] = record["label"] != "ERROR" and record["label"] == annotation["expected_label"]
            records.append(record)
            operation = output / f"{len(records):02d}-{selected_mode}"
            operation.mkdir()
            if answer.request is not None:
                write_json(operation / "request.json", answer.request)  # No Authorization header.
            if answer.response_body is not None:
                (operation / "response.body").write_bytes(answer.response_body)
            if isinstance(record["raw_model_output"], str):
                (operation / "raw_model_output.txt").write_bytes(record["raw_model_output"].encode("utf-8"))
            write_json(operation / "record.json", record)
            write_json(output / "records.json", records)
            render_review(records, output)
            providers = Counter(r["provider"] for r in records)
            write_json(output / "review_gate.json", {
                "status": "AWAITING_USER_REVIEW", "runner_integration_enabled": False,
                "completed_records": len(records), "planned_records": len(worlds) * len(modes),
                "error_count": sum(r["label"] == "ERROR" for r in records),
                "providers_reported": dict(providers),
                "provider_consistency": (
                    "unmeasured" if UNMEASURED in providers
                    else "single_reported_provider" if len(providers) == 1 else "multiple_reported_providers"),
            })
            print(f"  {record['label']}; provider={record['provider']}; error={record['error_kind']}", flush=True)
    print(f"Review: {output / 'review.md'}")
    return 2 if any(record["label"] == "ERROR" for record in records) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone three-world RAG/full-context pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("pilot", help="Run exactly three worlds for user review")
    run.add_argument("--worlds", type=Path, default=ROOT / "data/pilot/worlds.json")
    run.add_argument("--answer-key", type=Path, help="Optional annotations, added after answering")
    run.add_argument("--mode", choices=("rag", "full_context", "both"), default="both")
    run.add_argument("--output", type=Path)
    commands.add_parser("check", help="Validate .env and pilot input without any network calls")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            settings = load_settings(ROOT / ".env")
            worlds = read_pilot(ROOT / "data/pilot/worlds.json")
            read_key(ROOT / "data/pilot/answer_key.json", worlds)
            print(json.dumps({"configuration_valid": True, "live_services_checked": False,
                              "settings": settings.public()}, indent=2))
            return 0
        output = args.output or ROOT / "artifacts/pilot" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
        return pilot(args.worlds, args.answer_key, args.mode, output)
    except (ValueError, OSError) as exc:
        # Paths, configuration values, and exception bodies are not dumped to logs.
        print(f"Setup stopped ({type(exc).__name__}). Check .env, input files, and a fresh output path.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
