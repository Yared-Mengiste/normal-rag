"""Offline contract tests with fake responses; these are not benchmark results."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from urllib.error import HTTPError

from normal_rag.cli import ROOT, pilot, read_key, read_pilot
from normal_rag.config import Settings, load_settings
from normal_rag.core import (Failure, HTTP, Response, UNMEASURED, add_proof_coverage,
                             answer_question, parse_answer, usage_from)


def envelope(raw='{"label":"SUPPORTED","cited":["S1"]}', **overrides):
    result = {"id": "fake-offline-id", "model": "reported-model", "provider": "Reported Provider",
              "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}}
    result.update(overrides)
    return Response(200, json.dumps(result).encode())


class FakeHTTP:
    def __init__(self, completion=None, vectors=None, failure=None):
        self.completion = completion or envelope()
        self.vectors = vectors or {}
        self.failure = failure
        self.calls = []

    def post(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        if "prompt" in payload:
            vector = self.vectors.get(payload["prompt"], [1, 0])
            if isinstance(vector, Failure):
                raise vector
            return Response(200, json.dumps({"embedding": vector}).encode())
        if self.failure:
            raise self.failure
        return self.completion


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(api_key="test-key-never-logged", provider="chosen-provider")
        self.sentences = [{"id": f"S{i}", "text": f"Sentence {i}."} for i in range(1, 7)]

    def run_answer(self, raw, mode="full_context"):
        return answer_question(self.sentences, "Claim?", mode, self.settings, FakeHTTP(envelope(raw)))

    def test_cosine_top_four_and_raw_sentence_embedding(self):
        vectors = {"Claim?": [2, 0], "Sentence 1.": [100, 100], "Sentence 2.": [1, 0],
                   "Sentence 3.": [-1, 0], "Sentence 4.": [0, 1],
                   "Sentence 5.": [1, 1], "Sentence 6.": [3, 1]}
        http = FakeHTTP(vectors=vectors)
        answer = answer_question(self.sentences, "Claim?", "rag", self.settings, http)
        self.assertEqual([r["id"] for r in answer.record["retrieved"]], ["S2", "S6", "S1", "S5"])
        self.assertAlmostEqual(answer.record["retrieved"][2]["score"], 2 ** -0.5)
        embedding_calls = http.calls[:-1]
        self.assertEqual([c["payload"]["prompt"] for c in embedding_calls],
                         [s["text"] for s in self.sentences] + ["Claim?"])
        self.assertTrue(all(c["payload"]["model"] == "nomic-embed-text" for c in embedding_calls))
        self.assertTrue(all(c["headers"] == {} for c in embedding_calls))
        self.assertEqual(len(http.calls), 8)

    def test_fresh_index_and_no_world_leakage(self):
        http = FakeHTTP()
        answer_question(self.sentences, "First?", "rag", self.settings, http)
        other = [{"id": "S1", "text": "A different world's fact."}]
        answer = answer_question(other, "Second?", "rag", self.settings, http)
        self.assertEqual(len(http.calls), 8 + 3)
        self.assertEqual(answer.record["context"], other)
        self.assertNotIn("Sentence 1.", json.dumps(answer.request))

    def test_full_context_skips_embeddings_and_retrieval(self):
        http = FakeHTTP()
        answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, http)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(answer.record["retrieved"], [])
        self.assertEqual(answer.record["context"], self.sentences)
        self.assertEqual(answer.record["index_build_seconds"], 0)
        self.assertTrue(answer.record["index_skipped"])

    def test_modes_have_identical_prompt_and_request_settings(self):
        rag = answer_question(self.sentences[:3], "Claim?", "rag", self.settings, FakeHTTP())
        full = answer_question(self.sentences[:3], "Claim?", "full_context", self.settings, FakeHTTP())
        self.assertEqual(rag.request, full.request)
        self.assertEqual(rag.request["provider"], {"order": ["chosen-provider"], "only": ["chosen-provider"],
                                                  "allow_fallbacks": False, "require_parameters": True})

    def test_settings_drive_request_and_metadata_comes_from_response(self):
        settings = replace(self.settings, model="different/model", temperature=0.2, max_tokens=90)
        answer = answer_question(self.sentences, "Claim?", "full_context", settings, FakeHTTP())
        self.assertEqual(answer.request["model"], "different/model")
        self.assertEqual(answer.request["temperature"], 0.2)
        self.assertEqual(answer.request["max_tokens"], 90)
        self.assertEqual(answer.record["model"], "reported-model")
        self.assertEqual(answer.record["provider"], "Reported Provider")
        self.assertEqual(answer.record["usage"]["total_tokens"], 132)
        self.assertNotIn(settings.api_key, json.dumps(answer.record))
        self.assertNotIn(settings.api_key, json.dumps(answer.request))

    def test_raw_output_and_body_preserved_exactly(self):
        raw = ' \n { "label" : "SUPPORTED", "cited" : ["S1"] }\t\n'
        response = envelope(raw)
        answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, FakeHTTP(response))
        self.assertEqual(answer.record["raw_model_output"], raw)
        self.assertEqual(answer.response_body, response.body)

    def test_invalid_citations_dropped_only_from_normalized_record(self):
        raw = '{"label":"SUPPORTED","cited":["S1","S99","S99","S6"]}'
        answer = self.run_answer(raw, mode="rag")
        self.assertEqual(answer.record["label"], "SUPPORTED")
        self.assertEqual(answer.record["cited"], ["S1"])
        self.assertEqual(answer.record["invalid_citation_count"], 3)
        self.assertEqual(answer.record["raw_model_output"], raw)

    def test_bad_output_is_error_without_a_repair_call(self):
        bad = ["not json", '```json\n{"label":"SUPPORTED","cited":[]}\n```',
               '{"cited":[]}', '{"label":"MAYBE","cited":[]}',
               '{"label":"SUPPORTED"}', '{"label":"SUPPORTED","cited":"S1"}',
               '{"label":"SUPPORTED","cited":[1]}', '{"label":"SUPPORTED","cited":[],"extra":1}',
               '{"label":"SUPPORTED","label":"NOT_ESTABLISHED","cited":[]}',
               '{"label":"SUPPORTED","cited":[],"extra":NaN}', "[]", "null"]
        for raw in bad:
            with self.subTest(raw=raw):
                http = FakeHTTP(envelope(raw))
                answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, http)
                self.assertEqual(answer.record["label"], "ERROR")
                self.assertEqual(len(http.calls), 1)
                self.assertEqual(answer.record["raw_model_output"], raw)

    def test_citations_still_counted_with_missing_label(self):
        answer = self.run_answer('{"cited":["S1","invented"]}')
        self.assertEqual(answer.record["label"], "ERROR")
        self.assertEqual(answer.record["cited"], ["S1"])
        self.assertEqual(answer.record["invalid_citation_count"], 1)

    def test_not_established_is_a_valid_model_decision(self):
        answer = self.run_answer('{"label":"NOT_ESTABLISHED","cited":[]}')
        self.assertEqual(answer.record["label"], "NOT_ESTABLISHED")
        self.assertIsNone(answer.record["error_kind"])

    def test_missing_usage_model_provider_are_unmeasured_not_inferred(self):
        response = envelope(model=None, provider=None, usage={"prompt_tokens": 7, "completion_tokens": 0})
        answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, FakeHTTP(response))
        self.assertEqual(answer.record["model"], UNMEASURED)
        self.assertEqual(answer.record["provider"], UNMEASURED)
        self.assertEqual(answer.record["usage"], {"prompt_tokens": 7, "completion_tokens": UNMEASURED,
                                                 "total_tokens": UNMEASURED})
        for usage in (None, {}, [], {"total_tokens": True}, {"total_tokens": -1}):
            self.assertTrue(all(v == UNMEASURED for v in usage_from({"usage": usage}).values()))

    def test_api_failures_preserve_response_and_do_not_retry(self):
        responses = [Response(429, b'{"error":{"message":"rate limited"}}'),
                     Response(200, b'{"error":{"message":"provider error"}}'),
                     Response(502, b"bad gateway"), Response(200, b"not json"),
                     envelope(choices=[]), envelope(raw=None)]
        for response in responses:
            with self.subTest(body=response.body):
                http = FakeHTTP(response)
                answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, http)
                self.assertEqual(answer.record["label"], "ERROR")
                self.assertEqual(answer.response_body, response.body)
                self.assertEqual(len(http.calls), 1)

    def test_timeout_is_error_and_measured(self):
        http = FakeHTTP(failure=Failure("timeout"))
        answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, http)
        self.assertEqual(answer.record["label"], "ERROR")
        self.assertEqual(answer.record["error_kind"], "timeout")
        self.assertIsInstance(answer.record["answer_seconds"], float)
        self.assertEqual(len(http.calls), 1)

    def test_provider_is_recorded_from_openrouter_error_metadata(self):
        response = Response(429, json.dumps({"error": {"message": "Rate limited", "metadata": {
            "provider_name": "Google AI Studio"}}}).encode())
        answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, FakeHTTP(response))
        self.assertEqual(answer.record["label"], "ERROR")
        self.assertEqual(answer.record["provider"], "Google AI Studio")
        self.assertEqual(answer.record["model"], UNMEASURED)
        self.assertTrue(all(v == UNMEASURED for v in answer.record["usage"].values()))

    def test_absent_or_malformed_error_provider_remains_unmeasured(self):
        for error in ("bad", {"metadata": None}, {"metadata": {"provider_name": []}}):
            response = Response(429, json.dumps({"error": error}).encode())
            answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, FakeHTTP(response))
            self.assertEqual(answer.record["provider"], UNMEASURED)
            self.assertEqual(answer.record["label"], "ERROR")

    def test_truncated_json_completion_is_error_even_if_json_parses(self):
        response = envelope(choices=[{"message": {"content": '{"label":"SUPPORTED","cited":[]}'},
                                     "finish_reason": "length"}])
        answer = answer_question(self.sentences, "Claim?", "full_context", self.settings, FakeHTTP(response))
        self.assertEqual(answer.record["error_kind"], "incomplete_completion")
        self.assertEqual(answer.record["label"], "ERROR")

    def test_bad_embeddings_never_call_llm(self):
        for vector in ([0, 0], [], [True, 1], [float("nan"), 1], "bad", Failure("timeout")):
            with self.subTest(vector=vector):
                http = FakeHTTP(vectors={"Sentence 1.": vector})
                answer = answer_question(self.sentences, "Claim?", "rag", self.settings, http)
                self.assertEqual(answer.record["label"], "ERROR")
                self.assertEqual(len(http.calls), 1)
                self.assertIsNone(answer.request)
                self.assertEqual(answer.record["answer_seconds"], UNMEASURED)

    def test_mismatched_dimensions_are_errors(self):
        for vectors in ({"Sentence 2.": [1, 2, 3]}, {"Claim?": [1, 2, 3]}):
            http = FakeHTTP(vectors=vectors)
            answer = answer_question(self.sentences, "Claim?", "rag", self.settings, http)
            self.assertEqual(answer.record["error_kind"], "embedding_dimension_mismatch")
            self.assertIsNone(answer.request)

    def test_duplicate_sentence_ids_rejected_before_network(self):
        http = FakeHTTP()
        with self.assertRaises(ValueError):
            answer_question(self.sentences + self.sentences[:1], "Claim?", "rag", self.settings, http)
        self.assertEqual(http.calls, [])

    def test_proof_measurement_happens_after_answer_and_does_not_affect_prompt(self):
        answer = answer_question(self.sentences, "Claim?", "rag", self.settings, FakeHTTP())
        original_request = json.dumps(answer.request)
        add_proof_coverage(answer.record, ["S1", "S2"])
        self.assertTrue(answer.record["proof_coverage"]["retrieved_contains_required"])
        add_proof_coverage(answer.record, ["S1", "S6"])
        self.assertFalse(answer.record["proof_coverage"]["retrieved_contains_required"])
        self.assertEqual(original_request, json.dumps(answer.request))
        self.assertNotIn("required_ids", original_request)
        self.assertNotIn("expected_label", original_request)
        self.assertNotIn("depth", original_request)

    def test_no_proof_and_full_context_coverage_do_not_claim_retrieval_success(self):
        answer = self.run_answer('{"label":"NOT_ESTABLISHED","cited":[]}')
        add_proof_coverage(answer.record, None)
        self.assertEqual(answer.record["proof_coverage"]["retrieved_contains_required"], UNMEASURED)
        add_proof_coverage(answer.record, ["S6"])
        self.assertEqual(answer.record["proof_coverage"]["retrieved_contains_required"], UNMEASURED)
        self.assertTrue(answer.record["proof_coverage"]["context_contains_required"])

    def test_pilot_inputs_and_review_stage_limit(self):
        worlds = read_pilot(ROOT / "data/pilot/worlds.json")
        key = read_key(ROOT / "data/pilot/answer_key.json", worlds)
        self.assertEqual([len(w["sentences"]) for w in worlds], [12, 12, 12])
        self.assertEqual(len(key["two_rule_chain_q"]["required_ids"]), 3)
        self.assertIsNone(key["no_proof_q"]["required_ids"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worlds.json"
            path.write_text(json.dumps({"worlds": worlds * 14}))
            with self.assertRaises(ValueError):
                read_pilot(path)
            worlds[0]["question"]["expected_label"] = "SUPPORTED"
            path.write_text(json.dumps({"worlds": worlds}))
            with self.assertRaises(ValueError):
                read_pilot(path)

    def test_cli_artifacts_preserve_raw_and_always_require_review(self):
        raw = '\n{"label":"NOT_ESTABLISHED","cited":[]}  '
        http = FakeHTTP(envelope(raw))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with patch("normal_rag.cli.load_settings", return_value=self.settings), \
                    patch("normal_rag.core.HTTP", return_value=http):
                status = pilot(ROOT / "data/pilot/worlds.json", ROOT / "data/pilot/answer_key.json", "both", output)
            self.assertEqual(status, 0)  # Wrong answers are review results, not infrastructure failures.
            records = json.loads((output / "records.json").read_text())
            self.assertEqual(len(records), 6)
            self.assertEqual((output / "01-rag/raw_model_output.txt").read_bytes(), raw.encode())
            gate = json.loads((output / "review_gate.json").read_text())
            self.assertEqual(gate["status"], "AWAITING_USER_REVIEW")
            self.assertFalse(gate["runner_integration_enabled"])
            requests = list(output.glob("*/request.json"))
            self.assertEqual(len(requests), 6)
            for request in requests:
                text = request.read_text()
                self.assertNotIn("expected_label", text)
                self.assertNotIn("required_ids", text)
                self.assertNotIn(self.settings.api_key, text)
            self.assertTrue((output / "review.md").is_file())
            with patch("normal_rag.cli.load_settings", return_value=self.settings):
                with self.assertRaises(FileExistsError):
                    pilot(ROOT / "data/pilot/worlds.json", None, "both", output)


class ConfigTests(unittest.TestCase):
    def test_dotenv_used_without_shell_override_or_secret_repr(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text('OPENROUTER_API_KEY="private-test-key"\nOPENROUTER_PROVIDER=openai\n')
            with patch.dict("os.environ", {"OPENROUTER_MODEL": "wrong/model"}):
                settings = load_settings(path)
            self.assertEqual(settings.model, "openai/gpt-4o-mini")
            self.assertEqual(settings.temperature, 0.1)
            self.assertNotIn(settings.api_key, repr(settings))
            self.assertNotIn(settings.api_key, json.dumps(settings.public()))

    def test_missing_provider_is_not_guessed(self):
        with self.assertRaises(ValueError):
            Settings(api_key="a-test-key")

    def test_remote_embeddings_rejected(self):
        with self.assertRaises(ValueError):
            Settings(api_key="a-test-key", provider="openai", ollama_url="https://openrouter.ai/api/v1/embeddings")


class TransportTests(unittest.TestCase):
    def test_http_error_body_is_preserved(self):
        body = b'{"error":"rate limited"}'
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.test", 429, "limited", {}, io.BytesIO(body))
        with patch("normal_rag.core.build_opener", return_value=opener):
            response = HTTP().post("https://example.test", {}, {}, 1)
        self.assertEqual(response.status, 429)
        self.assertEqual(response.body, body)
        self.assertEqual(opener.open.call_count, 1)

    def test_timeout_while_reading_error_body_is_classified(self):
        stream = Mock()
        stream.closed = False
        stream.read.side_effect = TimeoutError()
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.test", 502, "gateway", {}, stream)
        with patch("normal_rag.core.build_opener", return_value=opener):
            with self.assertRaises(Failure) as result:
                HTTP().post("https://example.test", {}, {}, 1)
        self.assertEqual(result.exception.kind, "timeout")
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
