# Normal RAG and full-context baselines

Sentence-retrieval (`rag`) and full-context (`full_context`) baselines for the
[rule-chain-depth benchmark](../rule-chain-depth/README.md), including its
**20-world, 40-question dataset**. The `benchmark` command launches the shared
runner with this checkout's answering code. Dataset validation, model settings,
request budgets, scoring, and artifacts all come from `rule-chain-depth`.

The baselines need Python 3.10+ with no Python package installation and an
OpenRouter key. RAG also needs local Ollama with `nomic-embed-text`; full context
skips embeddings. The original standalone three-world pilot is available through
`pilot` and `check`.

## Run the shared 40-question benchmark

Keep `normal-rag` and `rule-chain-depth` as sibling folders. Configure
**`../rule-chain-depth/.env`** using its [running guide](../rule-chain-depth/RUNNING.md#configuration).
From this folder:

```sh
# Validate the dataset and settings and save a plan without model or embedding calls:
python3 -m normal_rag benchmark --prepare-only

# Normal RAG only: 40 questions / 40 records:
python3 -m normal_rag benchmark --systems rag

# Full context only: 40 questions / 40 records:
python3 -m normal_rag benchmark --systems full_context

# Both baselines (the default): 40 questions each / 80 records:
python3 -m normal_rag benchmark

# Small trial: four worlds, eight questions per baseline / 16 records:
python3 -m normal_rag benchmark --world-limit 4
```

Commands without `--prepare-only` make real model calls. A prepared plan contains
no model results, and a later live command starts a new run. Baseline-only runs
do not require Docker, Qdrant, or an installed PLN-RAG runtime.

The full dataset has five supported questions and five unsupported controls in
each depth bucket, 0–3. Every world has 12 sentences and two questions. The first
four worlds cover all four depths. The canonical inputs are
[worlds.json](../rule-chain-depth/data/benchmark/worlds.json) and
[hand_expected.json](../rule-chain-depth/data/benchmark/hand_expected.json);
the shared runner validates them with both independent solvers before any model
call. It sends the baselines only sentence IDs/text and question text.

`--dataset pilot` selects rule-chain-depth's original **eight worlds / 16
questions per system**. This is a separate dataset from this project's three-world
standalone pilot.

Shared configuration uses `OPENAI_API_KEY`, `OPENAI_MODEL=openrouter/<model>`,
`OPENROUTER_PROVIDER`, `OPENROUTER_BASE_URL`, `OLLAMA_*`, and `B1_*` settings.
Inherited environment values take precedence over `rule-chain-depth/.env`.
The benchmark does not load `normal-rag/.env`. The shared adapter matches the
model, provider pin, temperature, output limit, and timeout across systems and
enforces one `B1_MAX_LLM_CALLS` cap for the run. It does not raise that cap.

Results are saved under **`../rule-chain-depth/artifacts/benchmark/<run-id>/`**:

- `manifest.json`: dataset fingerprint, selected worlds, settings, and source hashes.
- `records.json`: one result per question and system, including explicit errors.
- `summary.json` and `report.md`: overall and per-depth scores, errors, and timings.
- Per-question input, request, response, and raw model output files.

Prepared runs contain the manifest and validated inputs, without a summary or
model results. Live exit codes match the shared runner: 0 for a completed run
without error records (answers can still be wrong), 2 for a run with error
records, and 1 for setup failure. See the [shared result guide](../rule-chain-depth/RUNNING.md#read-the-results)
for request audits, proof coverage, skipped questions, and scoring details.

To include PLN-RAG in the same run:

```sh
python3 -m normal_rag benchmark --systems pln_rag rag full_context \
  --container pln-rag-pln-rag-1
```

That selects 40 questions per system, 120 records total. Follow the shared guide
for Docker access, service configuration, and a sufficient request budget.

Use `--rule-chain-depth-repo PATH` if the shared checkout is elsewhere. Its location
defaults relative to this package, independent of the current working directory.
Explicit paths supplied with `--worlds`, `--expected`, `--repo`, or `--python` are
relative to the directory where you invoke the command; omitted paths use the
shared runner's defaults. `--repo` and `--python` select the local PLN runtime.
Other options include `--pln-runtime`, `--qdrant-url`, `--ollama-url`, and `--run-id`.
Run `python3 -m normal_rag benchmark --help` for the full list.

The equivalent command from `rule-chain-depth` is:

```sh
python3 -m b1 benchmark --systems rag full_context
```

## Standalone three-world pilot

`pilot` runs the original small smoke test with this project's own input schema
and configuration. `check` validates that pilot's configuration and fixtures;
use `benchmark --prepare-only` to validate the shared experiment instead.

To configure the standalone pilot, run from this folder:

```sh
# If .env does not exist yet:
cp .env.example .env
```

Edit `.env` locally:

```dotenv
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_PROVIDER=openai
TEMPERATURE=0.1
MAX_TOKENS=256
TIMEOUT_SECONDS=60
OLLAMA_URL=http://127.0.0.1:11434/api/embeddings
EMBEDDING_TIMEOUT_SECONDS=30
```

`openai` above is an example provider slug. Choose and set one explicitly before
running. Every request sends `provider.only` and `provider.order` containing only
that slug, `allow_fallbacks: false`, and `require_parameters: true`. An unavailable
provider produces an error; there is no automatic switch or retry. The reported
provider is still recorded independently of the requested provider, and the
review gate lists all reported providers. Routing reference:
<https://openrouter.ai/docs/guides/routing/provider-selection>.

The `pilot` and `check` settings are read from **this project's `.env`**, even when
the command is run elsewhere. These two commands do not use other repositories'
`.env` files or inherited shell settings.
Values may be bare or surrounded by matching quotes; use comments on separate
lines. There is no variable interpolation. The API key is excluded from settings
reports, requests saved to disk, and configuration object representations.

Ollama must be running locally (`ollama serve` in another terminal if necessary):

```sh
ollama pull nomic-embed-text
python3 -m normal_rag check
python3 -m unittest discover -s tests -v
python3 -m normal_rag pilot --mode both --answer-key data/pilot/answer_key.json
```

`check` validates local configuration and fixtures; it does not call services or
claim that they are reachable. `pilot` makes real, potentially billable model
calls: one per question per mode, six total for `both`. It creates a fresh output
directory under `artifacts/pilot/`. A run can take several minutes: RAG embeds
each of 12 sentences and the question, with no embedding cache. Network timeouts
are socket-operation timeouts, configured separately for embeddings and answers.
HTTP proxies from the shell are not used.

Optional flags:

```sh
python3 -m normal_rag pilot --mode rag --worlds data/pilot/worlds.json \
  --answer-key data/pilot/answer_key.json --output artifacts/pilot/my-first-run
python3 -m normal_rag pilot --mode full_context --answer-key data/pilot/answer_key.json
```

The output directory must not already exist. Exit code 0 means all calls produced
valid answers, including possibly wrong answers; 2 means at least one question
has label `ERROR`; 1 means setup failed before the run could proceed. Every run
provides outputs for human review; this review status applies only to pilot results.

## What the modes do

In `rag`, each sentence is one chunk with its original ID and exact text. A fresh
in-memory index is built for every world/question invocation. Both sentences and
questions are embedded with `nomic-embed-text` using Ollama's `/api/embeddings`
request shape `{"model": "nomic-embed-text", "prompt": "original text"}`, matching
PLN-RAG's unprefixed embedding inputs. Normalized dot products give cosine
similarities. The top four sentences are selected; ties keep input order. A world
smaller than four sentences supplies all of them. There is no vector database,
reranker, query rewriting, additional retrieval, or outside knowledge source.

In `full_context`, all sentences are passed in original order without embedding
or retrieval. Both modes use the exact same system prompt, question formatting,
model, temperature, token limit, JSON mode, provider pin, and error handling.
Only the selected sentence list changes. No logic engine or proof search is used.

The model must emit exactly:

```json
{"label": "SUPPORTED", "cited": ["S1", "S2"]}
```

`NOT_ESTABLISHED` is the other allowed model label. Malformed JSON, duplicate JSON
keys, extra/missing fields, unknown labels, missing completions, truncation,
timeouts, API errors, and failed embeddings are `ERROR`. The response is never
repaired or sent to a second call. Unknown or unshown cited IDs are removed from
the normalized `cited` list and counted per occurrence, without changing a valid
model label. Raw output remains untouched. Valid repeated citations are retained.

## Standalone pilot inputs

`data/pilot/worlds.json` contains three hand-authored 12-sentence worlds:

| World | Question | Hand-checked behavior |
| --- | --- | --- |
| `direct_fact` | Is Lena blue? | S4 explicitly establishes it. |
| `two_rule_chain` | Is Mira kind? | S6 establishes patient; S9 gives calm; S2 gives kind. |
| `no_proof` | Is Mira kind? | Mira is quiet; no rule reaches patient, calm, or kind for Mira. |

These are review fixtures, not claims about model performance. Their annotations
are in a separate `data/pilot/answer_key.json`. The model receives only sentence
IDs, sentence text, and question text. It never receives the world/question ID,
expected label, proof IDs, annotation note, or depth.

The world file has this shape (the pilot requires three world entries):

```json
{
  "worlds": [
    {
      "id": "world_name",
      "sentences": [{"id": "S1", "text": "A sentence."}],
      "question": {"id": "question_name", "text": "A question?"}
    }
  ]
}
```

Each question can optionally have an entry in the separate answer key:

```json
{
  "question_name": {
    "expected_label": "SUPPORTED",
    "required_ids": ["S1"]
  }
}
```

For no-proof cases use `required_ids: null`, never an empty proof. The key is
validated before spending API calls, but passed to the measurement code only
after answering. `retrieved_contains_required` checks whether **the supplied
proof's** IDs are all retrieved. It is not an independent proof-sufficiency
test: another valid proof could use different sentences. No proof key yields
`"unmeasured"`. Full context records retrieval coverage as `"unmeasured"` and
measures `context_contains_required` separately.

## Standalone pilot artifacts and record semantics

Open `artifacts/pilot/<run-id>/review.md` for the retrieved/shown sentences, scores,
raw completions, and complete records. The directory also contains:

- `records.json`: one record per question and mode, written after each call.
- `NN-mode/request.json`: exact chat request payload without authorization headers.
- `NN-mode/response.body`: unchanged response bytes, including API error bodies.
- `NN-mode/raw_model_output.txt`: exact returned content string, with no stripping,
  reformatting, or added newline; absent if no text completion was returned.
- `NN-mode/record.json`: normalized result and diagnostics.
- `manifest.json`, `worlds.json`, optional `answer_key.json`: settings, source/input
  hashes, and the inputs needed to audit the run.
- `review_gate.json`: completion/error counts, providers reported, and the fixed
  status `AWAITING_USER_REVIEW` with `scope: pilot_outputs_only`. The shared
  benchmark is available independently of this pilot review status.

Older saved pilots may still contain `runner_integration_enabled: false`; that
historical field does not describe the current benchmark command.

`retrieved` contains IDs and cosine scores. `context` contains exactly the IDs and
texts shown in the prompt, in order. Full context has `retrieved: []` and
`index_skipped: true`; similarity scores are not invented.

`index_build_seconds` includes all sentence embedding calls and construction of
the in-memory index. It is 0 in full context because the index is skipped.
`answer_seconds` includes query embedding and ranking (RAG only), the single chat
call, and validation. A failed index records build time and leaves answer time
`"unmeasured"`, because no answer was attempted. File writes and post-answer
measurements are outside answer timing.

`requested_model` and public `settings` record intent. `model` and `provider` come
from the OpenRouter response, never from guessed defaults. The provider is read
from `provider` on a completion, or `error.metadata.provider_name` on a provider
error such as an upstream rate limit. The three normalized
token fields (`prompt_tokens`, `completion_tokens`, `total_tokens`) use reported
positive integers. Missing, zero, negative, or malformed values are
`"unmeasured"`, never a fabricated zero or derived total. The original usage
object, including any provider-specific detail or zero value, remains in
`response.body`. A missing raw completion is `null`, not a made-up answer.

The handwritten proof key does not override model labels.

## Offline validation

```sh
python3 -m unittest discover -s tests -v
```

Core tests use fake in-memory HTTP responses. CLI integration tests use the
sibling rule-chain-depth source and temporary copies of its dataset and runner;
they exercise preparation without model or embedding calls. The shared project's
own tests also cover all 120 comparison records with fake model responses.
Passing tests verifies implementation behavior, not live model accuracy.
