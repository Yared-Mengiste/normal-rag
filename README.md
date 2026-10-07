# Normal RAG baseline — three-world review pilot

A conventional sentence-retrieval baseline for comparison with
[PLN-RAG](https://github.com/BekLG/PLN-RAG). This project runs independently and
does not import or modify PLN-RAG. It needs Python 3.10+ with no Python package
installation, local Ollama, and an OpenRouter key.

The current stage runs exactly three worlds for review. There is no integration
with the rule-chain-depth benchmark runner and no 40-question command. That work
requires approval of the live pilot outputs.

## Configure and run

Run from this folder:

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

Settings are read from **this project's `.env`**, even when the command is run
elsewhere. Other repositories' `.env` files and shell variables are not used.
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
still requires human review before benchmark integration.

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

## Inputs and pilot cases

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

## Review artifacts and record semantics

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
  status `AWAITING_USER_REVIEW`. It never authorizes integration automatically.

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

The tests use clearly fake in-memory HTTP responses; passing tests is not a live
benchmark result. The handwritten proof key does not override model labels.
