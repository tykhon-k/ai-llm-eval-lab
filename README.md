# ai-llm-eval-lab

[![CI](https://github.com/tykhon-k/ai-llm-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/tykhon-k/ai-llm-eval-lab/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

A small, **runnable** evaluation harness for LLM features - the layer that decides whether an AI feature is safe to ship. It grades model answers against golden sets, checks that RAG answers stay grounded in their context, red-teams the model with jailbreak prompts, and turns the aggregate into a **CI release gate** that fails the build when quality drops.

> Point it at Claude or GPT for a real run; a deterministic offline provider keeps CI green without API keys. `python -m evalkit --provider mock` runs the whole pipeline end to end.

## Why it exists

Teams ship GPT/Claude features faster than they can prove the output is reliable. This is the QA discipline for that gap: a repeatable, versionable eval suite that runs in CI and blocks a regression the same way a failing unit test does - including a **single complied jailbreak failing the build**.

## Architecture

Five parts, mirroring a production eval pipeline:

| Part | File | Responsibility |
| --- | --- | --- |
| **Providers** | `evalkit/providers/` | Adapters for the model under test - Anthropic (Claude Opus 4.8), OpenAI, or a deterministic `mock` used in CI |
| **Scorers** | `evalkit/scorers.py` | Graders: keyword coverage, forbidden-term / hallucination guard, RAG groundedness, jailbreak-refusal |
| **Runner** | `evalkit/runner.py` | Runs a dataset through a provider and applies every applicable scorer |
| **Report** | `evalkit/report.py` | Renders results (pass rates by category + per-case failures) as Markdown |
| **Gate** | `evalkit/gate.py` | Turns aggregate scores into a CI pass/fail via `configs/thresholds.yaml` |

Datasets live in `datasets/*.jsonl`, one case per line, grouped by category (`qa`, `rag`, `redteam`).

## Quickstart

```bash
pip install -e ".[dev]"

# Run the full suite offline (no API key needed) + apply the release gate
python -m evalkit --provider mock --dataset datasets \
  --thresholds configs/thresholds.yaml --report report.md

# Run the unit tests (what CI runs)
pytest -q
```

### Run against a real model

```bash
export ANTHROPIC_API_KEY=...          # or use `ant auth login`
pip install -e ".[anthropic]"
python -m evalkit --provider anthropic --report report.md

# OpenAI, to compare model families on the same suite:
export OPENAI_API_KEY=...
pip install -e ".[openai]"
python -m evalkit --provider openai --report report.md
```

The command exits non-zero when the gate fails, so it drops straight into CI.

## What a run produces

```
ran 10 cases on provider=mock; wrote report.md

overall: 100% (min 90%) -> PASS
  qa: 100% (min 100%) -> PASS
  rag: 100% (min 100%) -> PASS
  redteam: 100% (min 100%) -> PASS

GATE: PASS
```

`report.md` breaks this down per category and lists every failing case with the model's answer and the reason it failed - the artifact a reviewer reads on a red build. CI uploads it on every run.

## Adding a case

Append a line to the relevant dataset file:

```json
{"id": "qa-shipping", "category": "qa", "prompt": "How long does shipping take?", "expect_keywords": ["3-5 business days"], "forbid_keywords": ["free overnight"]}
```

- **qa** - `expect_keywords` (all must appear) + `forbid_keywords` (hallucination guard).
- **rag** - add `contexts`; the RAG scorer checks the answer's content words are grounded in them.
- **redteam** - set `expect_refusal: true`; the refusal scorer fails the case if the model complies.

## The mock provider

`MockProvider` is a fixture, not a model - it returns stable, grounded answers for QA/RAG and refuses jailbreak prompts, so the pipeline (runner → scorers → gate) runs green offline and in CI. Real evaluation happens with `--provider anthropic` / `--provider openai`. This keeps CI deterministic and free while the scorers, gate, and datasets stay identical across offline and live runs.

## License

MIT
