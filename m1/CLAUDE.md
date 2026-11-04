# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is coursework for the Coderhouse "AI Engineering" course. The git repository root is the parent directory (`ai_engineering/`), and `program-summary.md` there is the syllabus. This folder (`m1/`) is the Module 1 pre-entrega: a "Unified Async LLM Client" for OpenAI and Anthropic. It is graded against the rubric in `program-summary.md`, section "PRE-ENTREGA · Cliente de LLM robusto y asíncrono":

- `AsyncLLMManager`
- async streaming
- Pydantic validation
- error resilience

Code, comments, docstrings, test names and the README are in Spanish. Keep that convention.

## Commands

Use the virtual environment in `.venv` (created with the system Python; the code targets 3.12+):

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest                           # all tests (no network)
.venv/Scripts/python -m pytest tests/test_manager.py::test_generate_many_limita_la_concurrencia_y_respeta_el_orden
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python main.py --provider all              # REAL, billed API calls
```

Configuration comes from the environment or `.env` (see `.env.example`). `main.py` returns exit code 2 for config errors and 1 for failed calls.

## Architecture (`llm_client/`)

- `BaseLLMClient` (`base.py`) is a template method. The public `generate()` and `stream()` apply `asyncio.timeout` and translate SDK exceptions. Subclasses implement only `_generate`, `_stream`, `_map_error` and `aclose`, and declare `provider_name` and `sdk_error_types`.
  - In streaming, the timeout applies per chunk (inside `anext`), not to the whole response.
  - The base class catches only `sdk_error_types`, so bugs are not disguised as provider errors.
- `_errors.map_sdk_error` serves both SDKs, because `openai` and `anthropic` expose the same exception classes. Order matters: `APITimeoutError` subclasses `APIConnectionError`.
- `AsyncLLMManager` (`manager.py`) is the business-facing facade.
  - `from_settings()` is a factory over the `_PROVIDERS` registry.
  - `generate()` never raises `LLMError`; it returns `LLMResult(error=ErrorInfo)` instead.
  - `stream()` raises `LLMError`, because a failure can come after chunks were already yielded.
  - `generate_many()` uses a `Semaphore`.
- `Settings` (`config.py`) stores keys as `SecretStr`. A missing key is reported by `api_key_for()` as `LLMConfigurationError`, not at validation time, so `--provider all` works with only some keys set.
- Anthropic specifics in `AnthropicClient`:
  - system messages go in the `system=` parameter;
  - only `text` blocks are joined, because `thinking` blocks can come first;
  - `temperature` is never sent, since the current SDK/API (anthropic 2.x) rejects sampling params, and a warning is logged.
- OpenAI uses `max_completion_tokens`, not the deprecated `max_tokens`.
- Clients accept `sdk_client=` for dependency injection. Tests use doubles from `tests/fakes.py` (`FakeOpenAIStream`, `FakeAnthropicStream`, `FakeLLMClient`, and helpers that build real SDK exceptions with `httpx2`).
