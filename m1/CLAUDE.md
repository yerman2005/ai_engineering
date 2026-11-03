# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A small, single-file Python example (`main.py`) of a provider-agnostic async LLM client, with OpenAI and Anthropic implementations. Code comments and prompt strings are in Spanish; keep that convention when editing.

There is no package manifest, test suite, linter config, or git repository. VS Code uses the system Python (`C:\Python314\python.exe`, see `.vscode/settings.json`), not a virtual environment. Install packages into that interpreter.

## Running

```bash
python -m pip install openai anthropic python-dotenv
python main.py
```

API keys are read from the environment (loaded from a `.env` file via `python-dotenv`): `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. Each provider runs only if its key is set. The script makes real, billed API calls.

## Architecture

The script is built around a dependency-inversion pattern:

1. `BaseLLMClient` (ABC) defines the contract: one method, `async generate_response(prompt: str) -> str`.
2. Concrete implementations:
   - `OpenAIClient`: wraps `AsyncOpenAI`, uses `chat.completions`, default model `gpt-4o-mini`.
   - `AnthropicClient`: wraps `AsyncAnthropic`, uses `messages.create`, default model `claude-sonnet-5-5`. It returns the first `text` block, because current Claude models can put `thinking` blocks before the text.
3. `main()` types each client as `BaseLLMClient` and runs the prompts concurrently with `asyncio.gather`.

To add another provider, subclass `BaseLLMClient`, implement `generate_response` as an async method, and use it from `main()`. Consumer code should depend only on the abstract interface.
