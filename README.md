# llm-wiki

Inspired by [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

Drop markdown files into `samples/source/`. The agent reads them and builds a structured wiki of concepts, entities, and analyses — automatically cross-linked.

## Setup

```bash
uv sync
```

Configure paths and LLM endpoint in `.env`:

## Usage

```bash
uv run src/ingest.py
```

See [SCHEMA.md](SCHEMA.md) for page structure and naming conventions, and [AGENTS.md](AGENTS.md) for agent workflow details.
