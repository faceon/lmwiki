# llm-wiki

The agent reads this file and builds/maintains the wiki accordingly.
See [SCHEMA.md](SCHEMA.md) for page structure, naming, and data schemas.

## Project structure

``` markdown
samples/          # example files for testing
├── source/       # raw files written by humans
└── wiki/         # wiki pages maintained by LLM
    ├── index.md  # human-readable catalog of all pages
    ├── .logs/
    │   ├── state.json    # tracks source hashes and wiki state
    │   └── runs.jsonl    # append-only event log (LLM calls, decisions)
    ├── concepts/ # recurring concepts across pages
    ├── entities/ # people, organizations, works, etc.
    └── analyses/ # reasoning and synthesis from queries
src/              # code that operates llm-wiki
├── config.py     # configuration and environment settings
├── embed.py      # embedding generation
├── ingest.py     # source ingestion pipeline
└── vectordb.py   # vector database utilities
```

## Workflow

### ingest

1. Read source file
1. Check .logs/state.json for content hash — skip if unchanged
1. Search index.md for related wiki pages — a page is related if it shares topic, entity, or concept with the source
1. For each related page found: update frontmatter, rewrite content, append to timeline
1. For each concept or entity in the source not covered by any existing page: create a new page, write content, append to timeline
1. Update index.md and .logs/state.json

### query

1. Search index.md to find relevant pages
1. Read those pages
1. Synthesize answer with citations to wiki pages
1. Save as an analysis page if the answer produces non-trivial insight or is likely to be queried again
1. Update index.md and .logs/state.json

### lint

1. Find orphan pages (not referenced in any `related` field)
1. Find missing cross-references (entity or concept mentioned in body but not in `related`)
1. Find contradictions between pages
1. Update affected pages
1. Update index.md and .logs/state.json

## Operating rules

- **.logs/state.json**: Update after every ingest, query, and lint operation
- **index.md**: Keep current after every wiki modification; written for human readers
- **Timeline**: Append only — never edit past entries
- **Source immutability**: Never modify files under `source/`
- **Cross-links**: Use `[[Page Name]]` in body text; must match the page's filename stem exactly — entities include the `@` prefix (e.g. `[[@Andrej Karpathy]]`), concepts and analyses do not
