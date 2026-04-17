# Schema

## Naming convention

- Wiki page filenames use natural Title Case: `LLM Wiki.md`, `Ingest Process.md`, `Andrej Karpathy.md`
- Wikilinks reference the bare filename: `[[LLM Wiki]]`, `[[Andrej Karpathy]]`
- Wikilinks in frontmatter stick to YAML syntax: `- '[[LLM Wiki]]'`, `- '[[Andrej Karpathy]]'`
- Do not use directory paths in wikilinks: `[[source/LLM Wiki.md]]`, `[[entities/Andrej Karpathy.md]]`

## Page structure

```markdown
---
type: concept | entity | analysis
created: YYYY-MM-DD
sources:
  - '[[Contributing Source]]'
related:
  - '[[Related Wiki Page]]'
---

[content]

## Timeline

- YYYY-MM-DD: [created] initial definition of [[LLM Wiki]]
- YYYY-MM-DD: [refined] elaborated steps referencing [[Karpathy Gist]]
- YYYY-MM-DD: [corrected] changed [[Reasoning]] to [[Planning]]
```

### Timeline date

Use the first available value in order:

1. `created` metadata field from the source file
2. File system birth time (`btime`) of the source file
3. Current date

### Timeline tags

- [created]: Page first created
- [refined]: Content expanded or deepened
- [corrected]: Factual error fixed
- [linked]: New cross-reference added

## Page types

- **concept**: a recurring idea synthesized across multiple sources. Accumulate perspective and tendencies observed across documents.
- **entity**: a person, organization, or work appearing in source documents. Track attributes and relationships to other entities and concepts.
- **analysis**: a synthesized answer to a specific query. Created when a query produces non-trivial insight worth preserving.

## log.json schema

All timestamps are local time in `YYYY-MM-DD HH:MM` format. `ingest` and `query` accumulate history keyed by source path and query text. `lint` stores only the most recent run and is overwritten each time.

```json
{
  "ingest": {
    "Karpathy Gist.md": {
      "hash": "<sha256>",
      "ingested_at": "2026-04-15 10:00",
      "pages": ["LLM Wiki.md", "Andrej Karpathy.md"],
      "time_taken": "30s"
    }
  },
  "query": {
    "How does ingest work?": {
      "queried_at": "2026-04-15 11:00",
      "pages": ["Ingest Process.md"],
      "time_taken": "15s"
    }
  },
  "lint": {
    "linted_at": "2026-04-15 12:00",
    "time_taken": "60s",
    "issues": [
      {
        "type": "orphan | contradiction | missing_reference",
        "pages": ["LLM Wiki.md"],
        "detail": "<description>"
      }
    ]
  }
}
```

## index.md format

index.md is a human-readable catalog written and maintained by the agent. It should always reflect the current state of the wiki. Group pages by type with a one-line description per entry.

```markdown
## Concepts

- [[Page Name]] — one-line description

## Entities

- [[Page Name]] — one-line description

## Analyses

- [[Page Name]] — one-line description

---

**Last updated**: YYYY-MM-DD
```
