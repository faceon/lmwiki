---
type: journal
date: 2026-02-28
tags: [AI, memory, context-window, architecture]
---

Had a long session debugging a multi-file feature with an LLM today, and hit the familiar wall: the model started contradicting its own earlier suggestions once the conversation grew long enough. The context window had not technically overflowed, but attention had clearly degraded.

This keeps bringing me back to the memory problem. A large context is not the same as good memory. Cramming everything into a single prompt is brute force, not architecture. The more interesting question is how to structure persistent, retrievable knowledge outside the context window—summaries, entity graphs, indexed notes—so the model can reason over a project without holding all of it in working memory at once. That is really what the LLM-Wiki idea is reaching for.
