---
type: journal
date: 2026-04-18
tags: [LLM, Mac Studio, Gemma, local AI]
---

Ran Gemma 4 26B locally on the Mac Studio today and it genuinely caught me off guard. I expected it to be usable — I did not expect it to feel fast. Token generation keeps up well enough that I kept forgetting it wasn't hitting a cloud endpoint.

The quality held up too, which is the part that's harder to dismiss. Short questions, long context windows, multi-step reasoning — none of it fell apart the way I'd mentally budgeted for. Sensitive data staying entirely on-device went from a theoretical benefit to something I actually care about now that the model is capable enough to do real work.

I understood on paper that the unified memory architecture was why Apple Silicon handles large models differently. Knowing it and feeling it are different things. Two years ago running something like this locally was a research-lab problem. Now it's just sitting in my note ingestion pipeline, processing files while I'm doing something else.
