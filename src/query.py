import hashlib
import time
from pathlib import Path
from typing import Optional

import typer

from config import (
    LOCAL_LLM_MODEL, LOCAL_LLM_API_BASE, REMOTE_LLM_MODEL,
    EMBED_MODEL, EMBED_API_BASE, TYPE_DIRS,
)
from embed import embed as embed_texts, get_context_length
from vectordb import find_related, upsert_page
from ingest import (
    read_page, write_page, rebuild_index, load_log, save_log,
    call_llm, extract_json, _normalize_title, _autocorrect_wikilinks,
    wikilinks_in, today, now, load_index, _estimate_tokens,
)
import eventlog

app = typer.Typer(help="Query the wiki and synthesize answers.", invoke_without_command=True)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_FILE = _PROMPTS_DIR / "query.system.md"
_USER_PROMPT_FILE = _PROMPTS_DIR / "query.user.md"
SYSTEM_PROMPT = _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
_USER_TEMPLATE = _USER_PROMPT_FILE.read_text(encoding="utf-8")

def _prompt_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

_PROMPT_HASHES = {
    "system": _prompt_hash(_SYSTEM_PROMPT_FILE),
    "user": _prompt_hash(_USER_PROMPT_FILE),
}


def _load_all_pages() -> dict[str, str]:
    pages: dict[str, str] = {}
    for d in TYPE_DIRS.values():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            result = read_page(p.stem)
            if result:
                _, body, _ = result
                pages[p.stem] = body
    return pages


def _build_query_prompt(
    question: str,
    index_text: str,
    relevant: dict[str, tuple[str, float]],
    context_length: int = 8192,
) -> str:
    OUTPUT_RESERVE = 3000
    INSTRUCTIONS_TOKENS = _estimate_tokens(SYSTEM_PROMPT) + 300
    available_tokens = max(context_length - OUTPUT_RESERVE - INSTRUCTIONS_TOKENS, 500)
    available_chars = int(available_tokens * 1.5)

    index_limit = min(len(index_text), int(available_chars * 0.10))
    pages_budget = available_chars - index_limit - len(question)
    per_page = max(pages_budget // len(relevant), 200) if relevant else 0

    lines: list[str] = []
    for title, (excerpt, score) in relevant.items():
        result = read_page(title)
        if result:
            _, body, _ = result
            text = body[:per_page]
        else:
            text = excerpt[:per_page]
        lines += [f"### [[{title}]] [유사도: {score:.0%}]", text, ""]

    pages_section = "\n".join(lines) if lines else "(관련 페이지 없음)"

    return (
        _USER_TEMPLATE
        .replace("{question}", question)
        .replace("{index_content}", (index_text or "(empty)")[:index_limit])
        .replace("{pages_section}", pages_section)
    )


@app.callback(invoke_without_command=True)
def query(
    question: str = typer.Argument(..., help="Question to ask the wiki."),
    remote: bool = typer.Option(False, "--remote", help="Use remote (cloud) LLM instead of localhost."),
    model: str = typer.Option("", help="LiteLLM model string override."),
    n: int = typer.Option(8, "--n", help="Number of relevant pages to retrieve."),
):
    """Query the wiki and synthesize an answer with citations."""
    if remote:
        model = model or REMOTE_LLM_MODEL
        api_base = None
    else:
        model = model or LOCAL_LLM_MODEL
        api_base = LOCAL_LLM_API_BASE

    ctx_len = get_context_length(model, api_base, fallback=8192)
    print(f"LLM:   {model} @ {api_base or '(remote)'} — {ctx_len:,} tokens")
    print(f"Embed: {EMBED_MODEL} @ {EMBED_API_BASE}")
    print(f"\nQuestion: {question}\n")

    all_pages = _load_all_pages()
    if not all_pages:
        print("Wiki is empty. Run ingest first.")
        raise typer.Exit(1)

    relevant = find_related(all_pages, question, n=n, embed_fn=embed_texts)
    if relevant:
        ctx_summary = ", ".join(f"{t}({s:.0%})" for t, (_, s) in relevant.items())
        print(f"Retrieved {len(relevant)} page(s): {ctx_summary}\n")
    else:
        print("No relevant pages found.\n")

    index_text = load_index()
    prompt = _build_query_prompt(question, index_text, relevant, ctx_len)

    def live_status(msg: str, overwrite: bool = False) -> None:
        line = f"  {msg:<60}"
        if overwrite:
            print(f"\r{line}", end="", flush=True)
        else:
            print(f"\r{line}", flush=True)

    eventlog.start_run(
        workflow="query",
        model=model,
        api_base=api_base,
        question=question,
        n_retrieved=len(relevant),
    )
    run_start_t = time.monotonic()

    try:
        result = call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model,
            api_base=api_base,
            live_status=live_status,
        )
        if not result["text"].strip():
            raise ValueError("LLM returned empty response")
        print()
        data, json_repair_used = extract_json(result["text"])
    except Exception as e:
        error_msg = str(e)
        print(f"\nError: {error_msg}")
        eventlog.emit("error", workflow="query", phase="llm", error=error_msg)
        eventlog.end_run(duration_ms=0, n_processed=0, n_errors=1)
        raise typer.Exit(1)

    eventlog.emit(
        "llm_call",
        workflow="query",
        model=model,
        api_base=api_base,
        prompt_hashes=_PROMPT_HASHES,
        prompt_chars=len(prompt),
        context_length=ctx_len,
        n_retrieved=len(relevant),
        latency_ms=result["total_ms"],
        thinking_ms=result["thinking_ms"],
        output_ms=result["output_ms"],
        json_repair_used=json_repair_used,
    )

    answer = data.get("answer", "").strip()
    citations = data.get("citations", [])
    save_analysis = bool(data.get("save_analysis", False))
    analysis_title_raw = (data.get("analysis_title") or "").strip()
    analysis_body = (data.get("analysis_body") or "").strip()

    # Print the answer.
    print("─" * 60)
    print(answer)
    if citations:
        print(f"\n출처: {', '.join(f'[[{c}]]' for c in citations)}")
    print("─" * 60)

    log = load_log()
    query_key = question[:80]

    if save_analysis and analysis_title_raw and analysis_body:
        analysis_title = _normalize_title(analysis_title_raw, "analysis")
        all_titles: set[str] = {p.stem for d in TYPE_DIRS.values() if d.exists() for p in d.glob("*.md")}
        clean_body = _autocorrect_wikilinks(analysis_body, all_titles)
        crosslinks = [f"[[{t}]]" for t in wikilinks_in(clean_body) if t != analysis_title]

        write_page(
            title=analysis_title,
            page_type="analysis",
            created=today(),
            sources=[],
            related=list(dict.fromkeys(crosslinks)),
            body=clean_body,
            timeline=[f"- {today()}: [created] query: {question[:60]}"],
        )
        upsert_page(analysis_title, clean_body, embed_texts)
        print(f"\n[saved] analyses/{analysis_title}.md")
        eventlog.emit(
            "decision",
            workflow="query",
            action="new",
            title=analysis_title,
            page_type="analysis",
        )
        rebuild_index()

        log["query"][query_key] = {
            "queried_at": now(),
            "citations": citations,
            "analysis_page": analysis_title,
        }
    else:
        log["query"][query_key] = {
            "queried_at": now(),
            "citations": citations,
        }

    save_log(log)
    eventlog.end_run(
        duration_ms=int((time.monotonic() - run_start_t) * 1000),
        n_processed=1,
        n_errors=0,
        saved_analysis=save_analysis,
    )


if __name__ == "__main__":
    app()
