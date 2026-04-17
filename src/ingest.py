import os
import json
import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from litellm import completion

from config import LLM_MODEL, INDEX_FILE, API_BASE, LOG_FILE, SOURCE_DIR, TYPE_DIRS, WIKI_DIR
from embed import embed as embed_texts, get_context_length
from vectordb import find_related, upsert_page

app = typer.Typer(help="Ingest markdown files from a source directory into the wiki.", invoke_without_command=True)


@app.callback()
def _default(ctx: typer.Context):
    """Run ingest when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        ingest(source_dir=str(SOURCE_DIR), model=LLM_MODEL, limit=None, workers=4)

# ── helpers ────────────────────────────────────────────────────────────────────

def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def source_date(filepath: Path) -> str:
    """Return creation date for a source file.

    Priority:
    1. `created` field in the file's own frontmatter
    2. File system birth time (st_birthtime on macOS, st_mtime fallback)
    3. Current date
    """
    try:
        text = filepath.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if line.startswith("created:"):
                    val = line[8:].strip().strip("'\"")
                    if val:
                        return val
    except OSError:
        pass

    try:
        st = filepath.stat()
        ts = getattr(st, "st_birthtime", None) or st.st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except OSError:
        pass

    return today()

# ── log ────────────────────────────────────────────────────────────────────────

def load_log() -> dict:
    if LOG_FILE.exists():
        try:
            data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data.setdefault("ingest", {})
    data.setdefault("query", {})
    data.setdefault("lint", {})
    data.setdefault("errors", [])
    return data

def save_log(log_data: dict):
    os.makedirs(WIKI_DIR, exist_ok=True)
    LOG_FILE.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

# ── frontmatter ────────────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Return (meta_dict, body). meta always has keys: type, created, sources, related."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
    if not m:
        return {"type": "concept", "created": "", "sources": [], "related": []}, content
    fm_text, body = m.group(1), m.group(2).lstrip("\n")

    meta: dict = {"type": "concept", "created": "", "sources": [], "related": []}
    current_list: Optional[str] = None
    for line in fm_text.splitlines():
        if line.startswith("type:"):
            meta["type"] = line[5:].strip()
            current_list = None
        elif line.startswith("created:"):
            meta["created"] = line[8:].strip()
            current_list = None
        elif line.startswith("sources:"):
            current_list = "sources"
        elif line.startswith("related:"):
            current_list = "related"
        elif line.startswith("  - ") and current_list:
            val = line[4:].strip()
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            meta[current_list].append(val)
        else:
            current_list = None
    return meta, body

def render_frontmatter(page_type: str, created: str, sources: list[str], related: list[str]) -> str:
    lines = ["---", f"type: {page_type}", f"created: {created}"]
    if sources:
        lines.append("sources:")
        for s in sources:
            lines.append(f"  - '{s}'")
    if related:
        lines.append("related:")
        for r in related:
            lines.append(f"  - '{r}'")
    lines.append("---")
    return "\n".join(lines)

# ── wiki page I/O ──────────────────────────────────────────────────────────────

def _safe_filename(title: str) -> str:
    """Replace filesystem-unsafe characters so any title can be a valid filename."""
    return re.sub(r'[/\\:*?"<>|]', '-', title).strip('-').strip()

def find_page_path(title: str) -> Optional[Path]:
    safe = _safe_filename(title)
    for d in TYPE_DIRS.values():
        p = d / f"{safe}.md"
        if p.exists():
            return p
    return None

def read_page(title: str) -> Optional[tuple[dict, str, list[str]]]:
    """Return (meta, body, timeline_entries) or None if the page doesn't exist."""
    path = find_page_path(title)
    if path is None:
        return None
    content = path.read_text(encoding="utf-8")
    meta, rest = parse_frontmatter(content)

    if "\n## Timeline\n" in rest:
        body, tl = rest.split("\n## Timeline\n", 1)
        timeline = [l for l in tl.strip().splitlines() if l.startswith("- ")]
    else:
        body, timeline = rest, []
    return meta, body.strip(), timeline

def write_page(
    title: str,
    page_type: str,
    created: str,
    sources: list[str],
    related: list[str],
    body: str,
    timeline: list[str],
):
    target_dir = TYPE_DIRS.get(page_type, TYPE_DIRS["concept"])
    target_dir.mkdir(parents=True, exist_ok=True)

    fm = render_frontmatter(page_type, created, sources, related)
    parts = [fm, "", body.strip()]
    if timeline:
        parts += ["", "## Timeline", ""] + timeline
    path = target_dir / f"{_safe_filename(title)}.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")

def wikilinks_in(text: str) -> list[str]:
    titles = re.findall(r"\[\[([^\[\]]+)\]\]", text)
    return [t.strip().strip("\"'") for t in titles]

# ── index ──────────────────────────────────────────────────────────────────────

def load_index() -> str:
    return INDEX_FILE.read_text(encoding="utf-8") if INDEX_FILE.exists() else ""

def rebuild_index():
    """Rebuild index.md by scanning all wiki pages."""
    sections: dict[str, list[tuple[str, str]]] = {
        "concept": [], "entity": [], "analysis": []
    }

    for page_type, type_dir in TYPE_DIRS.items():
        if not type_dir.exists():
            continue
        for page_file in sorted(type_dir.glob("*.md")):
            title = page_file.stem
            content = page_file.read_text(encoding="utf-8")
            _, body = parse_frontmatter(content)
            if "\n## Timeline\n" in body:
                body = body.split("\n## Timeline\n")[0]
            desc = next(
                (re.sub(r"\[\[([^\]]+)\]\]", r"\1", l).strip()[:120]
                 for l in body.splitlines()
                 if l.strip() and not l.startswith("#")),
                "",
            )
            sections[page_type].append((title, desc))

    labels = {"concept": "Concepts", "entity": "Entities", "analysis": "Analyses"}
    lines: list[str] = []
    for pt in ("concept", "entity", "analysis"):
        entries = sections[pt]
        if not entries:
            continue
        lines += [f"## {labels[pt]}", ""]
        for title, desc in entries:
            lines.append(f"- [[{title}]] — {desc}" if desc else f"- [[{title}]]")
        lines.append("")

    lines += ["---", "", f"**Last updated**: {today()}", ""]
    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")

# ── LLM ───────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a wiki maintenance agent. Given a source document and the current wiki state, you:
1. Identify concepts and entities in the source that are not yet covered — create new pages for them.
2. Identify existing pages that should be updated with information from this source.

Write all page titles, body content, and descriptions in Korean.
Return ONLY a valid JSON object — no markdown fences, no prose."""

# ── context length ─────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate accounting for Korean (hangul ~1.5 chars/token) and Latin (~4 chars/token)."""
    korean = sum(1 for c in text if "\uAC00" <= c <= "\uD7A3")
    other = len(text) - korean
    return int(korean / 1.5 + other / 4)


def _compute_char_budgets(
    source_text: str,
    index_text: str,
    n_related: int,
    context_length: int,
) -> tuple[int, int, int]:
    """Return (source_limit, index_limit, excerpt_per_page_limit) in chars.

    Priority: source text > related excerpts > index listing.
    Reserves output (~2 k tokens) + system prompt + JSON instructions overhead.
    """
    OUTPUT_RESERVE = 2000  # tokens reserved for LLM output
    INSTRUCTIONS_TOKENS = _estimate_tokens(SYSTEM_PROMPT) + 200  # JSON schema overhead

    available_tokens = max(context_length - OUTPUT_RESERVE - INSTRUCTIONS_TOKENS, 500)
    # Conservative: 1.5 chars per token (Korean-heavy text)
    available_chars = int(available_tokens * 1.5)

    # Source gets up to 55 % — honour as much of the source as possible
    source_limit = min(len(source_text), int(available_chars * 0.55))

    # Index gets up to 10 % — just needs to convey existing page titles
    index_limit = min(len(index_text), int(available_chars * 0.10))

    # Related excerpts share the remainder
    remaining = available_chars - source_limit - index_limit
    excerpt_limit = max(remaining // n_related, 100) if n_related > 0 else 0

    return source_limit, index_limit, excerpt_limit


def build_prompt(
    source_path: str,
    source_text: str,
    index_text: str,
    related: dict[str, str],
    context_length: int = 8192,
) -> str:
    source_lim, index_lim, excerpt_lim = _compute_char_budgets(
        source_text, index_text, len(related), context_length
    )

    parts = [
        f"Source: {source_path}",
        "",
        "## Source content",
        source_text[:source_lim],
        "",
        "## Current wiki index",
        (index_text or "(empty — no pages yet)")[:index_lim],
    ]
    if related:
        parts += ["", "## Existing related pages (excerpts)"]
        for title, excerpt in related.items():
            parts += [f"### {title}", excerpt[:excerpt_lim]]
    parts += [
        "",
        "## Instructions",
        "Return JSON with this exact structure:",
        '{',
        '  "new_pages": [',
        '    {',
        '      "type": "concept OR entity",',
        '      "title": "Natural Title Case name",',
        '      "body": "full markdown body — use [[Wikilinks]] for cross-references",',
        '      "description": "one-line description for the index"',
        '    }',
        '  ],',
        '  "updated_pages": [',
        '    {',
        '      "title": "Exact existing page title",',
        '      "body": "updated full markdown body",',
        '      "timeline_tag": "[refined] OR [linked] OR [corrected]",',
        '      "timeline_detail": "what changed and why"',
        '    }',
        '  ]',
        '}',
    ]
    return "\n".join(parts)

def call_llm(
    messages: list[dict],
    model: str,
    buf: Optional[list[str]] = None,
    live_status: Optional[callable] = None,  # type: ignore[valid-type]
) -> str:
    """Call LLM with streaming.

    buf: if given, all output lines are buffered here instead of printed.
    live_status: optional callable(msg) for real-time progress lines (e.g. thinking status).
                 Always prints directly, independent of buf.
    """
    def emit(text: str, end: str = "\n", flush: bool = False):
        if buf is not None:
            if end == "\n":
                buf.append(text)
            else:
                if buf and not buf[-1].endswith("\n"):
                    buf[-1] += text
                else:
                    buf.append(text)
        else:
            print(text, end=end, flush=flush)

    kwargs: dict = {"api_base": API_BASE, "model": model, "messages": messages, "temperature": 0.2, "stream": True}
    response = completion(**kwargs)  # type: ignore[assignment]
    full = ""
    reasoning = ""
    thinking = False
    thinking_start = 0.0

    for chunk in response:
        delta = chunk.choices[0].delta  # type: ignore[union-attr]
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            if not thinking:
                emit("  Thinking", end="", flush=True)
                if live_status:
                    live_status("thinking...")
                thinking = True
                thinking_start = time.monotonic()
            prev_len = len(reasoning)
            reasoning += rc
            if live_status and len(reasoning) // 500 > prev_len // 500:
                elapsed = time.monotonic() - thinking_start
                live_status(f"thinking... {elapsed:.1f}s", True)
            continue
        c = delta.content
        if c:
            if thinking:
                elapsed = time.monotonic() - thinking_start
                emit(f" ({elapsed:.1f}s)\n  Output: ", end="", flush=True)
                if live_status:
                    live_status("output 생성 중...")
                thinking = False
            elif not full:
                emit("  Output: ", end="", flush=True)
                if live_status:
                    live_status("output 생성 중...")
            full += c
            emit(c, end="", flush=True)
    emit("")
    return full or reasoning

def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fix invalid backslash escapes (e.g. \k, \p) produced by some LLMs.
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Structural repair for badly formed LLM output (missing colons, quotes, etc.)
    from json_repair import repair_json
    return json.loads(repair_json(fixed))

# ── parallel worker ────────────────────────────────────────────────────────────

def _llm_phase(
    filepath: Path,
    log: dict,
    index_text: str,
    model: str,
    print_lock: threading.Lock,
    context_length: int = 8192,
) -> Optional[tuple[Path, Optional[str], Optional[dict], list[str], Optional[str]]]:
    """Hash-check and LLM call for one file. Thread-safe (read-only wiki access).

    Returns (filepath, file_hash, llm_data, output_lines, error) or None if skipped.
    error is None on success, error message string on failure.
    """
    buf: list[str] = []
    rel = str(filepath)

    file_hash = get_file_hash(filepath)

    with print_lock:
        print(f"  → {filepath.name} 처리 시작...", flush=True)

    source_text = filepath.read_text(encoding="utf-8")
    _, source_body = parse_frontmatter(source_text)

    # Load all wiki pages for hybrid search
    all_pages: dict[str, str] = {}
    for title in wikilinks_in(index_text):
        result = read_page(title)
        if result:
            _, body, _ = result
            all_pages[title] = body

    related = find_related(all_pages, source_body, n=10, embed_fn=embed_texts)

    buf.append(f"  Calling {model} ...")

    name = filepath.name

    def live_status(msg: str, overwrite: bool = False) -> None:
        with print_lock:
            line = f"  [{name}] {msg:<50}"
            if overwrite:
                print(f"\r{line}", end="", flush=True)
            else:
                print(f"\r{line}", flush=True)

    titles_inline = ", ".join(related.keys()) if related else "none"
    live_status(f"LLM 호출 중... ({len(related)} pages: {titles_inline})")

    prompt = build_prompt(rel, source_text, index_text, related, context_length)
    try:
        raw = call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model,
            buf=buf,
            live_status=live_status,
        )
        data = extract_json(raw)
    except Exception as e:
        error_msg = str(e)
        live_status(f"오류: {error_msg}")
        buf.append(f"  Error: {error_msg}")
        return filepath, None, None, buf, error_msg

    return filepath, file_hash, data, buf, None


# ── main command ───────────────────────────────────────────────────────────────

@app.command()
def ingest(
    source_dir: Optional[str] = typer.Argument(
        str(SOURCE_DIR), help="Source directory path."
    ),
    model: str = typer.Option(LLM_MODEL, help="LiteLLM model string."),
    limit: Optional[int] = typer.Option(None, help="Cap number of files to process."),
    workers: int = typer.Option(4, help="Number of parallel LLM workers."),
):
    """Ingest markdown source files into the wiki."""
    if not source_dir:
        print("Error: provide source_dir argument or set SOURCE_DIR env var.")
        raise typer.Exit(1)

    src = Path(source_dir)
    if not src.is_dir():
        print(f"Not a directory: {source_dir}")
        raise typer.Exit(1)

    for d in TYPE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    log = load_log()
    log["errors"] = []  # reset each run — errors reflect only the most recent run
    all_files = sorted(src.rglob("*.md"))
    if limit:
        all_files = all_files[:limit]

    # Query model context length once before spawning workers.
    ctx_len = get_context_length(model, API_BASE, fallback=8192)

    # ── Pre-filter: hash check (fast, synchronous) ─────────────────────────────
    files_to_process: list[Path] = []
    skipped: list[str] = []
    for fp in all_files:
        file_hash = get_file_hash(fp)
        if log["ingest"].get(str(fp), {}).get("hash") == file_hash:
            skipped.append(fp.name)
        else:
            files_to_process.append(fp)

    n_total = len(all_files)
    n_skip  = len(skipped)
    n_proc  = len(files_to_process)
    print(f"Found {n_total} file(s): {n_proc} to process, {n_skip} unchanged. "
          f"Workers: {workers}. Context: {ctx_len:,} tokens.")

    if not files_to_process:
        print("Nothing to do.")
        return

    # Snapshot index once — all workers share the same starting state.
    index_text = load_index()

    # ── Phase 1: LLM calls (parallel) ─────────────────────────────────────────
    results: list[tuple[Path, str, dict, list[str], None]] = []
    print_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_llm_phase, fp, log, index_text, model, print_lock, ctx_len): fp
            for fp in files_to_process
        }
        for future in as_completed(futures):
            fp = futures[future]
            completed += 1
            print(f"\n[{completed}/{n_proc}] ── {fp.name} ──")
            try:
                result = future.result()
            except Exception as e:
                error_msg = str(e)
                print(f"  Unexpected error: {error_msg}")
                log["errors"].append({
                    "file": str(fp),
                    "error": error_msg,
                    "phase": "llm",
                    "timestamp": now(),
                })
                continue
            if result is None:
                continue
            _, file_hash, data, buf, error = result
            for line in buf:
                print(line)
            if error is not None:
                log["errors"].append({
                    "file": str(fp),
                    "error": error,
                    "phase": "llm",
                    "timestamp": now(),
                })
                continue
            results.append(result)

    # ── Phase 2: write pages (sequential) ─────────────────────────────────────
    processed = 0
    for filepath, file_hash, data, _, _err in results:
        if not data:  # unchanged file
            continue

        rel = str(filepath)
        source_link = f"[[{filepath.stem}]]"
        src_date = source_date(filepath)
        page_names_created: list[str] = []
        touched: list[str] = []

        # Create new pages
        for page in data.get("new_pages", []):
            ptype = page.get("type", "concept")
            if ptype not in TYPE_DIRS:
                ptype = "concept"
            title = (page.get("title") or "").strip()
            body = (page.get("body") or "").strip()
            if not title or not body:
                continue

            crosslinks = [f"[[{t}]]" for t in wikilinks_in(body) if t != title]
            write_page(
                title=title,
                page_type=ptype,
                created=src_date,
                sources=[source_link],
                related=list(dict.fromkeys(crosslinks)),
                body=body,
                timeline=[f"- {src_date[:10]}: [created] initial page from {source_link}"],
            )
            upsert_page(title, body, embed_texts)
            print(f"  [new] {ptype}/{title}.md")
            page_names_created.append(title)

        # Update existing pages
        for page in data.get("updated_pages", []):
            title = (page.get("title") or "").strip()
            new_body = (page.get("body") or "").strip()
            tag = page.get("timeline_tag", "[refined]")
            detail = page.get("timeline_detail", "updated")
            if not title or not new_body:
                continue

            existing = read_page(title)
            if not existing:
                print(f"  [skip] unknown page in updated_pages: {title!r}")
                continue

            meta, _, timeline = existing
            sources = meta.get("sources", [])
            if source_link not in sources:
                sources.append(source_link)

            new_crosslinks = [f"[[{t}]]" for t in wikilinks_in(new_body) if t != title]
            merged_related = list(dict.fromkeys(meta.get("related", []) + new_crosslinks))
            timeline.append(f"- {today()}: {tag} {detail}")

            write_page(
                title=title,
                page_type=meta.get("type", "concept"),
                created=meta.get("created") or today(),
                sources=sources,
                related=merged_related,
                body=new_body,
                timeline=timeline,
            )
            upsert_page(title, new_body, embed_texts)
            print(f"  [updated] {title}.md")
            touched.append(title)

        log["ingest"][rel] = {
            "hash": file_hash,
            "ingested_at": now(),
            "pages": page_names_created + touched,
        }
        processed += 1

    if processed:
        rebuild_index()

    if processed or log["errors"]:
        save_log(log)

    if log["errors"]:
        print(f"  {len(log['errors'])} error(s) recorded in log.json.")
    print(f"\nDone. {processed} file(s) processed.")


@app.command()
def vectorize(
    workers: int = typer.Option(4, help="Parallel embedding workers."),
):
    """(Re-)embed all existing wiki pages into the local vector DB."""
    pages: list[tuple[str, str]] = []
    for type_dir in TYPE_DIRS.values():
        if not type_dir.exists():
            continue
        for page_file in sorted(type_dir.glob("*.md")):
            result = read_page(page_file.stem)
            if result:
                _, body, _ = result
                pages.append((page_file.stem, body))

    if not pages:
        print("No wiki pages found.")
        raise typer.Exit(0)

    print(f"Vectorizing {len(pages)} page(s) with {workers} workers...")
    total = len(pages)
    done = 0
    lock = threading.Lock()

    def _embed_one(args: tuple[str, str]) -> tuple[str, int]:
        title, body = args
        count = upsert_page(title, body, embed_texts)
        return title, count

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_embed_one, p): p[0] for p in pages}
        for future in as_completed(futures):
            title = futures[future]
            try:
                _, count = future.result()
            except Exception as e:
                print(f"  [error] {title}: {e}")
                continue
            with lock:
                done += 1
                print(f"  [{done}/{total}] {title} ({count} chunk(s))")

    print(f"\nDone. {done} page(s) vectorized.")


if __name__ == "__main__":
    app()
