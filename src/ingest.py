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

from config import LLM_MODEL, LLM_API_BASE, EMBED_MODEL, EMBED_API_BASE, INDEX_FILE, LOG_FILE, SOURCE_DIR, TYPE_DIRS, WIKI_DIR
from embed import embed as embed_texts, get_context_length
from vectordb import find_related, upsert_page
import eventlog

app = typer.Typer(help="Ingest markdown files from a source directory into the wiki.", invoke_without_command=True)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_FILE = _PROMPTS_DIR / "ingest.system.md"
_USER_PROMPT_FILE = _PROMPTS_DIR / "ingest.user.md"
SYSTEM_PROMPT = _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
_USER_TEMPLATE = _USER_PROMPT_FILE.read_text(encoding="utf-8")

def _prompt_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

_PROMPT_HASHES = {
    "system": _prompt_hash(_SYSTEM_PROMPT_FILE),
    "user": _prompt_hash(_USER_PROMPT_FILE),
}


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    reset: bool = typer.Option(False, "--reset/--no-reset", help="Delete all wiki pages and vector DB before ingesting."),
    remote: bool = typer.Option(False, "--remote", help="Use remote (cloud) LLM instead of localhost."),
    workers: int = typer.Option(4, help="Number of parallel LLM workers."),
    limit: Optional[int] = typer.Option(None, help="Cap number of files to process."),
    model: str = typer.Option(LLM_MODEL, help="LiteLLM model string."),
):
    """Ingest markdown source files into the wiki."""
    if ctx.invoked_subcommand is None:
        ingest(source_dir=str(SOURCE_DIR), model=model, remote=remote, limit=limit, workers=workers, reset=reset)

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
    1. `created` or `date` field in the file's own frontmatter
       (journals typically use `date`; the schema uses `created`)
    2. File system birth time (st_birthtime on macOS, st_mtime fallback)
    3. Current date
    """
    try:
        text = filepath.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                fm_match = re.match(r"^(created|date):\s*(.+)$", line)
                if fm_match:
                    val = fm_match.group(2).strip().strip("'\"")
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


def source_title(filepath: Path) -> str:
    """Human-readable wiki title derived from a source filename.

    Source files often use snake_case or kebab-case; wiki wikilink convention
    (SCHEMA.md) uses natural Title Case with spaces, so separators are replaced.
    The source file itself is not renamed.
    """
    return filepath.stem

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
    data.setdefault("orphans", [])
    return data

def save_log(log_data: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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

def _strip_prefix(title: str) -> str:
    return re.sub(r'^[~@]+', '', title).replace("_", " ").strip()

def _normalize_title(title: str, page_type: str = "concept") -> str:
    """Normalize a page title: strip any prefix/underscores, add @ for entities only."""
    bare = _strip_prefix(title)
    return f"@{bare}" if page_type == "entity" else bare

def _autocorrect_wikilinks(body: str, known_titles: set[str]) -> str:
    """Fix wikilinks in body: resolve correct prefix, strip dangling refs."""
    def fix(m: re.Match) -> str:
        inner = m.group(1).strip()
        bare = _strip_prefix(inner)
        for candidate in dict.fromkeys([inner, f"@{bare}", bare]):
            if candidate in known_titles or find_page_path(candidate):
                return f"[[{candidate}]]"
        # Preserve source file references as-is (original filename with underscores)
        if (SOURCE_DIR / f"{inner}.md").exists():
            return f"[[{inner}]]"
        return inner  # dangling — strip brackets, preserve original text
    return re.sub(r"\[\[([^\[\]]+)\]\]", fix, body)

def _strip_duplicate_h1(body: str, title: str) -> str:
    """Strip a leading `# Title` H1 that duplicates the page title.

    The filename already serves as the page title in wikilink-based viewers,
    so a matching H1 at the top of the body is redundant. H2+ is preserved —
    the LLM may legitimately use `## <title>` as a section under the body.
    """
    stripped = body.lstrip("\n")
    lines = stripped.splitlines()
    if not lines:
        return body
    m = re.match(r"^#\s+(.+?)\s*$", lines[0].strip())
    if m and m.group(1).strip() == title.strip():
        return "\n".join(lines[1:]).lstrip("\n")
    return body

def find_page_path(title: str) -> Optional[Path]:
    bare = _strip_prefix(title)
    for candidate in dict.fromkeys([title, f"@{bare}", bare]):
        safe = _safe_filename(candidate)
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
    clean_body = _strip_duplicate_h1(body, title).strip()
    parts = [fm, "", clean_body]
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


def link_source_siblings(titles: list[str]) -> int:
    """Cross-link pages derived from the same source.

    After a single source has produced a batch of pages, every page in that
    batch should list the others in its `related` frontmatter so the wiki's
    relationship graph stays connected. The page body is left untouched
    (LLM-authored content is preserved) and timelines are NOT amended —
    sibling linking is a structural step, not an editorial one.

    Returns the number of pages whose `related` field was actually modified.
    """
    unique_titles = [t for t in dict.fromkeys(titles) if t]
    if len(unique_titles) < 2:
        return 0

    modified = 0
    for title in unique_titles:
        result = read_page(title)
        if not result:
            continue
        meta, body, timeline = result
        existing = list(meta.get("related", []))
        sibling_links = [f"[[{s}]]" for s in unique_titles if s != title]
        merged = list(dict.fromkeys(existing + sibling_links))
        if merged == existing:
            continue
        write_page(
            title=title,
            page_type=meta.get("type", "concept"),
            created=meta.get("created") or today(),
            sources=meta.get("sources", []),
            related=merged,
            body=body,
            timeline=timeline,
        )
        modified += 1
    return modified


def collect_orphan_wikilinks() -> list[dict]:
    """Scan all wiki page bodies for wikilinks to non-existent wiki pages.

    Returns a list of {"page": <wiki page stem>, "missing": <missing title>} entries.
    Two kinds of references are intentionally skipped because they point to files
    outside the wiki tree:
      - Frontmatter `sources` (handled via page-level parse, not scanned here).
      - Timeline entries (e.g. "initial page from [[AI Productivity]]").
    """
    existing: set[str] = set()
    for d in TYPE_DIRS.values():
        if d.exists():
            for p in d.glob("*.md"):
                existing.add(p.stem)

    orphans: list[dict] = []
    for d in TYPE_DIRS.values():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            result = read_page(p.stem)
            if not result:
                continue
            _, body, _timeline = result
            seen: set[str] = set()
            for link in wikilinks_in(body):
                link = link.strip()
                if not link or link == p.stem or link in existing or link in seen:
                    continue
                orphans.append({"page": p.stem, "missing": link})
                seen.add(link)
    return orphans

# ── LLM ───────────────────────────────────────────────────────────────────────


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
    related: dict[str, tuple[str, float]],
    context_length: int = 8192,
) -> str:
    source_lim, index_lim, excerpt_lim = _compute_char_budgets(
        source_text, index_text, len(related), context_length
    )

    related_section = ""
    if related:
        lines = ["", "## Existing related pages (excerpts)"]
        for title, (excerpt, score) in related.items():
            merge_hint = " ← MERGE RULE: similarity ≥ 0.60, must use updated_pages" if score >= 0.60 else ""
            lines += [f"### {title} [유사도: {score:.0%}]{merge_hint}", excerpt[:excerpt_lim]]
        related_section = "\n".join(lines) + "\n"

    return (
        _USER_TEMPLATE
        .replace("{source_path}", source_path)
        .replace("{source_content}", source_text[:source_lim])
        .replace("{index_content}", (index_text or "(empty — no pages yet)")[:index_lim])
        .replace("{related_section}", related_section)
    )

def call_llm(
    messages: list[dict],
    model: str,
    api_base: Optional[str] = None,
    buf: Optional[list[str]] = None,
    live_status: Optional[callable] = None,  # type: ignore[valid-type]
) -> dict:
    """Call LLM with streaming.

    Returns {"text", "total_ms", "thinking_ms", "output_ms"}.
      total_ms    : wall-clock from request start to final chunk.
      thinking_ms : time spent in reasoning_content chunks (sum, in case the
                    provider interleaves). 0 if the model does not expose
                    reasoning tokens.
      output_ms   : from the first visible content chunk to the final chunk.

    buf: if given, all output lines are buffered here instead of printed.
    live_status: optional callable(msg) for real-time progress lines.
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

    kwargs: dict = {"model": model, "messages": messages, "temperature": 0.2, "stream": True}
    if api_base:
        kwargs["api_base"] = api_base
    total_start = time.monotonic()
    response = completion(**kwargs)  # type: ignore[assignment]
    full = ""
    reasoning = ""
    thinking = False
    thinking_start = 0.0
    thinking_total = 0.0
    output_start: Optional[float] = None

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
                thinking_total += elapsed
                emit(f" ({elapsed:.1f}s)\n  Output: ", end="", flush=True)
                if live_status:
                    live_status("output 생성 중...")
                thinking = False
                if output_start is None:
                    output_start = time.monotonic()
            elif not full:
                emit("  Output: ", end="", flush=True)
                if live_status:
                    live_status("output 생성 중...")
                if output_start is None:
                    output_start = time.monotonic()
            full += c
            emit(c, end="", flush=True)
    emit("")
    total_end = time.monotonic()
    return {
        "text": full or reasoning,
        "total_ms": int((total_end - total_start) * 1000),
        "thinking_ms": int(thinking_total * 1000),
        "output_ms": int((total_end - output_start) * 1000) if output_start is not None else 0,
    }

def extract_json(text: str) -> tuple[dict, bool]:
    """Return (parsed, repair_used). repair_used is True if json_repair was needed."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass
    # Fix invalid backslash escapes (e.g. \k, \p) produced by some LLMs.
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    try:
        return json.loads(fixed), False
    except json.JSONDecodeError:
        pass
    # Structural repair for badly formed LLM output (missing colons, quotes, etc.)
    from json_repair import repair_json
    return json.loads(repair_json(fixed)), True

# ── parallel worker ────────────────────────────────────────────────────────────

def _llm_phase(
    filepath: Path,
    log: dict,
    index_text: str,
    model: str,
    api_base: Optional[str],
    print_lock: threading.Lock,
    context_length: int = 8192,
) -> Optional[tuple[Path, Optional[str], Optional[dict], list[str], Optional[str], dict]]:
    """Hash-check and LLM call for one file. Thread-safe (read-only wiki access).

    Returns (filepath, file_hash, llm_data, output_lines, error, related) or None if skipped.
    error is None on success, error message string on failure.
    related is the hybrid-search result — forwarded to Phase 2 for merge-rule checks.
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

    titles_inline = ", ".join(f"{t}({s:.0%})" for t, (_, s) in related.items()) if related else "none"
    live_status(f"LLM 호출 중... ({len(related)} pages: {titles_inline})")

    prompt = build_prompt(rel, source_text, index_text, related, context_length)
    try:
        result = call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model,
            api_base=api_base,
            buf=buf,
            live_status=live_status,
        )
        data, json_repair_used = extract_json(result["text"])
    except Exception as e:
        error_msg = str(e)
        live_status(f"오류: {error_msg}")
        buf.append(f"  Error: {error_msg}")
        eventlog.emit(
            "error",
            file=rel,
            phase="llm",
            error=error_msg,
            exception=type(e).__name__,
            model=model,
            api_base=api_base,
            prompt_chars=len(prompt),
        )
        return filepath, None, None, buf, error_msg, related

    eventlog.emit(
        "llm_call",
        file=rel,
        model=model,
        api_base=api_base,
        prompt_files={"system": str(_SYSTEM_PROMPT_FILE.name), "user": str(_USER_PROMPT_FILE.name)},
        prompt_hashes=_PROMPT_HASHES,
        prompt_chars=len(prompt),
        context_length=context_length,
        n_related=len(related),
        related=[{"title": t, "score": round(s, 3)} for t, (_, s) in related.items()],
        latency_ms=result["total_ms"],
        thinking_ms=result["thinking_ms"],
        output_ms=result["output_ms"],
        output_chars=len(result["text"]),
        json_repair_used=json_repair_used,
    )

    return filepath, file_hash, data, buf, None, related


# ── main command ───────────────────────────────────────────────────────────────

@app.command()
def ingest(
    source_dir: Optional[str] = typer.Argument(
        str(SOURCE_DIR), help="Source directory path."
    ),
    model: str = typer.Option(LLM_MODEL, help="LiteLLM model string."),
    remote: bool = typer.Option(False, "--remote", help="Use remote (cloud) LLM instead of localhost."),
    limit: Optional[int] = typer.Option(None, help="Cap number of files to process."),
    workers: int = typer.Option(4, help="Number of parallel LLM workers."),
    reset: bool = typer.Option(False, "--reset/--no-reset", help="Delete all wiki pages and vector DB before ingesting."),
):
    """Ingest markdown source files into the wiki."""
    api_base = None if remote else LLM_API_BASE

    if not source_dir:
        print("Error: provide source_dir argument or set SOURCE_DIR env var.")
        raise typer.Exit(1)

    src = Path(source_dir)
    if not src.is_dir():
        print(f"Not a directory: {source_dir}")
        raise typer.Exit(1)

    if reset:
        import shutil
        print("Resetting wiki...")
        for d in TYPE_DIRS.values():
            if d.exists():
                shutil.rmtree(d)
                print(f"  deleted {d}")
        vectordb_dir = WIKI_DIR / ".vectordb"
        if vectordb_dir.exists():
            shutil.rmtree(vectordb_dir)
            print(f"  deleted {vectordb_dir}")
        if eventlog.EVENTS_DIR.exists():
            shutil.rmtree(eventlog.EVENTS_DIR)
            print(f"  deleted {eventlog.EVENTS_DIR}")
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
            print(f"  deleted {INDEX_FILE}")
        if LOG_FILE.exists():
            LOG_FILE.unlink()
            print(f"  deleted {LOG_FILE}")

    for d in TYPE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    log = load_log()
    log["errors"] = []  # reset each run — errors reflect only the most recent run
    all_files = sorted(src.rglob("*.md"))
    if limit:
        all_files = all_files[:limit]

    # Query model context length once before spawning workers.
    ctx_len = get_context_length(model, api_base, fallback=8192)

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

    eventlog.start_run(
        model=model,
        api_base=api_base,
        embed_model=EMBED_MODEL,
        embed_api_base=EMBED_API_BASE,
        context_length=ctx_len,
        workers=workers,
        n_files_total=n_total,
        n_files_skipped=n_skip,
        n_files_to_process=n_proc,
        reset=reset,
    )
    run_start_t = time.monotonic()

    if not files_to_process:
        eventlog.end_run(duration_ms=0, n_processed=0, n_errors=0)
        print("Nothing to do.")
        return

    # Snapshot index once — all workers share the same starting state.
    index_text = load_index()

    # ── Phase 1: LLM calls (parallel) ─────────────────────────────────────────
    results: list[tuple[Path, str, dict, list[str], None, dict]] = []
    print_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_llm_phase, fp, log, index_text, model, api_base, print_lock, ctx_len): fp
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
                eventlog.emit(
                    "error",
                    file=str(fp),
                    phase="worker",
                    error=error_msg,
                    exception=type(e).__name__,
                )
                continue
            if result is None:
                continue
            _, file_hash, data, buf, error, _related = result
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
    n_new_total = 0
    n_updated_total = 0
    n_merge_violations_total = 0
    for filepath, file_hash, data, _, _err, related in results:
        if not data:  # unchanged file
            continue

        rel = str(filepath)
        source_link = f"[[{source_title(filepath)}]]"
        src_date = source_date(filepath)
        page_names_created: list[str] = []
        touched: list[str] = []

        # Pre-collect normalized new page titles for wikilink correction
        new_titles_set: set[str] = {
            _normalize_title(
                (p.get("title") or "").strip(),
                p.get("type", "concept") if p.get("type") in TYPE_DIRS else "concept",
            )
            for p in data.get("new_pages", [])
            if (p.get("title") or "").strip()
        }

        # Create new pages
        for page in data.get("new_pages", []):
            ptype = page.get("type", "concept")
            if ptype not in TYPE_DIRS:
                ptype = "concept"
            title = _normalize_title((page.get("title") or "").strip(), ptype)
            body = _autocorrect_wikilinks((page.get("body") or "").strip(), new_titles_set)
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
            eventlog.emit(
                "decision",
                file=rel,
                action="new",
                title=title,
                page_type=ptype,
            )

        # Update existing pages
        for page in data.get("updated_pages", []):
            title_raw = (page.get("title") or "").strip()
            new_body = _autocorrect_wikilinks((page.get("body") or "").strip(), new_titles_set)
            tag = page.get("timeline_tag", "[refined]")
            detail = page.get("timeline_detail", "updated")
            if not title_raw or not new_body:
                continue

            # Resolve actual title from filesystem (handles prefix variants)
            page_path = find_page_path(title_raw)
            if not page_path:
                title = title_raw
            else:
                title = page_path.stem

            existing = read_page(title)
            if not existing:
                print(f"  [skip] unknown page in updated_pages: {title!r}")
                eventlog.emit(
                    "decision",
                    file=rel,
                    action="skipped",
                    title=title,
                    reason="unknown_page",
                    tag=tag,
                )
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
            eventlog.emit(
                "decision",
                file=rel,
                action="updated",
                title=title,
                page_type=meta.get("type", "concept"),
                tag=tag,
            )

        # MERGE RULE check: any related page with sim ≥ 0.60 that did NOT land
        # in updated_pages is a violation — the LLM ignored the hint.
        updated_titles = {
            (p.get("title") or "").strip() for p in data.get("updated_pages", [])
        }
        for rel_title, (_, score) in related.items():
            if score >= 0.60 and rel_title not in updated_titles:
                n_merge_violations_total += 1
                eventlog.emit(
                    "merge_violation",
                    file=rel,
                    ignored_related_title=rel_title,
                    similarity=round(score, 3),
                    new_pages_created=[
                        (p.get("title") or "").strip()
                        for p in data.get("new_pages", [])
                    ],
                )

        all_siblings = page_names_created + touched
        linked = link_source_siblings(all_siblings)
        if linked:
            print(f"  [linked] {linked} sibling page(s) cross-referenced")

        log["ingest"][rel] = {
            "hash": file_hash,
            "ingested_at": now(),
            "pages": all_siblings,
        }
        processed += 1
        n_new_total += len(page_names_created)
        n_updated_total += len(touched)

    if processed:
        rebuild_index()
        log["orphans"] = collect_orphan_wikilinks()
        if log["orphans"]:
            sample = ", ".join(
                f"[[{o['missing']}]] ← {o['page']}" for o in log["orphans"][:3]
            )
            more = "" if len(log["orphans"]) <= 3 else f", +{len(log['orphans']) - 3} more"
            print(f"  {len(log['orphans'])} orphan wikilink(s): {sample}{more}")

    if processed or log["errors"]:
        save_log(log)

    if log["errors"]:
        print(f"  {len(log['errors'])} error(s) recorded in log.json.")

    eventlog.end_run(
        duration_ms=int((time.monotonic() - run_start_t) * 1000),
        n_processed=processed,
        n_new_pages=n_new_total,
        n_updated_pages=n_updated_total,
        n_errors=len(log["errors"]),
        n_orphans=len(log.get("orphans", [])),
        n_merge_violations=n_merge_violations_total,
    )
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
