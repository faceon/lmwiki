import re
import time

import typer

from config import TYPE_DIRS
from ingest import (
    read_page, write_page, wikilinks_in,
    load_log, save_log, rebuild_index,
    today, now,
    collect_orphan_wikilinks, find_page_path,
)

app = typer.Typer(help="Lint wiki pages for consistency issues.", invoke_without_command=True)


def _load_all_pages() -> dict[str, tuple[dict, str, list[str]]]:
    pages = {}
    for d in TYPE_DIRS.values():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            result = read_page(p.stem)
            if result:
                pages[p.stem] = result
    return pages


def _strip_link(link: str) -> str:
    return re.sub(r"^\[\[|\]\]$", "", link).strip()


def find_orphan_pages(pages: dict) -> list[dict]:
    """Pages not referenced in any other page's `related` field."""
    referenced: set[str] = set()
    for meta, _body, _tl in pages.values():
        for link in meta.get("related", []):
            referenced.add(_strip_link(link))

    return [
        {
            "type": "orphan",
            "pages": [f"{stem}.md"],
            "detail": f"'{stem}' is not referenced in any other page's related field.",
        }
        for stem in pages
        if stem not in referenced
    ]


def find_missing_references(pages: dict) -> list[dict]:
    """Wikilinks in body that exist as wiki pages but are absent from `related`."""
    existing = set(pages.keys())
    issues = []
    for stem, (meta, body, _tl) in pages.items():
        related_titles = {_strip_link(l) for l in meta.get("related", [])}
        body_links = {t for t in wikilinks_in(body) if t != stem and t in existing}
        for title in sorted(body_links - related_titles):
            issues.append({
                "type": "missing_reference",
                "pages": [f"{stem}.md"],
                "detail": f"Body mentions [[{title}]] but it is absent from `related`.",
                "_stem": stem,
                "_add": title,
            })
    return issues


def fix_broken_wikilinks(orphans: list[dict], pages: dict) -> int:
    """Replace broken [[Title]] with the correct [[prefix+Title]] if the page exists under a different name."""
    fixes: dict[str, list[tuple[str, str]]] = {}
    for o in orphans:
        stem = o["page"]
        missing = o["missing"]
        path = find_page_path(missing)
        if path:
            fixes.setdefault(stem, []).append((missing, path.stem))

    fixed = 0
    for stem, replacements in fixes.items():
        result = pages.get(stem)
        if not result:
            continue
        meta, body, timeline = result
        new_body = body
        for wrong, correct in replacements:
            new_body = re.sub(rf"\[\[{re.escape(wrong)}\]\]", f"[[{correct}]]", new_body)
        if new_body == body:
            continue
        related = list(meta.get("related", []))
        for wrong, correct in replacements:
            link = f"[[{correct}]]"
            if link not in related:
                related.append(link)
        corrections = ", ".join(f"[[{w}]] → [[{c}]]" for w, c in replacements)
        write_page(
            title=stem,
            page_type=meta.get("type", "concept"),
            created=meta.get("created") or today(),
            sources=meta.get("sources", []),
            related=list(dict.fromkeys(related)),
            body=new_body,
            timeline=timeline + [f"- {today()}: [corrected] fixed wikilinks {corrections}"],
        )
        print(f"  [fixed] {stem}.md ← {corrections}")
        fixed += 1
    return fixed


def fix_missing_references(issues: list[dict], pages: dict) -> int:
    fixes: dict[str, list[str]] = {}
    for issue in issues:
        if issue.get("type") == "missing_reference":
            fixes.setdefault(issue["_stem"], []).append(issue["_add"])

    fixed = 0
    for stem, titles in fixes.items():
        result = pages.get(stem)
        if not result:
            continue
        meta, body, timeline = result
        existing_related = list(meta.get("related", []))
        new_links = [f"[[{t}]]" for t in titles]
        merged = list(dict.fromkeys(existing_related + new_links))
        if merged == existing_related:
            continue
        added = ", ".join(f"[[{t}]]" for t in titles)
        write_page(
            title=stem,
            page_type=meta.get("type", "concept"),
            created=meta.get("created") or today(),
            sources=meta.get("sources", []),
            related=merged,
            body=body,
            timeline=timeline + [f"- {today()}: [linked] added cross-references to {added}"],
        )
        print(f"  [fixed] {stem}.md ← added {added} to related")
        fixed += 1
    return fixed


@app.callback(invoke_without_command=True)
def lint(ctx: typer.Context):
    """Lint wiki pages: find orphans and missing cross-references."""
    if ctx.invoked_subcommand is not None:
        return

    start_t = time.monotonic()
    all_issues: list[dict] = []

    pages = _load_all_pages()
    if not pages:
        print("No wiki pages found.")
        raise typer.Exit(0)

    print(f"Linting {len(pages)} page(s)...")

    print("\n[1/2] Orphan pages...")
    unreferenced = find_orphan_pages(pages)
    raw_broken = collect_orphan_wikilinks()
    if raw_broken:
        fixed_broken = fix_broken_wikilinks(raw_broken, pages)
        if fixed_broken:
            pages = _load_all_pages()
            raw_broken = collect_orphan_wikilinks()
            print(f"  Fixed {fixed_broken} broken wikilink(s).")
    broken_links = [
        {
            "type": "orphan",
            "pages": [f"{o['page']}.md"],
            "detail": f"Body links to [[{o['missing']}]] which does not exist as a wiki page.",
        }
        for o in raw_broken
    ]
    all_orphan_issues = unreferenced + broken_links
    all_issues.extend(all_orphan_issues)
    if all_orphan_issues:
        print(f"  {len(unreferenced)} unreferenced page(s), {len(broken_links)} broken wikilink(s) remaining:")
        for o in all_orphan_issues:
            print(f"    - {o['pages'][0]}: {o['detail']}")
    else:
        print("  None found.")

    print("\n[2/2] Missing cross-references...")
    missing = find_missing_references(pages)
    if missing:
        print(f"  {len(missing)} missing reference(s). Fixing...")
        fixed = fix_missing_references(missing, pages)
        pages = _load_all_pages()
        all_issues.extend({k: v for k, v in i.items() if not k.startswith("_")} for i in missing)
        print(f"  Fixed {fixed} page(s).")
    else:
        print("  None found.")

    elapsed = time.monotonic() - start_t

    log = load_log()
    log["lint"] = {
        "linted_at": now(),
        "time_taken": f"{elapsed:.1f}s",
        "issues": all_issues,
    }
    save_log(log)
    rebuild_index()

    print(f"\nDone. {len(all_issues)} issue(s) in {elapsed:.1f}s.")


if __name__ == "__main__":
    app()
