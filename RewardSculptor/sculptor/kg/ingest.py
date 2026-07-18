"""sculptor/kg/ingest.py — arXiv → KG paper ingestion (no LLM yet).

What this does:
  - Fetch arXiv metadata via the `arxiv` package.
  - Download the PDF into `<kg_root>/pdfs/<arxiv_id>.pdf`.
  - Extract full text via pypdf; heuristically split abstract and conclusion.
  - Upsert a `Paper` node with `extracted=False`. LLM extraction runs later
    and flips the flag.

What this deliberately does NOT do:
  - Call any LLM. KG node extraction (Technique, FailureMode,
    RewardComponent) is a later prompt.
  - Parse citations. Edge creation happens during LLM extraction.

Idempotency:
  - If a Paper node with the target id already exists AND its PDF file is on
    disk, we return the existing Paper without a network call.
  - If the node exists but the PDF was deleted, we re-download and update
    the node (full_text re-extracted, extracted flag preserved).

Run as:
    uv run python -m sculptor.kg.ingest <seeds.yml>
    uv run python -m sculptor.kg.ingest --arxiv 1707.06347

Seeds format (see examples/hopper/kg_seeds.yml):
    papers:
      - arxiv_id: "1707.06347"
        title: "PPO"
        rationale: "why we ingested this"
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sculptor.kg.schema import Paper, make_paper_id
from sculptor.kg.store import SculptorKG


_CONCLUSION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:Conclusions?|Discussion(?:\s+and\s+Conclusions?)?|"
    r"Conclusions?\s+and\s+(?:Future\s+Work|Discussion)|Summary)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_CONCLUSION_END_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:References?|Bibliography|Acknowledg(?:e?)ments?|Appendi(?:ces|x)|Supplementar(?:y|ies))\b",
    re.MULTILINE | re.IGNORECASE,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _pdfs_dir(store: SculptorKG) -> Path:
    out = store.db_path.parent / "pdfs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_pdf_name(arxiv_id: str) -> str:
    # arxiv ids are safe for filenames except for the "/" in old-style ids.
    return arxiv_id.replace("/", "_") + ".pdf"


def _extract_full_text(pdf_path: Path) -> str:
    """Best-effort text extraction via pypdf. Returns '' on failure."""
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages)
    except Exception as e:  # corrupt PDF, missing font, etc.
        print(f"[ingest] warning: pypdf failed to extract {pdf_path.name}: {e}",
              file=sys.stderr)
        return ""


def _extract_conclusion(full_text: str) -> str:
    """Heuristic conclusion extraction. Returns '' when no heading matches.

    Strategy: find the LAST conclusion/discussion heading (many papers have
    'Related Work discussions' earlier that we want to skip), then take text
    until a References/Acknowledgments/Appendix heading.
    """
    if not full_text:
        return ""
    matches = list(_CONCLUSION_HEADING_RE.finditer(full_text))
    if not matches:
        return ""
    start = matches[-1].end()
    end_match = _CONCLUSION_END_RE.search(full_text, pos=start)
    end = end_match.start() if end_match else min(len(full_text), start + 12_000)
    return full_text[start:end].strip()


def _normalize_arxiv_id(s: str) -> str:
    """Strip URL wrappers / version suffix so ids round-trip cleanly."""
    s = s.strip()
    s = s.removeprefix("https://arxiv.org/abs/")
    s = s.removeprefix("http://arxiv.org/abs/")
    s = s.removeprefix("arxiv:")
    # Strip version suffix like v2 / v12
    s = re.sub(r"v\d+$", "", s)
    return s


def _save_full_text_sidecar(pdfs_dir: Path, arxiv_id: str, text: str) -> Path:
    """Write the extracted plain text next to the PDF for later reuse."""
    path = pdfs_dir / (_safe_pdf_name(arxiv_id).replace(".pdf", ".txt"))
    path.write_text(text, encoding="utf-8", errors="replace")
    return path


def _download_pdf_with_timeout(url: str, dest: Path, *, timeout_s: float) -> None:
    """Bounded-time PDF download — urlretrieve doesn't honor timeouts.

    We stream into a .partial file and rename on success so a stalled or
    interrupted download never leaves a partial `.pdf` that the idempotency
    check would later mistake for a successful ingest.
    """
    import urllib.request

    partial = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Reward-Sculptor/0.1 (arxiv ingest)"})
    # Use socket-level timeout for the initial connect + idle reads.
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, partial.open("wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    partial.replace(dest)


def make_arxiv_client(
    *,
    page_size: int,
    delay_seconds: float,
    num_retries: int,
    timeout_s: float,
):
    """Build an `arxiv.Client` whose HTTP timeout is scoped to ITS OWN
    requests session.

    §Phase-0 hardening 2026-07-18: the old approach set
    `socket.setdefaulttimeout(timeout_s)` around the call — a
    process-GLOBAL mutation. The UI backend runs ingest/research inside
    worker threads of a live uvicorn server, so any socket created by
    ANOTHER thread during that window (a websocket, a long-poll response)
    silently inherited a 30s timeout and could die mid-stream. The arxiv
    package keeps a `requests.Session` at `client._session`; wrapping its
    `request` with a default timeout confines the bound to arxiv traffic.
    Fallback: if a future arxiv version drops `_session`, the client is
    returned un-bounded (requests' default) rather than reintroducing the
    global mutation."""
    import arxiv

    client = arxiv.Client(
        page_size=page_size, delay_seconds=delay_seconds,
        num_retries=num_retries)
    sess = getattr(client, "_session", None)
    if sess is not None and hasattr(sess, "request"):
        import functools

        sess.request = functools.partial(sess.request, timeout=timeout_s)
    return client


# ── Public API ──────────────────────────────────────────────────────────────
ARXIV_PDF_URL_FMT = "https://arxiv.org/pdf/{arxiv_id}.pdf"


# §7.7: exponential backoff schedule for arxiv API retries. Arxiv's
# rate-limit window is ~2 min; spacing retries at 10/30/60/120 s gives
# one attempt immediately, two within the rate-limit window, and one
# after the window lifts. Tuning rationale: bulk-ingest failures from
# the cartwheel session (CONTEXT 00:15) showed ~9 papers stubbing out
# with `arxiv:XXXX.XXXXX` titles because the first attempt landed
# during rate-limit and the single retry also hit it. Four attempts
# covers ~3.7 min of wall-clock — longer than any observed rate-limit
# window on arxiv's public API.
_ARXIV_RETRY_DELAYS_S: tuple[float, ...] = (0.0, 10.0, 30.0, 60.0, 120.0)


def _fetch_arxiv_metadata(
    arxiv_id: str,
    *,
    timeout_s: float = 30.0,
    retry_delays_s: tuple[float, ...] | None = None,
    sleep_fn=None,
) -> dict[str, Any] | None:
    """Best-effort metadata fetch. Returns None (not raises) on rate-limit / timeout.

    The arxiv export API is aggressively rate-limited: any recent burst of
    requests makes this call return 'Rate exceeded' for minutes. We retry
    with exponential-ish backoff (§7.7) before giving up, then return None
    so callers can fall back to the seed YAML (which already has title,
    authors, etc.).

    `retry_delays_s` + `sleep_fn` are for tests — default schedule is
    defined at module scope, default sleep is `time.sleep`.
    """
    import time

    delays = retry_delays_s if retry_delays_s is not None else _ARXIV_RETRY_DELAYS_S
    sleep = sleep_fn if sleep_fn is not None else time.sleep
    last_err: str | None = None
    for attempt, delay in enumerate(delays):
        if delay > 0:
            print(
                f"[ingest]   arxiv retry {attempt}/{len(delays) - 1} for "
                f"{arxiv_id} after {delay:.0f}s (last error: {last_err})",
                file=sys.stderr, flush=True,
            )
            sleep(delay)
        try:
            client = make_arxiv_client(
                page_size=1, delay_seconds=1.0, num_retries=1,
                timeout_s=timeout_s)
            import arxiv

            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search))

            return {
                "title": str(result.title).strip(),
                "authors": [a.name for a in result.authors],
                "year": result.published.year if result.published is not None else None,
                "abstract": str(result.summary or "").strip(),
            }
        except Exception as e:  # noqa: BLE001 — every failure path retries
            last_err = f"{type(e).__name__}: {e}"
            if attempt == len(delays) - 1:
                print(
                    f"[ingest]   arxiv API unreachable after {len(delays)} "
                    f"attempts ({last_err}) — falling back to seed metadata",
                    file=sys.stderr, flush=True,
                )
                return None
    return None


def heal_stub_titles(
    store: SculptorKG | None = None,
    *,
    force: bool = True,
) -> dict[str, str]:
    """Scan the KG for Paper nodes whose title starts with `arxiv:`
    (stubs left by §7.7's fallback path when the arxiv API was
    unreachable) and re-ingest each one. Returns a map of
    `{arxiv_id: result}` where result is "healed" | "still_stubbed" |
    "error: ...".

    Call this after any bulk ingest to close out stubs. Cheap on a
    healthy network (~2 min for ~10 stubs); a no-op when no stubs
    exist. Intended to be called from the CLI (`sculpt kg heal-stubs`)
    or invoked programmatically by the backend.
    """
    owns_store = store is None
    store = store or SculptorKG()
    results: dict[str, str] = {}
    try:
        # Gather stubs first so we don't re-scan while mutating.
        stubs: list[str] = []
        for node in store.find_nodes(kind="Paper"):
            title = getattr(node, "title", "") or ""
            if isinstance(title, str) and title.startswith("arxiv:"):
                aid = getattr(node, "arxiv_id", "")
                if aid:
                    stubs.append(str(aid))
        if not stubs:
            return results
        print(f"[heal] found {len(stubs)} stub-titled paper(s)", flush=True)
        for aid in stubs:
            try:
                paper = ingest_arxiv(aid, store=store, force=force)
                new_title = getattr(paper, "title", "") or ""
                if new_title.startswith("arxiv:"):
                    results[aid] = "still_stubbed"
                    print(
                        f"[heal]   {aid}: still stubbed (arxiv API "
                        f"still rate-limited)", flush=True,
                    )
                else:
                    results[aid] = "healed"
                    print(
                        f"[heal]   {aid}: -> {new_title[:80]}", flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                results[aid] = f"error: {type(e).__name__}: {e}"
                print(
                    f"[heal]   {aid}: error {e}", file=sys.stderr, flush=True,
                )
    finally:
        if owns_store:
            store.close()
    return results


def heal_dead_text_paths(
    store: SculptorKG | None = None,
    *,
    force: bool = True,
) -> dict[str, str]:
    """Repair Paper nodes whose `full_text_path` no longer exists on disk.

    §Phase-0 hardening 2026-07-18: papers ingested while the KG was rooted
    somewhere transient (measured live: 20 papers pointing at a wiped
    `/tmp/pdfs/`) keep a dead sidecar path. Nothing breaks until the next
    `extract --force`, which then silently loses the BODY_EXCERPT and
    extracts from abstract+conclusion alone — a quality regression with no
    error. Re-running `ingest_arxiv(force=True)` re-downloads the PDF into
    THIS store's pdfs dir, rewrites the sidecar, and preserves the
    `extracted` flag. Returns `{arxiv_id: "healed" | "still_dead" |
    "error: ..."}`; papers with a live path are untouched."""
    owns_store = store is None
    store = store or SculptorKG()
    results: dict[str, str] = {}
    try:
        dead: list[str] = []
        for node in store.find_nodes(kind="Paper"):
            path = getattr(node, "full_text_path", None)
            if path and not Path(path).is_file():
                aid = getattr(node, "arxiv_id", "")
                if aid:
                    dead.append(str(aid))
        if not dead:
            return results
        print(f"[heal] found {len(dead)} paper(s) with dead full_text_path",
              flush=True)
        for aid in dead:
            try:
                paper = ingest_arxiv(aid, store=store, force=force)
                new_path = getattr(paper, "full_text_path", None)
                if new_path and Path(new_path).is_file():
                    results[aid] = "healed"
                    print(f"[heal]   {aid}: -> {new_path}", flush=True)
                else:
                    results[aid] = "still_dead"
                    print(f"[heal]   {aid}: still dead", flush=True)
            except Exception as e:  # noqa: BLE001
                results[aid] = f"error: {type(e).__name__}: {e}"
                print(f"[heal]   {aid}: error {e}", file=sys.stderr,
                      flush=True)
    finally:
        if owns_store:
            store.close()
    return results


def ingest_arxiv(
    arxiv_id: str,
    *,
    store: SculptorKG | None = None,
    force: bool = False,
    fallback_metadata: dict[str, Any] | None = None,
) -> Paper:
    """Ingest a single arXiv paper. Returns the Paper node (new or existing).

    `fallback_metadata`: used for title/authors/year/abstract when the arXiv
    export API is rate-limited. Pass fields from the seeds YAML when
    available; otherwise we'll store stub values for now and rely on the
    later LLM-extraction step to populate details from the PDF.

    Set `force=True` to re-download even if an existing node is present.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)
    owns_store = store is None
    store = store or SculptorKG()
    try:
        paper_id = make_paper_id(arxiv_id)
        existing = store.get_node(paper_id)

        pdf_path = _pdfs_dir(store) / _safe_pdf_name(arxiv_id)
        if existing is not None and not force and pdf_path.exists():
            return existing  # fully idempotent: node + file both present

        # ── Metadata: arxiv API → fallback → stub ─────────────────────────
        meta = _fetch_arxiv_metadata(arxiv_id)
        if meta is None:
            fm = fallback_metadata or {}
            meta = {
                "title": fm.get("title", f"arxiv:{arxiv_id}"),
                "authors": fm.get("authors", []) or [],
                "year": fm.get("year"),
                "abstract": fm.get("abstract", "") or "",
            }

        # ── PDF download (direct CDN URL, not API) ────────────────────────
        if not pdf_path.exists() or force:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_url = ARXIV_PDF_URL_FMT.format(arxiv_id=arxiv_id)
            _download_pdf_with_timeout(pdf_url, pdf_path, timeout_s=60.0)

        full_text = _extract_full_text(pdf_path)
        text_path = _save_full_text_sidecar(_pdfs_dir(store), arxiv_id, full_text)
        conclusion = _extract_conclusion(full_text)

        paper = Paper(
            id=paper_id,
            arxiv_id=arxiv_id,
            title=meta["title"],
            authors=list(meta["authors"]),
            year=meta["year"],
            abstract=meta["abstract"],
            conclusion_text=conclusion,
            full_text_path=str(text_path),
            ingested_at=time.time(),
            extracted=existing.extracted if isinstance(existing, Paper) else False,
        )
        store.add_node(paper, upsert=True)
        return paper
    finally:
        if owns_store:
            store.close()


def ingest_from_seeds(
    seeds_path: Path | str,
    *,
    store: SculptorKG | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Batch-ingest every arxiv_id in a YAML seeds file.

    YAML shape:
        papers:
          - arxiv_id: "1707.06347"
            title: "PPO"
            rationale: "..."

    Returns a dict with per-id outcomes: {"<arxiv_id>": "ingested" | "already_present" | "error: <msg>"}.
    """
    seeds_path = Path(seeds_path).expanduser().resolve()
    owns_store = store is None
    store = store or SculptorKG()

    results: dict[str, Any] = {}
    try:
        with seeds_path.open("r", encoding="utf-8") as f:
            seeds_doc = yaml.safe_load(f) or {}
        papers = seeds_doc.get("papers", [])
        if not isinstance(papers, list):
            raise ValueError(f"{seeds_path}: `papers` must be a list")

        print(f"[ingest] reading {len(papers)} seed(s) from {seeds_path}", flush=True)
        for entry in papers:
            fallback: dict[str, Any] = {}
            if isinstance(entry, str):
                arxiv_id = entry
                rationale = ""
            elif isinstance(entry, dict):
                arxiv_id = entry.get("arxiv_id") or entry.get("id")
                rationale = entry.get("rationale", "") or ""
                if not arxiv_id:
                    results["<missing arxiv_id>"] = "error: missing arxiv_id field"
                    continue
                # Forward any fields the user provided as metadata fallback.
                for k in ("title", "authors", "year", "abstract"):
                    if entry.get(k):
                        fallback[k] = entry[k]
            else:
                results[str(entry)] = f"error: unsupported entry type {type(entry).__name__}"
                continue

            paper_id = make_paper_id(_normalize_arxiv_id(arxiv_id))
            existing = store.get_node(paper_id)
            pdf_present = (
                _pdfs_dir(store) / _safe_pdf_name(_normalize_arxiv_id(arxiv_id))
            ).exists()
            if existing is not None and pdf_present and not force:
                print(f"[ingest]   skip {arxiv_id} — already present", flush=True)
                results[arxiv_id] = "already_present"
                continue

            try:
                print(f"[ingest]   fetch {arxiv_id}  ({rationale[:60]})", flush=True)
                paper = ingest_arxiv(
                    arxiv_id, store=store, force=force,
                    fallback_metadata=fallback or None,
                )
                results[arxiv_id] = "ingested"
                print(f"[ingest]     -> {paper.title[:80]}", flush=True)
            except Exception as e:  # noqa: BLE001
                results[arxiv_id] = f"error: {type(e).__name__}: {e}"
                print(f"[ingest]     !! error: {e}", file=sys.stderr, flush=True)
    finally:
        if owns_store:
            store.close()
    return results


# ── CLI entry ───────────────────────────────────────────────────────────────
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m sculptor.kg.ingest",
        description="Ingest arXiv papers into the Sculptor KG.",
    )
    ap.add_argument("seeds", nargs="?", type=Path,
                    help="Path to a seeds YAML file (see examples/).")
    ap.add_argument("--arxiv", action="append", default=[],
                    help="Directly ingest an arxiv id. Can repeat.")
    ap.add_argument("--store", type=Path, default=None,
                    help="Override KG DB path (default: $SCULPTOR_KG_PATH / "
                         "$RS_KG_PATH, else the shared "
                         "~/.local/share/sculptor/kg/graph.db).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the paper is already ingested.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if not args.seeds and not args.arxiv:
        print("error: specify a seeds file or --arxiv ID", file=sys.stderr)
        return 2

    store = SculptorKG(args.store) if args.store else SculptorKG()
    total: dict[str, Any] = {}
    try:
        if args.seeds:
            total.update(ingest_from_seeds(args.seeds, store=store, force=args.force))
        for arxiv_id in args.arxiv:
            try:
                ingest_arxiv(arxiv_id, store=store, force=args.force)
                total[arxiv_id] = "ingested"
            except Exception as e:
                total[arxiv_id] = f"error: {e}"

        # concise summary
        n_ingested = sum(1 for v in total.values() if v == "ingested")
        n_skipped = sum(1 for v in total.values() if v == "already_present")
        n_error = sum(1 for v in total.values() if isinstance(v, str) and v.startswith("error"))
        print(f"[ingest] done. ingested={n_ingested} already_present={n_skipped} errors={n_error}", flush=True)
        for k, v in total.items():
            print(f"  {k}: {v}", flush=True)
    finally:
        store.close()
    return 0 if all(v != "error" and not (isinstance(v, str) and v.startswith("error"))
                    for v in total.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
