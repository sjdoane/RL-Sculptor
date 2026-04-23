#!/usr/bin/env bash
# Regenerate the bundled pre-extracted KG shipped at
# `examples/kg_preextracted.db`. Call this whenever
# `examples/kg_seeds_global.yml` changes.
#
# Runtime cost:
#   - ~2-5 min (arxiv PDF downloads; rate-limited).
#   - ~2-5 min (Claude entity extraction; ~150K tokens for 46 papers).
#   - Requires ANTHROPIC_API_KEY (auto-loaded from .env by sculptor).
#
# Output: overwrites examples/kg_preextracted.db on success.
#         Leaves the previous file untouched on any failure.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
OUT="$REPO_ROOT/examples/kg_preextracted.db"
SEEDS="$REPO_ROOT/examples/kg_seeds_global.yml"

if [ ! -f "$SEEDS" ]; then
    echo "error: $SEEDS not found — expected the curated seed file." >&2
    exit 1
fi

STAGING="$(mktemp -u /tmp/kg_preextracted_staging_XXXXXX.db)"
trap 'rm -f "$STAGING"' EXIT

echo "[regen] staging DB at $STAGING"

# Ingest: fetch arxiv metadata + download PDFs.
echo "[regen] ingesting $(grep -c '^  - arxiv_id:' "$SEEDS") papers ..."
SCULPTOR_KG_PATH="$STAGING" uv run python -m sculptor.kg.ingest "$SEEDS"

# Extract: Claude-powered entity extraction over each paper.
echo "[regen] extracting entities via Claude ..."
SCULPTOR_KG_PATH="$STAGING" uv run sculpt kg extract --all

# Stats: sanity-check the output.
echo "[regen] final stats:"
SCULPTOR_KG_PATH="$STAGING" uv run sculpt kg stats

# Promote staging → committed binary.
mv "$STAGING" "$OUT"
trap - EXIT
SIZE="$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")"
echo "[regen] wrote $OUT ($SIZE bytes)"
echo "[regen] commit with:"
echo "[regen]   git add $OUT && git commit -m 'kg: regenerate pre-extracted DB'"
