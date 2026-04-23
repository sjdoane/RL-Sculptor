"""Knowledge graph — populated in a later iteration.

Planned layout:
    schema.py    — node/edge types (Paper, Technique, FailureMode, ...)
    store.py     — NetworkX-backed graph, JSON+parquet on disk
    ingest.py    — arxiv search + PDF fetch + cache
    extract.py   — LLM extraction PDF → typed nodes/edges
    embed.py     — sentence-transformers embeddings
    query.py     — by_failure / by_component / by_technique / nearest
    viz.py       — pyvis HTML renderer
"""
