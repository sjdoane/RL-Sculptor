"""Reference motion library (§R1_BUILD_SPEC).

A validated, on-disk library of retargeted G1 mocap clips (`sculptor.
reference` clip format), sourced from public HuggingFace datasets and
indexed for retrieval. Distinct from `sculptor.reference`'s procedural
RSI machinery: this package is the ingest/storage/search layer that
FEEDS a real clip into that machinery instead of a kinematic
approximation.

Layout:
    library.py — on-disk root, index.jsonl, provenance schema, clip_id
                 rules, content-hash, license guard.
    ingest.py  — dataset-specific ingesters (LAFAN1-g1 CSV, fleaven-g1
                 npy) + shared QC gates.
    segment.py — root_z hysteresis segmentation for multi-rep clips
                 (e.g. fallAndGetUp) into derived per-segment clips.
    retrieve.py, preview.py — later workers (W2); not present in R1 W1.
    track.py   — Tier-D tracking certification + `verify_tierd_
                 certificate` (§REFERENCE_TRAJECTORY_PLAN §6/§10 audit-
                 finding close): the ONLY way to turn a caller's "this is
                 Tier D" claim into a verified `TierDCertificate`.
"""

from sculptor.refs.track import TierDCertificate, verify_tierd_certificate

__all__ = ["TierDCertificate", "verify_tierd_certificate"]
