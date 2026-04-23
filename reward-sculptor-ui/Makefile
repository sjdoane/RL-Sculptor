# Reward Sculptor UI — developer tasks.
#
# Nothing in this Makefile runs automatically. These targets exist for
# contributor-side scaffolding only; normal checkouts should NOT invoke
# any of them. They're kept here so we don't regrow the same ad-hoc
# instructions in the README every release.

.PHONY: generate-thumbnails test

# One-shot regeneration of the library-robot thumbnails used by the
# RobotConfig picker. Outputs at frontend/public/robots/*.png are
# checked into git as the source of truth.
#
# *** DO NOT RUN THIS ON A NORMAL CHECKOUT ***
#
# Re-running can produce minor pixel drift (MuJoCo's renderer is not
# bit-for-bit reproducible across machines / driver versions / Python
# versions), leaving a dirty git working tree. That is expected. Only
# re-run when:
#   - Adding a new entry to LIBRARY_ROBOT_MAP.
#   - Intentionally changing the default camera preset or thumbnail size.
#   - Gymnasium ships a new MJCF for one of the existing robots.
#
# Windows users without `make`: run the target command directly:
#   UV_LINK_MODE=copy uv run python scripts/generate_library_thumbnails.py
generate-thumbnails:
	UV_LINK_MODE=copy uv run python scripts/generate_library_thumbnails.py

# Shortcut for the normal test suite.
test:
	UV_LINK_MODE=copy uv run pytest backend/tests/ -v
