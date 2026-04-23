"""tests/test_load_adapter.py — Phase 2 helper test for
`sculptor.adapters.base.load_adapter(config_path)`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOPPER_CONFIG = REPO_ROOT / "examples" / "hopper" / "config.toml"


def test_load_adapter_returns_gym_sb3_adapter():
    from sculptor.adapters.base import load_adapter
    from sculptor.adapters.gym_sb3 import GymSB3Adapter

    adapter = load_adapter(HOPPER_CONFIG)
    assert isinstance(adapter, GymSB3Adapter)
    assert adapter.env_id == "Hopper-v4"
    assert adapter.n_envs == 4


def test_load_adapter_rejects_missing_adapter_section(tmp_path):
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "bad.toml"
    cfg.write_text("[target]\nname = 'x'\n")
    with pytest.raises(ValueError, match="missing \\[adapter\\] section"):
        load_adapter(cfg)


def test_load_adapter_rejects_missing_class(tmp_path):
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "bad.toml"
    cfg.write_text("[adapter]\nconfig = {}\n")
    with pytest.raises(ValueError, match="adapter\\].class is required"):
        load_adapter(cfg)


def test_load_adapter_rejects_non_adapter_class(tmp_path):
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        "[adapter]\n"
        "class = \"pathlib.Path\"\n"
        "config = {}\n"
    )
    with pytest.raises(TypeError, match="must subclass SculptorAdapter"):
        load_adapter(cfg)
