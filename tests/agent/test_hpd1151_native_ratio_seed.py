"""HPD-1151 follow-up: the native-token ratio seed is per model and survives a restart.

On fc-e2e-2 (2026-10-11) the fixed 1.8 seed, measured on Claude Haiku 5.5,
made a freshly restarted DeepSeek agent (real ratio about 1.17) compact at
~78k native instead of 85k. The seed is now the last ratio measured for the
same model, persisted under HERMES_HOME; a model never measured starts at 1.0.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent import turn_context as tc

HAIKU = "anthropic/claude-haiku-5.5"
DEEPSEEK = "deepseek/deepseek-v4-flash-0731"


class _Compressor:
    def __init__(self):
        self.native_token_ratio = 0.0


class _Agent:
    def __init__(self, model, listed=True):
        self.model = model
        self._current_turn_sidecar_models = (model,) if listed else ()
        self.context_compressor = _Compressor()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_native_ratio_store_path", lambda: tmp_path / "native_token_ratios.json")
    return tmp_path


def test_unmeasured_model_seeds_at_one(home):
    a = _Agent(DEEPSEEK)
    assert tc.native_token_ratio(a) == 1.0
    assert a.context_compressor.native_token_ratio == 1.0


def test_calibration_persists_per_model_and_seeds_a_fresh_agent(home):
    a = _Agent(DEEPSEEK)
    tc.native_token_ratio(a)
    tc.calibrate_native_token_ratio(a, 100_000, 117_000)
    assert a.context_compressor.native_token_ratio == pytest.approx(1.17)
    # A restarted gateway builds a fresh agent: it starts from the stored value.
    fresh = _Agent(DEEPSEEK)
    assert tc.native_token_ratio(fresh) == pytest.approx(1.17)
    # Another model keeps its own value.
    h = _Agent(HAIKU)
    tc.calibrate_native_token_ratio(h, 84_184, 148_151)
    assert tc.native_token_ratio(_Agent(HAIKU)) == pytest.approx(1.76, abs=0.01)
    assert tc.native_token_ratio(_Agent(DEEPSEEK)) == pytest.approx(1.17)
    stored = json.loads((home / "native_token_ratios.json").read_text())
    assert set(stored) == {HAIKU, DEEPSEEK}


def test_bounds_hold_for_stored_values(home):
    (home / "native_token_ratios.json").write_text(json.dumps({DEEPSEEK: 9.0, HAIKU: 0.2}))
    assert tc.native_token_ratio(_Agent(DEEPSEEK)) == 3.0
    assert tc.native_token_ratio(_Agent(HAIKU)) == 1.0


def test_unreadable_store_falls_back_to_one(home):
    (home / "native_token_ratios.json").write_text("{not json")
    assert tc.native_token_ratio(_Agent(DEEPSEEK)) == 1.0


def test_unlisted_model_neither_reads_nor_writes(home):
    a = _Agent(DEEPSEEK, listed=False)
    assert tc.native_token_ratio(a) == 0.0
    tc.calibrate_native_token_ratio(a, 100_000, 117_000)
    assert not (home / "native_token_ratios.json").exists()
