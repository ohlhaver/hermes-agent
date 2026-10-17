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


def test_malformed_stored_entry_is_repaired_on_next_calibration(home):
    (home / "native_token_ratios.json").write_text(json.dumps({DEEPSEEK: "abc", HAIKU: [1]}))
    a = _Agent(DEEPSEEK)
    assert tc.native_token_ratio(a) == 1.0
    tc.calibrate_native_token_ratio(a, 100_000, 117_000)
    stored = json.loads((home / "native_token_ratios.json").read_text())
    assert stored[DEEPSEEK] == pytest.approx(1.17)


def test_failed_write_leaves_no_temp_file(home, monkeypatch):
    import json as _json

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(_json, "dump", boom)
    a = _Agent(DEEPSEEK)
    tc.calibrate_native_token_ratio(a, 100_000, 117_000)  # must not raise
    assert [p.name for p in home.iterdir() if p.name.startswith(".native_token_ratios.")] == []


# HPD-1289: a response that used OpenRouter's server-side web search sums both
# passes of the search loop into one prompt count and must not calibrate.
# Shapes and counts from fc-justus-3, 2026-10-17 (Claude Haiku 5.5).
def _openrouter_usage(prompt, cached, web_search_requests=0, key="server_tool_use"):
    from openai.types.completion_usage import CompletionUsage

    payload = {
        "prompt_tokens": prompt,
        "completion_tokens": 412,
        "total_tokens": prompt + 412,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if web_search_requests:
        payload[key] = {"web_search_requests": web_search_requests}
    # The streamed final chunk's usage as the OpenAI client parses it.
    return CompletionUsage.model_validate(payload)


def test_web_search_usage_is_read_from_the_client_object():
    assert tc.provider_web_search_requests(_openrouter_usage(132_817, 64_735, 1)) == 1
    assert tc.provider_web_search_requests(_openrouter_usage(103_013, 50_251, 1, key="server_tool_use_details")) == 1
    assert tc.provider_web_search_requests(_openrouter_usage(52_412, 50_000)) == 0
    assert tc.provider_web_search_requests({"server_tool_use": {"web_search_requests": 2}}) == 2
    assert tc.provider_web_search_requests(None) == 0


def test_search_answer_keeps_the_last_good_ratio_and_the_next_preflight_stays_below_85k(home):
    a = _Agent(HAIKU)
    tc.native_token_ratio(a)
    # Clean answer: calibrates (HPD-1151 measurement, ~1.76).
    tc.calibrate_native_token_ratio(a, 37_200, 65_468, provider_usage=_openrouter_usage(65_468, 60_000))
    good = a.context_compressor.native_token_ratio
    assert good == pytest.approx(1.76, abs=0.01)
    # 10:05:27 call #3: web search, in=132,817 against a rough ~38.5k.
    tc.calibrate_native_token_ratio(a, 38_536, 132_817, provider_usage=_openrouter_usage(132_817, 64_735, 1))
    assert a.context_compressor.native_token_ratio == good
    assert json.loads((home / "native_token_ratios.json").read_text())[HAIKU] == pytest.approx(good, abs=0.001)
    # The next preflight: rough 38,536 must not read as 115,608 (3.0).
    assert tc.trigger_pressure_tokens(a, 38_536) < 85_000
    assert tc.trigger_pressure_tokens(_Agent(HAIKU), 38_536) < 85_000


def test_a_stuck_stored_ratio_is_reset_by_the_next_clean_answer(home):
    (home / "native_token_ratios.json").write_text(json.dumps({HAIKU: 3.0}))
    a = _Agent(HAIKU)
    assert tc.native_token_ratio(a) == 3.0
    # A search answer leaves it alone, the next clean answer repairs memory and disk.
    tc.calibrate_native_token_ratio(a, 38_536, 132_817, provider_usage=_openrouter_usage(132_817, 64_735, 1))
    assert a.context_compressor.native_token_ratio == 3.0
    tc.calibrate_native_token_ratio(a, 29_900, 52_412, provider_usage=_openrouter_usage(52_412, 50_000))
    assert a.context_compressor.native_token_ratio == pytest.approx(1.75, abs=0.01)
    assert json.loads((home / "native_token_ratios.json").read_text())[HAIKU] == pytest.approx(1.75, abs=0.01)
    assert tc.native_token_ratio(_Agent(HAIKU)) == pytest.approx(1.75, abs=0.01)
