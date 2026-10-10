"""HPD-1148: a finished cron job must not leave ``active_agents`` stuck in
``gateway_state.json``.

``_active_work_count`` folds in-flight cron jobs (``cron.scheduler.
_running_job_ids``) and API-server runs into ``active_agents``, but the file is
only rewritten at messaging turn boundaries and lifecycle transitions. A cron
job leaves ``_running_job_ids`` in the scheduler's worker thread and nothing
rewrites the file then, so any write that happened while the job ran (a chat
turn boundary, a lifecycle write) freezes ``active_agents=1`` until the next
messaging turn. Operators that refuse disruptive actions unless the count is
0 then refuse indefinitely.

Driven through the real producer: ``cron.scheduler.tick(sync=False)`` (as the
built-in scheduler provider calls it) dispatches the job onto the real pool,
``_run_and_release`` removes it in the worker thread, and the real gateway
watcher and ``write_runtime_status`` write the real file.
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


# Other gateway tests reload gateway.status; look its functions up at call time
# so the sentinel they compare against is the current one.
def read_runtime_status():
    from gateway import status
    return status.read_runtime_status()


def write_runtime_status(**kwargs):
    from gateway import status
    status.write_runtime_status(**kwargs)


def _persisted_active_agents():
    record = read_runtime_status()
    assert isinstance(record, dict)
    return record.get("active_agents")


async def _wait_for(predicate, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


@pytest.fixture
def scheduler(monkeypatch):
    import cron.scheduler as sched

    sched._parallel_pool = None
    sched._parallel_pool_max_workers = None
    sched._running_job_ids.clear()
    yield sched
    sched._running_job_ids.clear()
    sched._shutdown_parallel_pool()


@pytest.mark.asyncio
async def test_finished_cron_job_returns_persisted_active_agents_to_zero(scheduler, monkeypatch):
    sched = scheduler
    job = {
        "id": "6894f37a03cd",
        "name": "Wassertemperatur aktualisieren",
        "prompt": "test",
        "schedule": "0 18 * * *",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00",
        "deliver": "local",
    }
    started = threading.Event()
    release = threading.Event()

    def run_job(j, **_kw):
        started.set()
        assert release.wait(5)
        return True, "out", "resp", None

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "run_job", run_job)
    monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

    runner, _ = make_restart_runner()
    runner._external_drain_active = False
    write_runtime_status(gateway_state="running", active_agents=0)

    watcher = asyncio.create_task(runner._drain_control_watcher(interval=0.02))
    try:
        assert sched.tick(verbose=False, sync=False) == 1
        assert await asyncio.to_thread(started.wait, 5)
        assert runner._active_work_count() == 1

        # Some write during the run - here a chat turn boundary - records the
        # in-flight cron job, exactly as fc-justus-3 showed active_agents=1.
        runner._persist_active_agents()
        assert _persisted_active_agents() == 1

        release.set()
        assert await _wait_for(lambda: not sched.get_running_job_ids())
        assert await _wait_for(lambda: _persisted_active_agents() == 0), (
            "active_agents stayed " + json.dumps(_persisted_active_agents())
            + " after the cron job finished"
        )
        # Lifecycle state is left as it was.
        assert read_runtime_status()["gateway_state"] == "running"
    finally:
        release.set()
        runner._running = False
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_watcher_does_not_rewrite_an_accurate_count(monkeypatch):
    runner, _ = make_restart_runner()
    runner._external_drain_active = False
    write_runtime_status(gateway_state="running", active_agents=0)
    writes = []
    original = runner._persist_active_agents
    monkeypatch.setattr(runner, "_persist_active_agents", lambda: writes.append(1) or original())

    watcher = asyncio.create_task(runner._drain_control_watcher(interval=0.01))
    await asyncio.sleep(0.08)
    runner._running = False
    watcher.cancel()
    try:
        await watcher
    except asyncio.CancelledError:
        pass

    assert writes == []
    assert _persisted_active_agents() == 0


@pytest.mark.asyncio
async def test_finished_api_server_run_returns_persisted_active_agents_to_zero():
    from unittest.mock import MagicMock

    from gateway.config import Platform

    api_runs = {"count": 1}
    runner, _ = make_restart_runner()
    runner._external_drain_active = False
    runner.adapters = {
        Platform.API_SERVER: MagicMock(active_agent_work_count=lambda: api_runs["count"])
    }
    write_runtime_status(gateway_state="running", active_agents=0)
    runner._persist_active_agents()
    assert _persisted_active_agents() == 1

    watcher = asyncio.create_task(runner._drain_control_watcher(interval=0.02))
    try:
        api_runs["count"] = 0
        assert await _wait_for(lambda: _persisted_active_agents() == 0)
    finally:
        runner._running = False
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
