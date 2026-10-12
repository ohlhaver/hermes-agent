"""Policy gate for every model request that leaves outside the main loop.

The main conversation loop hands each request to the ``llm_request`` and
``llm_execution`` middleware (``agent/conversation_loop.py``), so a plugin can
check or refuse it before the provider sees anything. Every other model request
goes past that layer: the auxiliary client (``call_llm`` / ``async_call_llm``:
compression, title generation, session search, vision, memory query rewrite,
MoA slots, plugin tasks) and the iteration-limit summary in
``agent/chat_completion_helpers.py`` build and send their own requests. A data
policy that holds for the main call therefore did not hold for them, and the
compression summary in particular carries the whole conversation.

This module is the one seam for those sends. A plugin registers a guard with
:func:`register_auxiliary_send_guard`; the auxiliary client calls
:func:`guard_auxiliary_send` right before every request leaves the process --
the primary send, a same-provider rebuild after credential recovery, and every
fallback candidate -- with the route that request is about to take. A guard
may:

* return ``None`` to let the request go unchanged;
* return a dict to replace the request (for example to attach a provider
  block the route needs);
* raise :class:`AuxiliarySendRefused` to stop it. Nothing is sent and no
  fallback is tried for that refusal.

Any other exception from a guard is turned into a refusal: a guard that cannot
decide must not let the request through.

The session a request belongs to is told by :func:`current_auxiliary_session_id`:
an explicit :func:`auxiliary_session_scope` (set around context compression,
which also runs outside a turn -- gateway session hygiene, ``/compress``) and
otherwise the session published for aux accounting at turn entry
(``agent.aux_accounting``), which background-review forks and title generation
inherit.

:class:`AuxiliarySendRefused` carries status 403 so callers that classify
errors treat it as a terminal access failure: context compression then keeps
the conversation unchanged instead of replacing it with a placeholder summary
(``ContextCompressor.compress``, the access/quota carve-out).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuxiliarySendRefused(PermissionError):
    """A registered guard stopped a request before it was sent."""

    status_code = 403

    def __init__(self, message: str, *, user_message: Optional[str] = None, reason: str = "") -> None:
        super().__init__(message)
        self.user_message = user_message or message
        self.reason = reason


_GUARDS: List[Callable[..., Any]] = []
_GUARDS_LOCK = threading.Lock()

_SESSION_SCOPE: ContextVar[Optional[str]] = ContextVar("auxiliary_send_session", default=None)


def register_auxiliary_send_guard(guard: Callable[..., Any]) -> None:
    """Register *guard*; registering the same callable twice is a no-op."""
    if not callable(guard):
        raise TypeError("an auxiliary send guard must be callable")
    with _GUARDS_LOCK:
        if guard not in _GUARDS:
            _GUARDS.append(guard)


def unregister_auxiliary_send_guard(guard: Callable[..., Any]) -> None:
    with _GUARDS_LOCK:
        if guard in _GUARDS:
            _GUARDS.remove(guard)


def registered_auxiliary_send_guards() -> List[Callable[..., Any]]:
    with _GUARDS_LOCK:
        return list(_GUARDS)


@contextmanager
def auxiliary_session_scope(session_id: Optional[str]):
    """Name the session the auxiliary requests inside this block belong to."""
    token = _SESSION_SCOPE.set(str(session_id or "").strip() or None)
    try:
        yield
    finally:
        _SESSION_SCOPE.reset(token)


def current_auxiliary_session_id() -> Optional[str]:
    scoped = _SESSION_SCOPE.get()
    if scoped:
        return scoped
    try:
        from agent.aux_accounting import get_accounting_context

        ctx = get_accounting_context()
    except Exception:
        ctx = None
    if isinstance(ctx, tuple) and len(ctx) == 2 and ctx[1]:
        return str(ctx[1])
    return None


def _route_base_url(client: Any = None, base_url: Optional[str] = None) -> str:
    value = getattr(client, "base_url", None) if client is not None else None
    return str(value or base_url or "")


def guard_auxiliary_send(
    *,
    task: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    request: Dict[str, Any],
    client: Any = None,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every registered guard over *request* and return the request to send.

    Raises :class:`AuxiliarySendRefused` when a guard refuses, or when a guard
    fails in any other way (fail closed).
    """
    guards = registered_auxiliary_send_guards()
    if not guards:
        return request
    route_base_url = _route_base_url(client, base_url)
    session = str(session_id or "").strip() or current_auxiliary_session_id()
    current = request
    for guard in guards:
        try:
            result = guard(
                task=task,
                provider=str(provider or ""),
                base_url=route_base_url,
                model=str(model or ""),
                request=current,
                session_id=session,
            )
        except AuxiliarySendRefused:
            logger.warning(
                "Auxiliary %s: request refused before sending (provider=%s session=%s)",
                task or "call", provider or "auto", session or "none",
            )
            raise
        except Exception as exc:
            logger.error(
                "Auxiliary %s: send guard %s failed (%s); request refused",
                task or "call", getattr(guard, "__name__", "guard"), type(exc).__name__,
            )
            raise AuxiliarySendRefused(
                "An auxiliary send guard could not decide; the request was not sent.",
                reason="guard_error",
            ) from None
        if isinstance(result, dict):
            current = result
    return current
