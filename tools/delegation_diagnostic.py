"""Content-free HPD-629 observation; never changes tool selection or dispatch.

Eligibility is a momentary snapshot, not proof of credentials, scheduling or
completion. Unknown stays null. No credential resolution or child is started.
"""
import hashlib
import json

from tools.schema_sanitizer import summarize_request_tools


def summarize_delegation_request(api_kwargs, api_mode, agent):
    result = dict(present=None, schemaSha256=None, parentPresent=agent is not None,
                  depthAllowed=None, spawnPaused=None, asyncDeliverySupported=None,
                  asyncCapacityAvailable=None)
    try:
        summary = summarize_request_tools(api_kwargs, api_mode)
        if summary["complete"]:
            result["present"] = "delegate_task" in summary["names"]
            entries = (api_kwargs.get("toolConfig", {}) if api_mode == "bedrock_converse"
                       else api_kwargs).get("tools", [])
            matches = []
            for entry in entries:
                spec = (entry.get("function") if api_mode == "chat_completions" else
                        entry.get("toolSpec") if api_mode == "bedrock_converse" else entry)
                if isinstance(spec, dict) and spec.get("name") == "delegate_task":
                    matches.append(entry)
            if len(matches) == 1:
                encoded = json.dumps(matches[0], sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()
                result["schemaSha256"] = hashlib.sha256(encoded).hexdigest()
    except Exception:
        pass
    # Reuse the actual dispatch guards. These reads neither resolve provider
    # credentials nor touch durable delegation records or the executor.
    try:
        from tools import delegate_tool
        result["spawnPaused"] = delegate_tool.is_spawn_paused()
        if agent is not None:
            depth = getattr(agent, "_delegate_depth", 0)
            if type(depth) is int:
                result["depthAllowed"] = depth < delegate_tool._get_max_spawn_depth()
    except Exception:
        pass
    try:
        from gateway.session_context import async_delivery_supported
        result["asyncDeliverySupported"] = async_delivery_supported()
    except Exception:
        pass
    try:
        from tools import async_delegation, delegate_tool
        cap = delegate_tool._get_max_concurrent_children()
        # Match dispatch_async_delegation_batch's running-only capacity guard;
        # active_count() also includes finalizing records and would misreport it.
        with async_delegation._records_lock:
            running = sum(r.get("status") == "running" for r in async_delegation._records.values())
        result["asyncCapacityAvailable"] = running < cap
    except Exception:
        pass
    return result
