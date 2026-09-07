"""Compatible terminal outcome validation for the existing cron stores."""


def resolve_outcome(success: bool | None, outcome: str | None) -> str:
    if outcome is None:
        if success is None:
            raise ValueError("success or an explicit outcome is required")
        return "completed" if success else "failed"
    if outcome not in ("completed", "failed", "unknown"):
        raise ValueError("invalid execution outcome")
    if success is not None and (
        outcome == "unknown" or bool(success) != (outcome == "completed")
    ):
        raise ValueError("success contradicts explicit outcome")
    return outcome
