"""Shared tool-result truncation for agent loops."""

import json
import re

# Default cap for tool output included in LLM context
DEFAULT_MAX_CHARS = 3000


def truncate_tool_result(result: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Truncate a tool result to keep LLM context lean.

    Strips ANSI escape codes, then applies format-aware truncation:
    - JSON is pretty-printed and prefix-truncated so visible output stays valid.
    - Plain text uses head truncation with a clear sentinel.
    """
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', result)
    if len(clean) <= max_chars:
        return clean

    stripped = clean.lstrip()
    if stripped and stripped[0] in ('{', '['):
        try:
            parsed = json.loads(clean)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            if len(pretty) <= max_chars:
                return pretty
            budget = max_chars - 120
            return (
                pretty[:budget]
                + f"\n\n... [JSON truncated - showed {budget} of {len(pretty)} chars. "
                + "Do NOT re-run this tool to see more.]"
            )
        except (json.JSONDecodeError, ValueError):
            pass

    budget = max_chars - 100
    return (
        clean[:budget]
        + f"\n\n... [truncated - showed {budget} of {len(clean)} chars. "
        + "Do NOT re-run this tool to see more.]"
    )
