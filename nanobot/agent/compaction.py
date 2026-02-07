"""Lightweight context compaction and session extraction.

Compaction: summarizes older conversation history to stay within the context
window, keeping recent messages intact.

Extraction: after each conversation turn, extracts notable facts and appends
them to the daily memory note.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider

COMPACTION_PROMPT = (
    "Summarize this conversation history concisely while preserving:\n"
    "1. Key decisions made and their reasoning\n"
    "2. Important facts, names, dates, and numbers mentioned\n"
    "3. User preferences and requests\n"
    "4. Pending tasks or commitments\n"
    "5. Technical context that may be needed later\n"
    "\n"
    "Previous summary (if any):\n"
    "{previous_summary}\n"
    "\n"
    "Messages to summarize:\n"
    "{messages}\n"
    "\n"
    "Write a concise summary (max 500 words) that captures the essential context. "
    "Do not include preamble - just the summary."
)

EXTRACTION_PROMPT = (
    "Review this conversation exchange and extract any facts worth remembering "
    "long-term. Focus on:\n"
    "- User preferences, habits, or personal details shared\n"
    "- Decisions made or commitments given\n"
    "- Project names, technical choices, or configuration details\n"
    "- Anything the user would expect you to remember next time\n"
    "\n"
    "User: {user_message}\n"
    "\n"
    "Assistant: {assistant_message}\n"
    "\n"
    "If there are notable facts, respond with a short bullet list (one line per fact). "
    "If nothing is worth remembering, respond with exactly: NOTHING"
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens across a list of messages."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multimodal content (images + text)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
    return total


class MessageCompactor:
    """Compacts conversation history and extracts session facts.

    Uses a (preferably cheap) LLM to summarize older messages when the
    conversation history grows beyond a token threshold, and to extract
    notable facts after each exchange.
    """

    def __init__(self, provider: LLMProvider, model: str | None = None):
        self.provider = provider
        self.model = model

    async def compact(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str = "",
    ) -> str:
        """Summarize a list of messages into a compact summary string.

        Args:
            messages: The older messages to summarize.
            previous_summary: Any existing compaction summary to build upon.

        Returns:
            A concise summary string.
        """
        # Format messages for the prompt
        formatted = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                # Extract text from multimodal content
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            # Truncate very long individual messages
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            formatted.append(f"{role}: {content}")

        messages_text = "\n".join(formatted)
        # Cap total prompt size
        if len(messages_text) > 20000:
            messages_text = messages_text[:20000] + "\n... [truncated]"

        prompt = COMPACTION_PROMPT.format(
            previous_summary=previous_summary or "(none)",
            messages=messages_text,
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
            )
            summary = (response.content or "").strip()
            logger.info(
                f"Compacted {len(messages)} messages into ~{estimate_tokens(summary)} tokens"
            )
            return summary
        except Exception as e:
            logger.warning(f"Compaction failed: {e}")
            # Fallback: just keep the previous summary
            return previous_summary or ""

    async def extract_facts(
        self,
        user_message: str,
        assistant_message: str,
    ) -> str | None:
        """Extract notable facts from a single exchange.

        Args:
            user_message: What the user said.
            assistant_message: What the assistant replied.

        Returns:
            A bullet list of facts, or None if nothing notable.
        """
        # Skip trivially short exchanges
        if len(user_message) < 20 and len(assistant_message) < 50:
            return None

        prompt = EXTRACTION_PROMPT.format(
            user_message=user_message[:3000],
            assistant_message=assistant_message[:3000],
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=512,
                temperature=0.2,
            )
            result = (response.content or "").strip()
            if not result or "NOTHING" in result.upper():
                return None
            return result
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return None
