"""Async message queue for decoupled channel-agent communication."""

import asyncio
import time

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage

# Rate-limiting defaults
_DEFAULT_RATE_LIMIT = 30  # max inbound messages per window
_DEFAULT_RATE_WINDOW_S = 60  # window length in seconds


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.
    
    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.

    Includes per-sender rate limiting on inbound messages to prevent
    flooding the agent with excessive LLM calls.
    """
    
    def __init__(
        self,
        rate_limit: int = _DEFAULT_RATE_LIMIT,
        rate_window_s: int = _DEFAULT_RATE_WINDOW_S,
    ):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

        # Per-sender rate limiting
        self._rate_limit = rate_limit
        self._rate_window_s = rate_window_s
        self._sender_timestamps: dict[str, list[float]] = {}
    
    def _is_rate_limited(self, sender_key: str) -> bool:
        """Check (and record) whether *sender_key* exceeds the rate limit."""
        now = time.monotonic()
        cutoff = now - self._rate_window_s
        timestamps = self._sender_timestamps.get(sender_key, [])
        # Prune old entries
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= self._rate_limit:
            self._sender_timestamps[sender_key] = timestamps
            return True
        timestamps.append(now)
        self._sender_timestamps[sender_key] = timestamps
        return False

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent.

        Messages that exceed the per-sender rate limit are silently dropped
        and a warning is logged.
        """
        sender_key = f"{msg.channel}:{msg.sender_id}"
        if self._is_rate_limited(sender_key):
            logger.warning(f"Rate-limited inbound message from {sender_key}")
            return
        await self.inbound.put(msg)
    
    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()
    
    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)
    
    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()
    
    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()
    
    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
