"""Async message queue for decoupled channel-agent communication."""

import asyncio
import time
from collections import defaultdict

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage

# Per-sender rate limiting defaults
_DEFAULT_RATE_LIMIT = 30       # max messages per window
_DEFAULT_RATE_WINDOW_S = 60    # window size in seconds


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
        rate_window: float = _DEFAULT_RATE_WINDOW_S,
    ):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._rate_limit = rate_limit
        self._rate_window = rate_window
        self._sender_timestamps: dict[str, list[float]] = defaultdict(list)
    
    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent.

        Applies per-sender rate limiting; excess messages are silently dropped.
        """
        now = time.monotonic()
        key = f"{msg.channel}:{msg.sender_id}"
        timestamps = self._sender_timestamps[key]

        # Prune timestamps outside the window
        cutoff = now - self._rate_window
        self._sender_timestamps[key] = timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= self._rate_limit:
            logger.warning(
                f"Rate limit hit for {key} ({self._rate_limit}/{self._rate_window}s) – dropping message"
            )
            return

        timestamps.append(now)
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
