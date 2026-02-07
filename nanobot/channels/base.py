"""Base channel interface for chat platforms."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.
    
    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the nanobot message bus.
    """
    
    name: str = "base"
    
    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the channel.
        
        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._running = False
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.
        
        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass
    
    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.
        
        Args:
            msg: The message to send.
        """
        pass
    
    def is_allowed(self, sender_id: str) -> bool:
        """
        Check if a sender is allowed to use this bot.
        
        Args:
            sender_id: The sender's identifier.
        
        Returns:
            True if allowed, False otherwise.
        """
        allow_list = getattr(self.config, "allow_from", [])
        
        # If no allow list, allow everyone
        if not allow_list:
            return True
        
        # Normalize: strip leading '+' so both "+1234" and "1234" match
        normalized = {a.lstrip("+") for a in allow_list}
        
        sender_str = str(sender_id)
        parts = sender_str.split("|") if "|" in sender_str else [sender_str]
        for part in parts:
            if part and (part in normalized or part.lstrip("+") in normalized):
                return True
        return False
    
    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def split_message(text: str, limit: int = 2000) -> list[str]:
        """Split a long message into chunks that fit within *limit* chars.

        Tries paragraph boundaries first, then single newlines, then hard-cuts.
        """
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            slice_ = remaining[:limit]
            break_at = slice_.rfind("\n\n")
            if break_at < limit // 3:
                break_at = slice_.rfind("\n")
            if break_at < limit // 3:
                break_at = limit
            chunks.append(remaining[:break_at].rstrip())
            remaining = remaining[break_at:].lstrip("\n")
        return chunks

    # ------------------------------------------------------------------
    # Media helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_media_dir() -> Path:
        """Return (and ensure) the shared media download directory."""
        d = Path.home() / ".nanobot" / "media"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def _download_media(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        suffix: str = "",
    ) -> Path:
        """Download a URL to the shared media directory.

        Returns the local Path on success.  Filenames are derived from the
        URL content hash to avoid collisions while deduplicating identical
        downloads.
        """
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers or {})
            resp.raise_for_status()
            data = resp.content

        name = hashlib.sha256(data).hexdigest()[:16] + suffix
        dest = self._get_media_dir() / name
        dest.write_bytes(data)
        logger.debug(f"Downloaded media to {dest} ({len(data)} bytes)")
        return dest

    # ------------------------------------------------------------------

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Handle an incoming message from the chat platform.
        
        This method checks permissions and forwards to the bus.
        
        Args:
            sender_id: The sender's identifier.
            chat_id: The chat/channel identifier.
            content: Message text content.
            media: Optional list of media URLs.
            metadata: Optional channel-specific metadata.
        """
        if not self.is_allowed(sender_id):
            return
        
        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {}
        )
        
        await self.bus.publish_inbound(msg)
    
    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running
