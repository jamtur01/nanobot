"""Discord channel implementation using discord.py."""

import asyncio
import re

import discord
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig


def _markdown_to_discord(text: str) -> str:
    """
    Clean up markdown for Discord.

    Discord natively supports most markdown, so this is light-touch:
    just ensure code blocks and formatting pass through cleanly,
    and cap message length at Discord's 2000-char limit.
    """
    if not text:
        return ""
    # Discord has a 2000-char limit per message; truncate with notice
    if len(text) > 1990:
        text = text[:1990] + "\n…(truncated)"
    return text


class DiscordChannel(BaseChannel):
    """
    Discord channel using discord.py with gateway intents.

    Listens for DMs and messages in channels where the bot is mentioned
    or configured to respond.
    """

    name = "discord"

    def __init__(self, config: DiscordConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self._client: discord.Client | None = None
        # Map channel/DM id -> discord channel object for sending replies
        self._channels: dict[str, discord.abc.Messageable] = {}

    async def start(self) -> None:
        """Start the Discord bot."""
        if not self.config.token:
            logger.error("Discord bot token not configured")
            return

        self._running = True

        # We need message_content intent to read message text
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            assert self._client is not None
            user = self._client.user
            logger.info(f"Discord bot connected as {user} (ID: {user.id})")  # type: ignore[union-attr]

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            await self._on_message(message)

        logger.info("Starting Discord bot...")
        try:
            await self._client.start(self.config.token)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Discord bot error: {e}")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False
        if self._client and not self._client.is_closed():
            logger.info("Stopping Discord bot...")
            await self._client.close()
            self._client = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord."""
        if not self._client:
            logger.warning("Discord bot not running")
            return

        try:
            channel_id = int(msg.chat_id)

            # Try cache first
            target = self._channels.get(msg.chat_id)
            if target is None:
                target = self._client.get_channel(channel_id)
            if target is None:
                # Might be a DM — fetch the user and create DM channel
                try:
                    user = await self._client.fetch_user(channel_id)
                    target = await user.create_dm()
                except Exception:
                    logger.error(f"Could not find Discord channel or user for id {channel_id}")
                    return

            if not isinstance(target, discord.abc.Messageable):
                logger.error(f"Discord target {channel_id} is not messageable")
                return

            content = _markdown_to_discord(msg.content)

            # Discord has a 2000-char hard limit; split if needed
            for chunk in _split_message(content):
                await target.send(chunk)

        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")

    async def _on_message(self, message: discord.Message) -> None:
        """Handle an incoming Discord message."""
        # Ignore our own messages
        if self._client and message.author.id == self._client.user.id:  # type: ignore[union-attr]
            return

        # Determine if we should respond:
        # 1. Always respond to DMs
        # 2. In guild channels, respond only if bot is mentioned
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self._client is not None and self._client.user in message.mentions  # type: ignore[union-attr]

        if not is_dm and not is_mentioned:
            return

        # Build sender_id: "user_id|username"
        sender_id = str(message.author.id)
        if message.author.name:
            sender_id = f"{sender_id}|{message.author.name}"

        # Use the channel id as chat_id (works for both DM and guild channels)
        chat_id = str(message.channel.id)

        # Cache the channel for replies
        self._channels[chat_id] = message.channel

        # Build content
        content_parts: list[str] = []

        # Strip the bot mention from the text for cleaner input
        text = message.content or ""
        if is_mentioned and self._client and self._client.user:
            text = re.sub(rf"<@!?{self._client.user.id}>\s*", "", text).strip()

        if text:
            content_parts.append(text)

        # Handle attachments
        media_paths: list[str] = []
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                # Download image to local media dir
                try:
                    from pathlib import Path

                    media_dir = Path.home() / ".nanobot" / "media"
                    media_dir.mkdir(parents=True, exist_ok=True)
                    ext = _ext_from_content_type(attachment.content_type)
                    file_path = media_dir / f"discord_{attachment.id}{ext}"
                    await attachment.save(file_path)
                    media_paths.append(str(file_path))
                    content_parts.append(f"[image: {file_path}]")
                except Exception as e:
                    logger.error(f"Failed to download Discord attachment: {e}")
                    content_parts.append(f"[image: download failed]")
            else:
                content_parts.append(f"[attachment: {attachment.filename}]")

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug(f"Discord message from {sender_id}: {content[:50]}...")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.id,
                "user_id": message.author.id,
                "username": message.author.name,
                "guild_id": message.guild.id if message.guild else None,
                "is_dm": is_dm,
            },
        )


def _ext_from_content_type(content_type: str) -> str:
    """Map content type to file extension."""
    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    return ext_map.get(content_type, ".bin")


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a message into chunks that fit Discord's character limit."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
