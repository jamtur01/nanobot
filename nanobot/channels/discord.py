"""Discord channel implementation using discord.py."""

import asyncio
import re
from pathlib import Path

import discord
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig

# Prevent the bot from accidentally pinging @everyone, @here, roles, or users
_SAFE_MENTIONS = discord.AllowedMentions.none()


def _markdown_to_discord(text: str) -> str:
    """Clean up markdown for Discord.

    Discord natively supports most markdown, so this is light-touch.
    Length is handled by _split_message, not here.
    """
    return text or ""


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
        # Map channel/DM id -> last incoming message (for threaded replies)
        self._pending_replies: dict[str, discord.Message] = {}
        # Typing context tasks per channel (for cancellation)
        self._typing_tasks: dict[str, asyncio.Task] = {}

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

        self._client = discord.Client(
            intents=intents,
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="messages",
            ),
        )

        @self._client.event
        async def on_ready() -> None:
            assert self._client is not None
            user = self._client.user
            logger.info(f"Discord bot connected as {user} (ID: {user.id})")  # type: ignore[union-attr]

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            await self._on_message(message)

        @self._client.event
        async def on_error(event: str, *args, **kwargs) -> None:  # type: ignore[override]
            logger.exception(f"Unhandled Discord error in event '{event}'")

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
        for task in list(self._typing_tasks.values()):
            task.cancel()
        self._typing_tasks.clear()
        self._pending_replies.clear()
        if self._client and not self._client.is_closed():
            logger.info("Stopping Discord bot...")
            await self._client.close()
            self._client = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord (discord.py handles rate limits internally)."""
        if not self._client:
            logger.warning("Discord bot not running")
            return

        try:
            target = await self._resolve_channel(msg.chat_id)
            if target is None:
                return

            content = _markdown_to_discord(msg.content)

            # Collect any file attachments from outbound media paths
            files = _build_files(msg.media) if msg.media else []

            chunks = self.split_message(content)
            reply_msg = self._pending_replies.pop(msg.chat_id, None)

            for i, chunk in enumerate(chunks):
                try:
                    # Reply to the original message for the first chunk
                    if i == 0 and reply_msg is not None:
                        await reply_msg.reply(
                            chunk,
                            allowed_mentions=_SAFE_MENTIONS,
                            files=files if i == 0 else [],
                        )
                    else:
                        await target.send(
                            chunk,
                            allowed_mentions=_SAFE_MENTIONS,
                            files=files if i == 0 else [],
                        )
                except Exception as e:
                    logger.error(f"Error sending Discord message: {e}")
        finally:
            self._stop_typing(msg.chat_id)

    async def _resolve_channel(self, chat_id: str) -> discord.abc.Messageable | None:
        """Resolve a chat_id to a messageable Discord channel."""
        if not self._client:
            return None

        channel_id = int(chat_id)

        # Try cache first
        target = self._channels.get(chat_id)
        if target is None:
            target = self._client.get_channel(channel_id)
        if target is None:
            # Might be a DM — fetch the user and create DM channel
            try:
                user = await self._client.fetch_user(channel_id)
                target = await user.create_dm()
            except Exception:
                logger.error(f"Could not find Discord channel or user for id {channel_id}")
                return None

        if not isinstance(target, discord.abc.Messageable):
            logger.error(f"Discord target {channel_id} is not messageable")
            return None

        return target

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

        # Cache the channel and original message for replies
        self._channels[chat_id] = message.channel
        self._pending_replies[chat_id] = message

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

        self._start_typing(chat_id, message.channel)

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

    def _start_typing(self, chat_id: str, channel: discord.abc.Messageable) -> None:
        """Start a typing indicator using discord.py's context manager."""
        self._stop_typing(chat_id)

        async def typing_wrapper() -> None:
            try:
                async with channel.typing():
                    # Block until cancelled — the context manager keeps typing active
                    await asyncio.sleep(600)  # 10 min upper bound
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._typing_tasks[chat_id] = asyncio.create_task(typing_wrapper())

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a channel."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()


def _ext_from_content_type(content_type: str) -> str:
    """Map content type to file extension."""
    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    return ext_map.get(content_type, ".bin")




def _build_files(media_paths: list[str]) -> list[discord.File]:
    """Build discord.File objects from local file paths."""
    files: list[discord.File] = []
    for path_str in media_paths:
        path = Path(path_str)
        if path.is_file():
            try:
                files.append(discord.File(path, filename=path.name))
            except Exception as e:
                logger.warning(f"Could not attach file {path}: {e}")
    return files
