"""Slack channel implementation using slack-bolt with Socket Mode.

Socket Mode connects via WebSocket so no public URL or ngrok is needed —
works the same way as Telegram polling or Discord gateway.
"""

import asyncio
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import SlackConfig


class SlackChannel(BaseChannel):
    """
    Slack channel using Socket Mode (WebSocket, no public URL needed).

    Responds to:
    - Direct messages (DMs) to the bot
    - @mentions in channels the bot is a member of
    """

    name = "slack"

    def __init__(self, config: SlackConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: SlackConfig = config
        self._app: Any = None
        self._handler: Any = None
        self._bot_user_id: str = ""
        self._web_client: Any = None

    async def start(self) -> None:
        """Start the Slack bot with Socket Mode."""
        if not self.config.bot_token:
            logger.error("Slack bot_token not configured")
            return
        if not self.config.app_token:
            logger.error("Slack app_token not configured (needed for Socket Mode)")
            return

        self._running = True

        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        self._app = AsyncApp(token=self.config.bot_token)
        self._web_client = self._app.client

        # Fetch bot's own user ID so we can strip @mentions
        try:
            auth = await self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id", "")
            bot_name = auth.get("user", "unknown")
            logger.info(f"Slack bot connected as @{bot_name} (ID: {self._bot_user_id})")
        except Exception as e:
            logger.error(f"Slack auth_test failed: {e}")
            return

        # Register event listeners BEFORE starting the handler
        @self._app.event("message")
        async def on_message(body: dict, event: dict, say: Any, logger: Any) -> None:
            await self._on_message(event)

        @self._app.event("app_mention")
        async def on_mention(body: dict, event: dict, say: Any, logger: Any) -> None:
            await self._on_message(event)

        logger.info("Starting Slack bot (Socket Mode)...")
        self._handler = AsyncSocketModeHandler(self._app, self.config.app_token)

        # connect_async() opens the WebSocket properly without blocking
        await self._handler.connect_async()
        logger.info("Slack Socket Mode connected and listening for events")

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Slack bot."""
        self._running = False
        if self._handler:
            logger.info("Stopping Slack bot...")
            await self._handler.close_async()
            self._handler = None
            self._app = None
            self._web_client = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Slack."""
        if not self._web_client:
            logger.warning("Slack bot not running")
            return

        try:
            content = _format_for_slack(msg.content)

            # Split long messages (Slack limit is ~40k but 4000 is more readable)
            for chunk in _split_message(content, limit=4000):
                await self._web_client.chat_postMessage(
                    channel=msg.chat_id,
                    text=chunk,
                    mrkdwn=True,
                )
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")

    async def _on_message(self, event: dict) -> None:
        """Handle an incoming Slack message or app_mention event."""
        # Ignore bot messages and message_changed subtypes
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        text = event.get("text", "")

        if not user_id or not channel_id:
            return

        logger.info(f"Slack: received message from {user_id} in {channel_id}")

        # Strip the bot @mention from text
        if self._bot_user_id:
            text = re.sub(rf"<@{self._bot_user_id}>\s*", "", text).strip()

        if not text:
            return

        # Build sender_id — try to enrich with username
        sender_id = user_id
        if self._web_client:
            try:
                info = await self._web_client.users_info(user=user_id)
                username = info.get("user", {}).get("name", "")
                if username:
                    sender_id = f"{user_id}|{username}"
            except Exception:
                pass

        # Handle file attachments (images)
        media_paths: list[str] = []
        content_parts: list[str] = [text] if text else []

        for file_info in event.get("files", []):
            mimetype = file_info.get("mimetype", "")
            if mimetype.startswith("image/"):
                local_path = await self._download_file(file_info)
                if local_path:
                    media_paths.append(local_path)
                    content_parts.append(f"[image: {local_path}]")
            else:
                content_parts.append(f"[file: {file_info.get('name', 'unknown')}]")

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug(f"Slack message from {sender_id} in {channel_id}: {content[:50]}...")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            content=content,
            media=media_paths,
            metadata={
                "ts": event.get("ts", ""),
                "user_id": user_id,
                "channel_type": event.get("channel_type", ""),
            },
        )

    async def _download_file(self, file_info: dict) -> str | None:
        """Download a Slack file to local media directory."""
        if not self._web_client:
            return None

        url = file_info.get("url_private_download") or file_info.get("url_private")
        if not url:
            return None

        try:
            import httpx

            media_dir = Path.home() / ".nanobot" / "media"
            media_dir.mkdir(parents=True, exist_ok=True)

            ext = _ext_from_mimetype(file_info.get("mimetype", ""))
            file_path = media_dir / f"slack_{file_info.get('id', 'unknown')}{ext}"

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.config.bot_token}"},
                    timeout=30.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                file_path.write_bytes(resp.content)

            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to download Slack file: {e}")
            return None


def _format_for_slack(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn where they differ."""
    if not text:
        return ""
    # Slack uses *bold* not **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Slack uses _italic_ (same as standard markdown)
    # Slack uses ~strike~ not ~~strike~~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)
    # Slack links: [text](url) -> <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    return text


def _ext_from_mimetype(mimetype: str) -> str:
    """Map MIME type to file extension."""
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(mimetype, ".bin")


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """Split a message into chunks that fit Slack's limits."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
