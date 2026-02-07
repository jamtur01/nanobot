"""Agent loop: the core processing engine."""

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.truncation import truncate_tool_result

# Empty LLM response retry config
EMPTY_RESPONSE_RETRIES = 2

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        google_config: "GoogleConfig | None" = None,
        compaction_config: "CompactionConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, GoogleConfig, CompactionConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.google_config = google_config
        self.compaction_config = compaction_config or CompactionConfig()
        
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        # Compaction / extraction
        self._compactor = None
        if self.compaction_config.enabled:
            from nanobot.agent.compaction import MessageCompactor
            self._compactor = MessageCompactor(
                provider=provider,
                model=self.compaction_config.model,
            )

        # Serializes daemon execution with user message processing
        self._processing_lock = asyncio.Lock()

        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace + home if configured)
        allowed_roots = (
            [self.workspace, Path.home()]
            if self.restrict_to_workspace
            else None
        )
        self.tools.register(ReadFileTool(allowed_roots=allowed_roots))
        self.tools.register(WriteFileTool(allowed_roots=allowed_roots))
        self.tools.register(EditFileTool(allowed_roots=allowed_roots))
        self.tools.register(ListDirTool(allowed_roots=allowed_roots))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Tmux tool (persistent shell sessions)
        from nanobot.agent.tools.tmux import TmuxTool
        self.tools.register(TmuxTool())

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Google tools (Gmail, Calendar)
        self._register_google_tools()
    
    def _register_google_tools(self) -> None:
        """Register Google tools if configured and credentials are available."""
        if not self.google_config or not self.google_config.enabled:
            return
        if not self.google_config.client_id or not self.google_config.client_secret:
            logger.warning("Google enabled but client_id/client_secret not set — skipping")
            return

        try:
            from nanobot.auth.google import get_credentials
            from nanobot.agent.tools.google_mail import GoogleMailTool
            from nanobot.agent.tools.google_calendar import GoogleCalendarTool

            creds = get_credentials(
                client_id=self.google_config.client_id,
                client_secret=self.google_config.client_secret,
                scopes=self.google_config.scopes,
            )
            self.tools.register(GoogleMailTool(creds))
            self.tools.register(GoogleCalendarTool(creds))
            logger.info("Google tools registered (gmail, calendar)")
        except RuntimeError as e:
            logger.warning(f"Google tools not available: {e}")
        except ImportError as e:
            logger.warning(
                f"Google dependencies not installed. "
                f"Install with: pip install nanobot-ai[google]  ({e})"
            )

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Acquires processing lock to serialize with daemon execution.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        async with self._processing_lock:
            return await self._process_message_unlocked(msg)

    def _set_tool_contexts(self, channel: str, chat_id: str) -> None:
        """Update channel/chat context on context-aware tools."""
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(channel, chat_id)
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(channel, chat_id)
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, messages: list[dict[str, Any]]) -> str | None:
        """Run the LLM tool-call loop. Returns final text content or None."""
        empty_retries_left = EMPTY_RESPONSE_RETRIES
        # Track whether the last iteration used a delivery tool (message/spawn)
        # so we can treat a subsequent empty response as intentional completion.
        last_used_delivery_tool = False

        for iteration in range(1, self.max_iterations + 1):
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )
            if response.has_tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                last_used_delivery_tool = any(
                    n in ("message", "spawn") for n in tool_names
                )
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                for tool_call in response.tool_calls:
                    logger.debug(
                        f"Executing tool: {tool_call.name} "
                        f"with arguments: {json.dumps(tool_call.arguments)}"
                    )
                    raw_result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    result = truncate_tool_result(raw_result)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            elif response.content:
                return response.content
            else:
                # LLM returned empty.  If the last iteration already
                # delivered a message to the user, that's a valid
                # completion — no need to retry or send a fallback.
                if last_used_delivery_tool:
                    logger.debug(
                        "LLM returned empty after delivery tool — "
                        "treating as successful completion"
                    )
                    return None
                if empty_retries_left > 0:
                    empty_retries_left -= 1
                    retry_num = EMPTY_RESPONSE_RETRIES - empty_retries_left
                    delay = min(2 ** retry_num + random.uniform(0, 1), 10.0)
                    logger.warning(
                        f"LLM returned empty on iteration {iteration}, "
                        f"retries left: {empty_retries_left}, backing off {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("LLM returned empty, no retries left - giving up")
                break
        return None

    async def _process_message_unlocked(self, msg: InboundMessage) -> OutboundMessage | None:
        """Inner message processing (called under lock)."""
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")
        
        session = self.sessions.get_or_create(msg.session_key)
        self._set_tool_contexts(msg.channel, msg.chat_id)
        
        history = await self._get_compacted_history(session)
        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        
        final_content = await self._run_agent_loop(messages)
        
        # Save conversation history regardless of whether we got a text reply
        assistant_text = final_content or ""
        session.add_message("user", msg.content)
        if assistant_text:
            session.add_message("assistant", assistant_text)
        self.sessions.save(session)
        
        # Background fact extraction (fire and forget)
        if (
            self._compactor
            and self.compaction_config.extraction_enabled
            and msg.channel != "system"
            and assistant_text
        ):
            asyncio.create_task(
                self._extract_and_save_facts(msg.content, assistant_text)
            )
        
        # If the agent already delivered via the message tool, don't
        # send an additional outbound message (would be a duplicate).
        if final_content is None:
            return None
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content
        )
    
    async def _get_compacted_history(
        self, session: "Session"
    ) -> list[dict[str, Any]]:
        """Get session history, compacting older messages if over threshold."""
        from nanobot.agent.compaction import estimate_messages_tokens

        if not self._compactor or not self.compaction_config.enabled:
            return session.get_history()

        full_history = session.get_full_history()
        if not full_history:
            return []

        keep_recent = self.compaction_config.keep_recent
        threshold = self.compaction_config.threshold_tokens
        token_est = estimate_messages_tokens(full_history)

        # Under threshold - return as-is (with normal truncation)
        if token_est < threshold:
            return session.get_history()

        # Split into old and recent
        if len(full_history) <= keep_recent:
            return full_history

        old_messages = full_history[:-keep_recent]
        recent_messages = full_history[-keep_recent:]

        # Get existing summary if any
        prev_summary = session.metadata.get("compaction_summary", "")

        # Check if we've already compacted up to this point
        compacted_up_to = session.metadata.get("compacted_up_to", 0)
        if compacted_up_to >= len(old_messages):
            # Already compacted, just prepend existing summary
            if prev_summary:
                summary_msg = {
                    "role": "assistant",
                    "content": f"[Earlier conversation summary]\n{prev_summary}",
                }
                return [summary_msg] + recent_messages
            return recent_messages

        # Run compaction
        logger.info(
            f"Compacting {len(old_messages)} messages "
            f"(~{estimate_messages_tokens(old_messages)} tokens)"
        )
        summary = await self._compactor.compact(old_messages, prev_summary)

        # Store in session metadata
        session.metadata["compaction_summary"] = summary
        session.metadata["compacted_up_to"] = len(old_messages)
        self.sessions.save(session)

        summary_msg = {
            "role": "assistant",
            "content": f"[Earlier conversation summary]\n{summary}",
        }
        return [summary_msg] + recent_messages

    async def _extract_and_save_facts(
        self, user_message: str, assistant_message: str
    ) -> None:
        """Extract facts from an exchange and save to today's daily note."""
        try:
            assert self._compactor is not None
            facts = await self._compactor.extract_facts(user_message, assistant_message)
            if facts:
                from nanobot.agent.memory import MemoryStore
                memory = MemoryStore(self.workspace)
                memory.append_today(f"\n### Extracted Facts\n{facts}\n")
                logger.debug(f"Extracted facts saved to daily note")
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        self._set_tool_contexts(origin_channel, origin_chat_id)
        
        # Build messages (with compaction)
        history = await self._get_compacted_history(session)
        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        final_content = await self._run_agent_loop(messages)
        
        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
