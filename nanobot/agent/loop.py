"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

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
from nanobot.agent.subagent import SubagentManager, build_base_tools
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
        google_config: "GoogleConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, GoogleConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.google_config = google_config
        
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
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools.

        Uses ``build_base_tools`` for the shared baseline (file, shell, web)
        so that tool creation stays in sync between the main agent and
        subagents.  Then layers on agent-only tools (message, spawn, cron,
        edit_file, Google).
        """
        # Shared base tools (read_file, write_file, list_dir, exec, web)
        base = build_base_tools(
            workspace=self.workspace,
            exec_config=self.exec_config,
            brave_api_key=self.brave_api_key,
        )
        for tool_name in base.tool_names:
            tool = base.get(tool_name)
            if tool:
                self.tools.register(tool)

        # edit_file is main-agent only (subagents get read + write)
        from pathlib import Path
        allowed_roots = [self.workspace, Path.home() / ".nanobot"]
        self.tools.register(EditFileTool(allowed_roots=allowed_roots))
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
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
    
    # ------------------------------------------------------------------
    # Shared agent turn: LLM call + tool execution loop
    # ------------------------------------------------------------------

    async def _run_agent_turn(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Run the LLM + tool-call loop until the agent produces a final text response.

        Args:
            messages: The starting message list (system + history + user).

        Returns:
            (final_content, messages) — the assistant's text reply (or None
            if max iterations reached) and the updated message list.
        """
        iteration = 0
        final_content: str | None = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )

            if response.has_tool_calls:
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
                    messages, response.content, tool_call_dicts
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break

        return final_content, messages

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _update_tool_contexts(self, channel: str, chat_id: str) -> None:
        """Set the current channel/chat context on context-aware tools."""
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(channel, chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(channel, chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(channel, chat_id)

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")
        
        session = self.sessions.get_or_create(msg.session_key)
        self._update_tool_contexts(msg.channel, msg.chat_id)
        
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        
        final_content, _ = await self._run_agent_turn(messages)
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        if ":" in msg.chat_id:
            origin_channel, origin_chat_id = msg.chat_id.split(":", 1)
        else:
            origin_channel, origin_chat_id = "cli", msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._update_tool_contexts(origin_channel, origin_chat_id)
        
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        final_content, _ = await self._run_agent_turn(messages)
        
        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content,
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
