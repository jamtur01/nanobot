"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        
        # Bootstrap file caching
        self._bootstrap_cache: str | None = None
        self._bootstrap_mtimes: dict[str, float] = {}
    
    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        query: str | None = None,
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
            query: The user's current message (used for relevance-based
                memory retrieval via FTS).
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context (FTS-powered when query is provided)
        memory = self.memory.get_memory_context(query=query)
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section, loading from IDENTITY.md if available."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        # Try to load identity from IDENTITY.md
        identity_file = self.workspace / "IDENTITY.md"
        if identity_file.exists():
            try:
                identity_content = identity_file.read_text(encoding="utf-8")
                return self._build_identity_with_context(
                    identity_content, now, runtime, workspace_path
                )
            except Exception as e:
                logger.warning(f"Failed to load IDENTITY.md, using defaults: {e}")
        
        # Fallback to hardcoded defaults
        return self._get_default_identity(now, runtime, workspace_path)
    
    def _build_identity_with_context(
        self, identity_content: str, now: str, runtime: str, workspace_path: str
    ) -> str:
        """Build identity from IDENTITY.md content with dynamic context appended."""
        return f"""{identity_content}

## Current Context

**Time**: {now}
**Runtime**: {runtime}
**Workspace**: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{self._get_behavioural_notes(workspace_path)}"""
    
    def _get_default_identity(self, now: str, runtime: str, workspace_path: str) -> str:
        """Get the hardcoded default identity (fallback when IDENTITY.md doesn't exist)."""
        return f"""# nanobot

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now}

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{self._get_behavioural_notes(workspace_path)}"""
    
    def _get_behavioural_notes(self, workspace_path: str) -> str:
        """Get behavioural instructions appended to both custom and default identity."""
        return f"""IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, explain what you're doing.
When remembering something, write to {workspace_path}/memory/MEMORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace, with mtime-based caching."""
        # Check if any file has been modified since last cache
        current_mtimes: dict[str, float] = {}
        for filename in self.BOOTSTRAP_FILES:
            # Skip IDENTITY.md - it's handled separately in _get_identity
            if filename == "IDENTITY.md":
                continue
            file_path = self.workspace / filename
            if file_path.exists():
                current_mtimes[filename] = file_path.stat().st_mtime
        
        # Return cached content if nothing changed
        if self._bootstrap_cache is not None and current_mtimes == self._bootstrap_mtimes:
            return self._bootstrap_cache
        
        # Rebuild from disk
        parts = []
        for filename in self.BOOTSTRAP_FILES:
            if filename == "IDENTITY.md":
                continue
            file_path = self.workspace / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    parts.append(f"## {filename}\n\n{content}")
                except Exception as e:
                    logger.warning(f"Failed to read bootstrap file {filename}: {e}")
        
        self._bootstrap_cache = "\n\n".join(parts) if parts else ""
        self._bootstrap_mtimes = current_mtimes
        return self._bootstrap_cache
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt (pass current_message for FTS-based memory retrieval)
        system_prompt = self.build_system_prompt(skill_names, query=current_message)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Optional reasoning content (for thinking models).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        
        messages.append(msg)
        return messages
