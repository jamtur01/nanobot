"""Google Mail tool for reading, searching, and sending email via Gmail API."""

import asyncio
import base64
import json
from email.mime.text import MIMEText
from functools import partial
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class GoogleMailTool(Tool):
    """Search, read, send, reply, and label Gmail messages."""

    name = "google_mail"
    description = (
        "Interact with Gmail. Actions: search, read, send, reply, list_labels, label."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "read", "send", "reply", "list_labels", "label"],
                "description": "Action to perform",
            },
            "query": {
                "type": "string",
                "description": "Gmail search query (for search). e.g. 'is:unread from:alice'",
            },
            "message_id": {
                "type": "string",
                "description": "Message ID (for read / reply / label)",
            },
            "to": {
                "type": "string",
                "description": "Recipient email address (for send)",
            },
            "subject": {
                "type": "string",
                "description": "Email subject (for send)",
            },
            "body": {
                "type": "string",
                "description": "Email body text (for send / reply)",
            },
            "label_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Label IDs to add (for label)",
            },
            "remove_label_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Label IDs to remove (for label)",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (for search, default 10)",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["action"],
    }

    def __init__(self, credentials: Any):
        """
        Args:
            credentials: google.oauth2.credentials.Credentials object.
        """
        self._creds = credentials
        self._service = None

    def _get_service(self) -> Any:
        if self._service is None:
            from googleapiclient.discovery import build

            self._service = build("gmail", "v1", credentials=self._creds)
        return self._service

    async def execute(self, action: str, **kwargs: Any) -> str:
        """Dispatch to the synchronous Gmail helpers via a thread executor."""
        try:
            loop = asyncio.get_running_loop()
            if action == "search":
                fn = partial(self._search, kwargs.get("query", ""), kwargs.get("max_results", 10))
            elif action == "read":
                fn = partial(self._read, kwargs.get("message_id", ""))
            elif action == "send":
                fn = partial(self._send, kwargs.get("to", ""), kwargs.get("subject", ""), kwargs.get("body", ""))
            elif action == "reply":
                fn = partial(self._reply, kwargs.get("message_id", ""), kwargs.get("body", ""))
            elif action == "list_labels":
                fn = self._list_labels
            elif action == "label":
                fn = partial(
                    self._label,
                    kwargs.get("message_id", ""),
                    kwargs.get("label_ids", []),
                    kwargs.get("remove_label_ids", []),
                )
            else:
                return f"Unknown action: {action}"
            return await loop.run_in_executor(None, fn)
        except Exception as e:
            logger.error(f"GoogleMailTool error: {e}")
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int) -> str:
        if not query:
            return "Error: 'query' is required for search"
        svc = self._get_service()
        results = (
            svc.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        messages = results.get("messages", [])
        if not messages:
            return f"No messages found for query: {query}"

        lines = [f"Found {len(messages)} message(s) for: {query}\n"]
        for msg_stub in messages:
            msg = (
                svc.users()
                .messages()
                .get(userId="me", id=msg_stub["id"], format="metadata",
                     metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            snippet = msg.get("snippet", "")
            lines.append(
                f"- ID: {msg['id']}\n"
                f"  From: {headers.get('From', '?')}\n"
                f"  Subject: {headers.get('Subject', '(no subject)')}\n"
                f"  Date: {headers.get('Date', '?')}\n"
                f"  Snippet: {snippet}"
            )
        return "\n".join(lines)

    def _read(self, message_id: str) -> str:
        if not message_id:
            return "Error: 'message_id' is required for read"
        svc = self._get_service()
        msg = (
            svc.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body = self._extract_body(msg.get("payload", {}))
        labels = ", ".join(msg.get("labelIds", []))

        return (
            f"From: {headers.get('From', '?')}\n"
            f"To: {headers.get('To', '?')}\n"
            f"Subject: {headers.get('Subject', '(no subject)')}\n"
            f"Date: {headers.get('Date', '?')}\n"
            f"Labels: {labels}\n"
            f"---\n{body}"
        )

    def _send(self, to: str, subject: str, body: str) -> str:
        if not to:
            return "Error: 'to' is required for send"
        if not subject:
            return "Error: 'subject' is required for send"
        if not body:
            return "Error: 'body' is required for send"

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        svc = self._get_service()
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent successfully (ID: {sent['id']})"

    def _reply(self, message_id: str, body: str) -> str:
        if not message_id:
            return "Error: 'message_id' is required for reply"
        if not body:
            return "Error: 'body' is required for reply"

        svc = self._get_service()
        original = (
            svc.users()
            .messages()
            .get(userId="me", id=message_id, format="metadata",
                 metadataHeaders=["From", "Subject", "Message-ID"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}
        thread_id = original.get("threadId", "")

        reply_to = headers.get("From", "")
        subject = headers.get("Subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        message = MIMEText(body)
        message["to"] = reply_to
        message["subject"] = subject
        message["In-Reply-To"] = headers.get("Message-ID", "")
        message["References"] = headers.get("Message-ID", "")
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        sent = (
            svc.users()
            .messages()
            .send(userId="me", body={"raw": raw, "threadId": thread_id})
            .execute()
        )
        return f"Reply sent successfully (ID: {sent['id']})"

    def _list_labels(self) -> str:
        svc = self._get_service()
        results = svc.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        if not labels:
            return "No labels found."
        lines = ["Gmail Labels:\n"]
        for lbl in sorted(labels, key=lambda l: l.get("name", "")):
            lines.append(f"- {lbl['name']} (ID: {lbl['id']})")
        return "\n".join(lines)

    def _label(
        self,
        message_id: str,
        add_label_ids: list[str],
        remove_label_ids: list[str],
    ) -> str:
        if not message_id:
            return "Error: 'message_id' is required for label"
        if not add_label_ids and not remove_label_ids:
            return "Error: provide 'label_ids' and/or 'remove_label_ids'"

        body: dict[str, Any] = {}
        if add_label_ids:
            body["addLabelIds"] = add_label_ids
        if remove_label_ids:
            body["removeLabelIds"] = remove_label_ids

        svc = self._get_service()
        svc.users().messages().modify(userId="me", id=message_id, body=body).execute()
        return f"Labels updated on message {message_id}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_body(self, payload: dict) -> str:
        """Extract plain-text body from a Gmail message payload."""
        # Direct body
        if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode(errors="replace")

        # Multipart — look for text/plain first, then text/html
        parts = payload.get("parts", [])
        for mime in ("text/plain", "text/html"):
            for part in parts:
                if part.get("mimeType") == mime and part.get("body", {}).get("data"):
                    text = base64.urlsafe_b64decode(part["body"]["data"]).decode(errors="replace")
                    if mime == "text/html":
                        # Quick strip tags for readability
                        import re
                        text = re.sub(r"<[^>]+>", "", text)
                    return text
                # Nested multipart
                if part.get("parts"):
                    nested = self._extract_body(part)
                    if nested:
                        return nested

        return "(no readable body)"
