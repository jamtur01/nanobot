"""Google Calendar tool for listing, creating, and managing calendar events."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class GoogleCalendarTool(Tool):
    """List, create, update, and delete Google Calendar events."""

    name = "google_calendar"
    description = (
        "Interact with Google Calendar. "
        "Actions: list_events, get_event, create_event, update_event, delete_event, list_calendars."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_events",
                    "get_event",
                    "create_event",
                    "update_event",
                    "delete_event",
                    "list_calendars",
                ],
                "description": "Action to perform",
            },
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID (default: 'primary')",
            },
            "event_id": {
                "type": "string",
                "description": "Event ID (for get/update/delete)",
            },
            "time_min": {
                "type": "string",
                "description": "Start of time range (ISO 8601, for list_events). Defaults to now.",
            },
            "time_max": {
                "type": "string",
                "description": "End of time range (ISO 8601, for list_events). Defaults to 7 days from now.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max events to return (for list_events, default 20)",
                "minimum": 1,
                "maximum": 100,
            },
            "summary": {
                "type": "string",
                "description": "Event title (for create/update)",
            },
            "description": {
                "type": "string",
                "description": "Event description (for create/update)",
            },
            "location": {
                "type": "string",
                "description": "Event location (for create/update)",
            },
            "start": {
                "type": "string",
                "description": "Event start time in ISO 8601 (for create/update). e.g. '2026-02-06T10:00:00'",
            },
            "end": {
                "type": "string",
                "description": "Event end time in ISO 8601 (for create/update). e.g. '2026-02-06T11:00:00'",
            },
            "timezone": {
                "type": "string",
                "description": "Timezone for the event (e.g. 'America/New_York'). Defaults to UTC.",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee email addresses (for create/update)",
            },
            "all_day": {
                "type": "boolean",
                "description": "If true, create an all-day event (use date instead of dateTime)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, credentials: Any):
        self._creds = credentials
        self._service = None

    def _get_service(self) -> Any:
        if self._service is None:
            from googleapiclient.discovery import build

            self._service = build("calendar", "v3", credentials=self._creds)
        return self._service

    async def execute(self, action: str, **kwargs: Any) -> str:
        """Dispatch to synchronous Calendar helpers via a thread executor."""
        try:
            loop = asyncio.get_running_loop()
            cal_id = kwargs.get("calendar_id", "primary") or "primary"
            if action == "list_events":
                fn = partial(self._list_events, cal_id, kwargs)
            elif action == "get_event":
                fn = partial(self._get_event, cal_id, kwargs.get("event_id", ""))
            elif action == "create_event":
                fn = partial(self._create_event, cal_id, kwargs)
            elif action == "update_event":
                fn = partial(self._update_event, cal_id, kwargs)
            elif action == "delete_event":
                fn = partial(self._delete_event, cal_id, kwargs.get("event_id", ""))
            elif action == "list_calendars":
                fn = self._list_calendars
            else:
                return f"Unknown action: {action}"
            return await loop.run_in_executor(None, fn)
        except Exception as e:
            logger.error(f"GoogleCalendarTool error: {e}")
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _list_events(self, cal_id: str, kwargs: dict) -> str:
        now = datetime.now(timezone.utc)
        time_min = kwargs.get("time_min") or now.isoformat()
        time_max = kwargs.get("time_max") or (now + timedelta(days=7)).isoformat()
        max_results = kwargs.get("max_results", 20)

        svc = self._get_service()
        result = (
            svc.events()
            .list(
                calendarId=cal_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = result.get("items", [])
        if not events:
            return "No upcoming events found."

        lines = [f"Found {len(events)} event(s):\n"]
        for ev in events:
            start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date", "?")
            end = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date", "?")
            summary = ev.get("summary", "(no title)")
            location = ev.get("location", "")
            loc_str = f"\n  Location: {location}" if location else ""
            attendees = ev.get("attendees", [])
            att_str = ""
            if attendees:
                names = [a.get("email", "?") for a in attendees[:5]]
                att_str = f"\n  Attendees: {', '.join(names)}"
                if len(attendees) > 5:
                    att_str += f" (+{len(attendees) - 5} more)"

            lines.append(
                f"- {summary}\n"
                f"  ID: {ev['id']}\n"
                f"  Start: {start}\n"
                f"  End: {end}"
                f"{loc_str}{att_str}"
            )
        return "\n".join(lines)

    def _get_event(self, cal_id: str, event_id: str) -> str:
        if not event_id:
            return "Error: 'event_id' is required for get_event"
        svc = self._get_service()
        ev = svc.events().get(calendarId=cal_id, eventId=event_id).execute()
        return self._format_event_detail(ev)

    def _create_event(self, cal_id: str, kwargs: dict) -> str:
        summary = kwargs.get("summary", "")
        if not summary:
            return "Error: 'summary' is required for create_event"
        start = kwargs.get("start", "")
        if not start:
            return "Error: 'start' is required for create_event"

        tz = kwargs.get("timezone", "UTC") or "UTC"
        all_day = kwargs.get("all_day", False)

        body: dict[str, Any] = {"summary": summary}

        if kwargs.get("description"):
            body["description"] = kwargs["description"]
        if kwargs.get("location"):
            body["location"] = kwargs["location"]

        if all_day:
            # All-day events use 'date' (YYYY-MM-DD)
            body["start"] = {"date": start[:10]}
            end = kwargs.get("end") or start
            body["end"] = {"date": end[:10]}
        else:
            body["start"] = {"dateTime": start, "timeZone": tz}
            end = kwargs.get("end")
            if not end:
                # Default to 1 hour
                try:
                    dt = datetime.fromisoformat(start)
                    end = (dt + timedelta(hours=1)).isoformat()
                except ValueError:
                    return "Error: invalid 'start' format (use ISO 8601)"
            body["end"] = {"dateTime": end, "timeZone": tz}

        if kwargs.get("attendees"):
            body["attendees"] = [{"email": e} for e in kwargs["attendees"]]

        svc = self._get_service()
        ev = svc.events().insert(calendarId=cal_id, body=body).execute()
        return f"Event created: {ev.get('summary')} (ID: {ev['id']})\nLink: {ev.get('htmlLink', '')}"

    def _update_event(self, cal_id: str, kwargs: dict) -> str:
        event_id = kwargs.get("event_id", "")
        if not event_id:
            return "Error: 'event_id' is required for update_event"

        svc = self._get_service()
        ev = svc.events().get(calendarId=cal_id, eventId=event_id).execute()

        tz = kwargs.get("timezone") or "UTC"

        if kwargs.get("summary"):
            ev["summary"] = kwargs["summary"]
        if kwargs.get("description"):
            ev["description"] = kwargs["description"]
        if kwargs.get("location"):
            ev["location"] = kwargs["location"]
        if kwargs.get("start"):
            if kwargs.get("all_day"):
                ev["start"] = {"date": kwargs["start"][:10]}
            else:
                ev["start"] = {"dateTime": kwargs["start"], "timeZone": tz}
        if kwargs.get("end"):
            if kwargs.get("all_day"):
                ev["end"] = {"date": kwargs["end"][:10]}
            else:
                ev["end"] = {"dateTime": kwargs["end"], "timeZone": tz}
        if kwargs.get("attendees"):
            ev["attendees"] = [{"email": e} for e in kwargs["attendees"]]

        updated = svc.events().update(calendarId=cal_id, eventId=event_id, body=ev).execute()
        return f"Event updated: {updated.get('summary')} (ID: {updated['id']})"

    def _delete_event(self, cal_id: str, event_id: str) -> str:
        if not event_id:
            return "Error: 'event_id' is required for delete_event"
        svc = self._get_service()
        svc.events().delete(calendarId=cal_id, eventId=event_id).execute()
        return f"Event {event_id} deleted."

    def _list_calendars(self) -> str:
        svc = self._get_service()
        result = svc.calendarList().list().execute()
        cals = result.get("items", [])
        if not cals:
            return "No calendars found."
        lines = ["Your calendars:\n"]
        for cal in cals:
            primary = " (primary)" if cal.get("primary") else ""
            lines.append(f"- {cal.get('summary', '?')}{primary}\n  ID: {cal['id']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_event_detail(self, ev: dict) -> str:
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date", "?")
        end = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date", "?")
        parts = [
            f"Summary: {ev.get('summary', '(no title)')}",
            f"ID: {ev.get('id', '')}",
            f"Start: {start}",
            f"End: {end}",
        ]
        if ev.get("location"):
            parts.append(f"Location: {ev['location']}")
        if ev.get("description"):
            parts.append(f"Description: {ev['description']}")
        if ev.get("attendees"):
            att = [a.get("email", "?") for a in ev["attendees"]]
            parts.append(f"Attendees: {', '.join(att)}")
        if ev.get("htmlLink"):
            parts.append(f"Link: {ev['htmlLink']}")
        if ev.get("status"):
            parts.append(f"Status: {ev['status']}")
        return "\n".join(parts)
