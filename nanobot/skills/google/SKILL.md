---
name: google
description: "Interact with Gmail and Google Calendar. Search/read/send emails, list/create/update calendar events."
metadata: {"nanobot":{"emoji":"📧","always":true,"requires":{"env":["NANOBOT_GOOGLE_ENABLED"]}}}
---

# Google Skill (Gmail & Calendar)

You have two tools for interacting with Google services: `google_mail` and `google_calendar`.

## Gmail — `google_mail`

### Search emails

```json
{"action": "search", "query": "is:unread from:alice", "max_results": 5}
```

Common Gmail search operators:
- `is:unread` — unread messages
- `from:user@example.com` — from a specific sender
- `to:me` — sent to you
- `subject:meeting` — subject contains "meeting"
- `has:attachment` — messages with attachments
- `after:2026/01/01 before:2026/02/01` — date range
- `label:important` — by label
- `in:inbox` / `in:sent` / `in:trash`

### Read a specific email

```json
{"action": "read", "message_id": "18d...abc"}
```

### Send a new email

```json
{"action": "send", "to": "alice@example.com", "subject": "Hello", "body": "Hi Alice!"}
```

### Reply to an email

```json
{"action": "reply", "message_id": "18d...abc", "body": "Thanks for the update!"}
```

### List labels

```json
{"action": "list_labels"}
```

### Add/remove labels

```json
{"action": "label", "message_id": "18d...abc", "label_ids": ["STARRED"], "remove_label_ids": ["UNREAD"]}
```

## Google Calendar — `google_calendar`

### List upcoming events

```json
{"action": "list_events"}
```

With custom range:

```json
{"action": "list_events", "time_min": "2026-02-05T00:00:00Z", "time_max": "2026-02-12T00:00:00Z", "max_results": 10}
```

### Get event details

```json
{"action": "get_event", "event_id": "abc123xyz"}
```

### Create an event

```json
{
  "action": "create_event",
  "summary": "Team standup",
  "start": "2026-02-06T10:00:00",
  "end": "2026-02-06T10:30:00",
  "timezone": "America/New_York",
  "description": "Daily sync",
  "attendees": ["alice@example.com", "bob@example.com"]
}
```

All-day event:

```json
{"action": "create_event", "summary": "Holiday", "start": "2026-02-14", "all_day": true}
```

### Update an event

```json
{"action": "update_event", "event_id": "abc123xyz", "summary": "Renamed standup", "start": "2026-02-06T11:00:00"}
```

### Delete an event

```json
{"action": "delete_event", "event_id": "abc123xyz"}
```

### List all calendars

```json
{"action": "list_calendars"}
```

Use `calendar_id` to target a specific calendar (defaults to `"primary"`).

## Tips

- Always search before sending to find context / thread IDs.
- When the user asks "check my email", use `{"action": "search", "query": "is:unread", "max_results": 10}`.
- When the user asks "what's on my calendar", use `{"action": "list_events"}`.
- Use `reply` instead of `send` when responding to an existing thread.
- For calendar events, always include a timezone when the user specifies one.
