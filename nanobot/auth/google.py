"""Google OAuth2 authentication and credential management.

Handles the OAuth2 flow for Google APIs (Gmail, Calendar, etc.),
stores refresh tokens, and provides auto-refreshing credentials.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

# Default scopes for Gmail + Calendar
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# Where tokens are persisted
DEFAULT_TOKEN_PATH = Path.home() / ".nanobot" / "google_tokens.json"


def get_credentials(
    client_id: str,
    client_secret: str,
    scopes: list[str] | None = None,
    token_path: Path | None = None,
) -> Any:
    """
    Get valid Google OAuth2 credentials, refreshing if needed.

    Args:
        client_id: Google OAuth2 client ID.
        client_secret: Google OAuth2 client secret.
        scopes: OAuth2 scopes to request. Defaults to Gmail + Calendar.
        token_path: Path to stored token file.

    Returns:
        google.oauth2.credentials.Credentials object.

    Raises:
        RuntimeError: If no valid credentials and can't refresh.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    scopes = scopes or DEFAULT_SCOPES
    token_path = token_path or DEFAULT_TOKEN_PATH

    creds = _load_credentials(token_path, scopes)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds, token_path)
            logger.debug("Google credentials refreshed successfully")
            return creds
        except Exception as e:
            logger.warning(f"Failed to refresh Google credentials: {e}")

    raise RuntimeError(
        "No valid Google credentials. Run 'nanobot auth google' to authenticate."
    )


def run_oauth_flow(
    client_id: str,
    client_secret: str,
    scopes: list[str] | None = None,
    token_path: Path | None = None,
    port: int = 8099,
    headless: bool = False,
) -> Any:
    """
    Run the full OAuth2 authorization flow.

    When *headless* is False (default), tries to open a browser and run a
    local redirect server.  When True (or when no browser is available),
    falls back to a manual copy-paste flow that works over SSH / headless
    servers.

    Args:
        client_id: Google OAuth2 client ID.
        client_secret: Google OAuth2 client secret.
        scopes: OAuth2 scopes to request.
        token_path: Path to save the resulting token.
        port: Local port for the OAuth2 redirect server.
        headless: Force the manual copy-paste flow.

    Returns:
        google.oauth2.credentials.Credentials object.
    """
    scopes = scopes or DEFAULT_SCOPES
    token_path = token_path or DEFAULT_TOKEN_PATH

    # Build client config used by both flows
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    if headless:
        creds = _run_manual_flow(client_config, scopes)
    else:
        try:
            creds = _run_browser_flow(client_config, scopes, port)
        except Exception as e:
            # Browser unavailable — fall back to manual flow automatically
            logger.info(f"Browser flow failed ({e}), falling back to manual flow")
            creds = _run_manual_flow(client_config, scopes)

    # Persist
    _save_credentials(creds, token_path)
    logger.info(f"Google credentials saved to {token_path}")

    return creds


def _run_browser_flow(client_config: dict, scopes: list[str], port: int) -> Any:
    """Run the local-server browser flow."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_config["installed"]["redirect_uris"] = [f"http://localhost:{port}"]
    flow = InstalledAppFlow.from_client_config(client_config, scopes=scopes)
    return flow.run_local_server(port=port, open_browser=True)


def _run_manual_flow(client_config: dict, scopes: list[str]) -> Any:
    """
    Manual copy-paste flow for headless / SSH environments.

    Prints an authorization URL for the user to visit in any browser,
    then prompts them to paste back the authorization code.
    """
    from urllib.parse import urlencode, urlparse, parse_qs
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    import httpx

    installed = client_config["installed"]
    redirect_uri = "http://localhost"

    # Build authorization URL
    params = {
        "client_id": installed["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{installed['auth_uri']}?{urlencode(params)}"

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  Open this URL in any browser and authorize access: │")
    print("└─────────────────────────────────────────────────────┘\n")
    print(auth_url)
    print(
        "\nAfter authorizing, you will be redirected to a localhost URL.\n"
        "It may show an error page — that's OK.\n"
        "Copy the FULL URL from your browser's address bar and paste it below.\n"
        "(It will look like: http://localhost/?code=4/0A...&scope=...)\n"
    )

    response_input = input("Paste the redirect URL (or just the code): ").strip()

    # Extract the authorization code
    if response_input.startswith("http"):
        parsed = urlparse(response_input)
        qs = parse_qs(parsed.query)
        code = qs.get("code", [None])[0]
        if not code:
            raise RuntimeError("Could not find 'code' parameter in the URL you pasted.")
    else:
        code = response_input

    # Exchange authorization code for tokens
    token_response = httpx.post(
        installed["token_uri"],
        data={
            "code": code,
            "client_id": installed["client_id"],
            "client_secret": installed["client_secret"],
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30.0,
    )
    token_response.raise_for_status()
    token_data = token_response.json()

    if "error" in token_data:
        raise RuntimeError(f"Token exchange failed: {token_data['error_description']}")

    creds = Credentials(
        token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri=installed["token_uri"],
        client_id=installed["client_id"],
        client_secret=installed["client_secret"],
        scopes=scopes,
    )
    return creds


def has_valid_credentials(
    client_id: str,
    client_secret: str,
    scopes: list[str] | None = None,
    token_path: Path | None = None,
) -> bool:
    """Check whether we have valid (or refreshable) credentials on disk."""
    try:
        get_credentials(client_id, client_secret, scopes, token_path)
        return True
    except (RuntimeError, Exception):
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_credentials(token_path: Path, scopes: list[str]) -> Any | None:
    """Load credentials from the token file."""
    if not token_path.exists():
        return None

    try:
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        return creds
    except Exception as e:
        logger.debug(f"Could not load Google credentials from {token_path}: {e}")
        return None


def _save_credentials(creds: Any, token_path: Path) -> None:
    """Persist credentials to the token file."""
    import os

    token_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else [],
    }
    token_path.write_text(json.dumps(data, indent=2))

    # Restrict permissions: token file contains secrets
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        pass
