"""App-level authentication via streamlit-authenticator.

Credentials and cookie settings are read from Streamlit Secrets:

    [credentials.usernames.<username>]
    email, first_name, last_name, password (bcrypt hash or plain text if auto_hash=True)

    [cookie]
    name, key, expiry_days
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError

_SESSION_AUTHENTICATOR = "authenticator"


def _to_plain_dict(value: Any) -> Any:
    """Recursively convert Streamlit SecretDict / AttrDict to plain Python types."""
    if isinstance(value, Mapping):
        return {str(k): _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def _secrets_section(section: str) -> Dict[str, Any]:
    if section not in st.secrets:
        raise KeyError(
            f"Missing [{section}] in Streamlit Secrets. "
            "See .streamlit/secrets.toml.example for the required shape."
        )
    return _to_plain_dict(st.secrets[section])


def _load_auth_config() -> Dict[str, Any]:
    credentials = _secrets_section("credentials")
    cookie = _secrets_section("cookie")

    for field in ("name", "key", "expiry_days"):
        if field not in cookie:
            raise KeyError(f"Missing cookie.{field} in Streamlit Secrets.")

    usernames = credentials.get("usernames") or {}
    if not usernames:
        raise KeyError(
            "Missing credentials.usernames in Streamlit Secrets. "
            "Add at least one named user."
        )

    for username, user in usernames.items():
        if not isinstance(user, dict) or not user.get("password"):
            raise KeyError(
                f"User '{username}' is missing a password in Streamlit Secrets. "
                'Use password = "$2b$12$..." with double quotes around the bcrypt hash.'
            )

    return {"credentials": credentials, "cookie": cookie}


def get_authenticator() -> stauth.Authenticate:
    """Return a cached Authenticate instance (one per Streamlit session)."""
    if _SESSION_AUTHENTICATOR not in st.session_state:
        config = _load_auth_config()
        cookie = config["cookie"]
        st.session_state[_SESSION_AUTHENTICATOR] = stauth.Authenticate(
            credentials=config["credentials"],
            cookie_name=str(cookie["name"]),
            cookie_key=str(cookie["key"]),
            cookie_expiry_days=float(cookie["expiry_days"]),
            auto_hash=True,
        )
    return st.session_state[_SESSION_AUTHENTICATOR]


def get_display_name() -> str:
    """Human-readable name for the signed-in user."""
    return str(st.session_state.get("name") or st.session_state.get("username") or "")


def render_logout() -> None:
    """Sidebar logout control for authenticated users."""
    if not st.session_state.get("authentication_status"):
        return
    get_authenticator().logout(
        button_name="Logout",
        location="sidebar",
        use_container_width=True,
        key="doc_helper_logout",
    )


def require_auth() -> str:
    """Render login UI and return the authenticated username.

    Calls ``st.stop()`` when the user is not authenticated.
    """
    st.session_state.setdefault("logout", None)

    try:
        authenticator = get_authenticator()
    except KeyError as exc:
        st.error(f"Authentication configuration error: {exc}")
        st.stop()

    try:
        authenticator.login(
            location="main",
            max_login_attempts=5,
        )
    except LoginError as exc:
        st.error(str(exc))
        st.caption("Clear your browser cookies for this site or reboot the app, then try again.")
        st.stop()
    except Exception as exc:
        st.error("Authentication is temporarily unavailable. Please try again later.")
        st.caption(f"Details: {exc}")
        st.stop()

    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if auth_status is True and username:
        return str(username)

    if auth_status is False:
        st.error("Username or password is incorrect.")
    else:
        st.info("Please log in to use the documentation helper.")

    st.stop()