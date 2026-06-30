"""App-level authentication via streamlit-authenticator.

Credentials and cookie settings are read from Streamlit Secrets:

    [credentials.usernames.<username>]
    email, first_name, last_name, password (bcrypt hash)

    [cookie]
    name, key, expiry_days
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import streamlit_authenticator as stauth

_SESSION_AUTHENTICATOR = "authenticator"


def _secrets_section(section: str) -> Dict[str, Any]:
    if section not in st.secrets:
        raise KeyError(
            f"Missing [{section}] in Streamlit Secrets. "
            "See .streamlit/secrets.toml.example for the required shape."
        )
    return dict(st.secrets[section])


def _load_auth_config() -> Dict[str, Any]:
    credentials = _secrets_section("credentials")
    cookie = _secrets_section("cookie")

    for field in ("name", "key", "expiry_days"):
        if field not in cookie:
            raise KeyError(f"Missing cookie.{field} in Streamlit Secrets.")

    if "usernames" not in credentials or not credentials["usernames"]:
        raise KeyError(
            "Missing credentials.usernames in Streamlit Secrets. "
            "Add at least one named user."
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
            auto_hash=False,
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
    authenticator = get_authenticator()

    try:
        authenticator.login(
            location="main",
            max_login_attempts=5,
        )
    except Exception:
        st.error("Authentication is temporarily unavailable. Please try again later.")
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
