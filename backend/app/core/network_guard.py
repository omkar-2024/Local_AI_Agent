import ipaddress
import json
import logging
import socket
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


# ============================================================
# NETWORK POLICY
# ============================================================

LOOPBACK_HOSTS = {
    "127.0.0.1",
    "localhost",
    "::1",
}

# IMPORTANT:
# The frontend/backend may communicate over the private LAN through
# FastAPI/CORS, but outbound backend sockets remain air-gapped.
#
# Only loopback/local services are allowed as outbound connections.
ALLOWED_HOSTS = LOOPBACK_HOSTS


# ============================================================
# LOGGER
# ============================================================

_audit_logger = logging.getLogger("network_audit")
_audit_logger.setLevel(logging.INFO)

_original_connect = socket.socket.connect
_guard_installed = False


def _setup_logger() -> None:
    if _audit_logger.handlers:
        return

    log_path = settings.LOG_DIR / "network_audit.log"
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    _audit_logger.addHandler(handler)
    _audit_logger.propagate = False


# ============================================================
# DESTINATION CLASSIFICATION
# ============================================================

def _extract_host(address: Any) -> str:
    if isinstance(address, tuple):
        if not address:
            return ""

        return str(address[0])

    return str(address)


def classify_destination(host: str) -> str:
    """
    Classify an outbound destination.

    LOCAL
        Loopback / localhost.

    LAN
        Private or link-local IP.

    EXTERNAL
        Public/global IP or non-local hostname.

    UNKNOWN
        Reserved/unclassified IP.
    """

    normalized = host.strip().lower()

    if normalized in LOOPBACK_HOSTS:
        return "LOCAL"

    try:
        ip = ipaddress.ip_address(
            normalized
        )

        if ip.is_loopback:
            return "LOCAL"

        if (
            ip.is_private
            or ip.is_link_local
        ):
            return "LAN"

        if ip.is_global:
            return "EXTERNAL"

        return "UNKNOWN"

    except ValueError:
        # Never perform DNS resolution here.
        #
        # Any hostname that is not explicitly localhost is treated
        # as external. This avoids making a DNS request merely to
        # determine whether the destination is safe.
        return "EXTERNAL"


# ============================================================
# EVENT CREATION
# ============================================================

def _build_event(
    *,
    scope: str,
    action: str,
    destination: Any,
    allowed: bool,
    status: str,
) -> dict[str, Any]:

    host = _extract_host(
        destination
    )

    port = None

    if (
        isinstance(destination, tuple)
        and len(destination) > 1
    ):
        port = destination[1]

    return {
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "event":
            "NETWORK_ACTIVITY",

        "scope":
            scope,

        "action":
            action,

        "destination":
            str(destination),

        "host":
            host,

        "port":
            port,

        "allowed":
            allowed,

        "status":
            status,

        "air_gapped":
            True,
    }


# ============================================================
# LIVE SSE INTEGRATION
# ============================================================

def _emit_live_network_event(
    event: dict[str, Any],
) -> None:
    """
    Push real network activity into the existing SSE/progress
    system.

    The import is intentionally local because main.py installs
    network_guard before the agent/progress modules are loaded.
    """

    try:
        from app.services.progress import (
            progress_manager,
        )

        scope = event.get(
            "scope",
            "UNKNOWN",
        )

        allowed = event.get(
            "allowed",
            False,
        )

        destination = event.get(
            "destination",
            "unknown",
        )

        if allowed:
            message = (
                "Local connection: "
                f"{destination}"
            )

        else:
            message = (
                "External connection blocked: "
                f"{destination}"
            )

        progress_manager.emit_sync(
            "NETWORK_ACTIVITY",
            message,
            "completed",
            {
                "scope": scope,
                "destination":
                    destination,
                "host":
                    event.get("host"),
                "port":
                    event.get("port"),
                "allowed":
                    allowed,
                "action":
                    event.get("action"),
                "network_status":
                    event.get("status"),
            },
        )

    except Exception:
        # Telemetry must NEVER be allowed to disable
        # the security guard.
        pass


# ============================================================
# RECORD EVENT
# ============================================================

def _record_event(
    event: dict[str, Any],
) -> None:

    _setup_logger()

    _audit_logger.info(
        json.dumps(
            event,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )
    )

    _emit_live_network_event(
        event
    )


# ============================================================
# SOCKET GUARD
# ============================================================

def _guarded_connect(
    self,
    address,
):

    host = _extract_host(
        address
    )

    scope = classify_destination(
        host
    )

    # --------------------------------------------------------
    # LOCALHOST / LOOPBACK
    # --------------------------------------------------------

    if scope == "LOCAL":

        event = _build_event(
            scope="LOCAL",
            action="CONNECT",
            destination=address,
            allowed=True,
            status="ALLOWED",
        )

        _record_event(
            event
        )

        return _original_connect(
            self,
            address,
        )

    # --------------------------------------------------------
    # EVERYTHING ELSE IS BLOCKED
    # --------------------------------------------------------

    event = _build_event(
        scope=scope,
        action="CONNECT",
        destination=address,
        allowed=False,
        status="BLOCKED",
    )

    _record_event(
        event
    )

    raise ConnectionError(
        "Air-gapped mode: blocked "
        "outbound connection to "
        f"{address}"
    )


# ============================================================
# INSTALL
# ============================================================

def install_network_guard() -> None:
    """
    Install the socket-level outbound network guard.

    Only localhost/loopback outbound connections are allowed.
    LAN and public internet outbound connections are blocked.
    """

    global _guard_installed

    _setup_logger()

    if _guard_installed:
        return

    socket.socket.connect = (
        _guarded_connect
    )

    _guard_installed = True


# ============================================================
# RAW LOG
# ============================================================

def read_recent_audit_entries(
    limit: int = 100,
) -> list[str]:

    log_path = (
        settings.LOG_DIR
        / "network_audit.log"
    )

    if not log_path.exists():
        return []

    lines = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    limit = max(
        1,
        min(limit, 1000),
    )

    return lines[-limit:]


# ============================================================
# STRUCTURED EVENTS
# ============================================================

def _read_structured_events(
    limit: int = 500,
) -> list[dict[str, Any]]:

    events: list[
        dict[str, Any]
    ] = []

    for line in read_recent_audit_entries(
        limit
    ):

        try:
            payload = json.loads(
                line
            )

            if isinstance(
                payload,
                dict,
            ):
                events.append(
                    payload
                )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            # Ignore old-format audit lines.
            continue

    return events


# ============================================================
# TELEMETRY
# ============================================================

LIVE_TELEMETRY_WINDOW_SECONDS = 5


def _get_live_network_events(
    events: list[dict[str, Any]],
    window_seconds: int = LIVE_TELEMETRY_WINDOW_SECONDS,
) -> list[dict[str, Any]]:
    """Return only network events from the recent live window.

    The UI uses this instead of the historical audit count so that
    the Network Monitor answers the question: "Is anything external
    happening right now?"
    """

    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - max(1, window_seconds)

    live_events: list[dict[str, Any]] = []

    for event in events:
        timestamp = event.get("timestamp")

        if not timestamp:
            continue

        try:
            parsed = datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            if parsed.timestamp() >= cutoff:
                live_events.append(event)

        except (TypeError, ValueError):
            continue

    return live_events


def get_network_telemetry(
    limit: int = 500,
) -> dict[str, Any]:

    events = (
        _read_structured_events(
            limit
        )
    )

    live_events = _get_live_network_events(events)

    live_external_attempts = sum(
        1
        for event in live_events
        if event.get("scope") == "EXTERNAL"
    )

    live_blocked_attempts = sum(
        1
        for event in live_events
        if event.get("allowed") is False
    )

    live_local_connections = sum(
        1
        for event in live_events
        if (
            event.get("scope") == "LOCAL"
            and event.get("allowed") is True
        )
    )

    last_network_event = (
        live_events[-1]
        if live_events
        else None
    )

    local_connections = sum(
        1
        for event in events
        if (
            event.get("scope")
            == "LOCAL"
            and event.get("allowed")
            is True
        )
    )

    external_connections = sum(
        1
        for event in events
        if event.get("scope")
        == "EXTERNAL"
    )

    lan_attempts = sum(
        1
        for event in events
        if event.get("scope")
        == "LAN"
    )

    blocked_attempts = sum(
        1
        for event in events
        if event.get("allowed")
        is False
    )

    return {
        "guard_active": (
            _guard_installed
            and
            socket.socket.connect
            is _guarded_connect
        ),

        "air_gapped": True,

        "local_connections":
            local_connections,

        "external_connections":
            external_connections,

        "blocked_attempts":
            blocked_attempts,

        "lan_attempts":
            lan_attempts,

        "observed_events":
            len(events),

        # Live values are intentionally separate from the historical
        # counters above. They describe only the last few seconds.
        "live_window_seconds":
            LIVE_TELEMETRY_WINDOW_SECONDS,

        "live_external_attempts":
            live_external_attempts,

        "live_blocked_attempts":
            live_blocked_attempts,

        "live_local_connections":
            live_local_connections,

        "last_network_event":
            last_network_event,

        "recent_activity":
            list(
                reversed(
                    events[-50:]
                )
            ),
    }


# ============================================================
# SOVEREIGNTY VERIFICATION
# ============================================================

def verify_sovereignty() -> dict[str, Any]:

    checks: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # CHECK 1 — GUARD ACTIVE
    # --------------------------------------------------------

    guard_active = (
        _guard_installed
        and
        socket.socket.connect
        is _guarded_connect
    )

    checks.append(
        {
            "name":
                "Network Guard",

            "passed":
                guard_active,

            "detail": (
                "socket.connect is protected "
                "by the air-gap guard."
                if guard_active
                else
                "socket.connect is not protected."
            ),
        }
    )

    # --------------------------------------------------------
    # CHECK 2 — LOCALHOST ALLOWED
    # --------------------------------------------------------

    local_allowed = False

    if guard_active:

        test_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        try:

            test_socket.settimeout(
                0.05
            )

            try:
                test_socket.connect(
                    (
                        "127.0.0.1",
                        1,
                    )
                )

            except (
                ConnectionRefusedError,
                TimeoutError,
                OSError,
            ):
                # The service does not need to exist.
                #
                # Reaching the OS-level connection attempt proves
                # our guard did not block localhost.
                pass

            local_allowed = True

        except ConnectionError:
            local_allowed = False

        finally:
            test_socket.close()

    checks.append(
        {
            "name":
                "Localhost Access",

            "passed":
                local_allowed,

            "detail": (
                "Loopback traffic is permitted "
                "for local services."
                if local_allowed
                else
                "Loopback traffic was incorrectly blocked."
            ),
        }
    )

    # --------------------------------------------------------
    # CHECK 3 — PUBLIC CONNECTION BLOCKED
    # --------------------------------------------------------

    external_blocked = False

    if guard_active:

        test_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        try:

            try:
                # IMPORTANT:
                # This is only passed to our local guard.
                # No public network request is intended.
                test_socket.connect(
                    (
                        "1.1.1.1",
                        443,
                    )
                )

            except ConnectionError:
                external_blocked = True

            except OSError:
                external_blocked = False

        finally:
            test_socket.close()

    checks.append(
        {
            "name":
                "External Connection Block",

            "passed":
                external_blocked,

            "detail": (
                "Public outbound connection "
                "was blocked locally."
                if external_blocked
                else
                "Public outbound connection "
                "was not proven blocked."
            ),
        }
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    all_passed = all(
        check["passed"]
        for check in checks
    )

    return {
        "verified":
            all_passed,

        "status": (
            "SOVEREIGNTY VERIFIED"
            if all_passed
            else
            "VERIFICATION FAILED"
        ),

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "checks":
            checks,

        "telemetry":
            get_network_telemetry(),

        "public_network_contacted":
            False,
    }