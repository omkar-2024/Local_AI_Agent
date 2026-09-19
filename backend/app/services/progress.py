import asyncio
import json

from collections import defaultdict
from contextvars import ContextVar
from typing import (
    Any,
    AsyncGenerator,
)


# ============================================================
# EVENT NAMES
# ============================================================

REQUEST_RECEIVED = "REQUEST_RECEIVED"
REQUEST_UNDERSTOOD = "REQUEST_UNDERSTOOD"

PURIFYING_REQUEST = "PURIFYING_REQUEST"
PROCESSING_ATTACHMENT = "PROCESSING_ATTACHMENT"
OCR_PROCESSING = "OCR_PROCESSING"

ROUTING_REQUEST = "ROUTING_REQUEST"
ROUTE_SELECTED = "ROUTE_SELECTED"
MODEL_SELECTED = "MODEL_SELECTED"

EXECUTING_TOOL = "EXECUTING_TOOL"
RUNNING_PYTHON = "RUNNING_PYTHON"

CREATING_WORD_DOCUMENT = "CREATING_WORD_DOCUMENT"
CREATING_EXCEL_DOCUMENT = "CREATING_EXCEL_DOCUMENT"
CREATING_POWERPOINT = "CREATING_POWERPOINT"

NETWORK_ACTIVITY = "NETWORK_ACTIVITY"

GENERATING_RESPONSE = "GENERATING_RESPONSE"

COMPLETED = "COMPLETED"
ERROR = "ERROR"


# ============================================================
# CURRENT SESSION
# ============================================================

_current_session_id: ContextVar[
    str | None
] = ContextVar(
    "current_session_id",
    default=None,
)


# ============================================================
# PROGRESS MANAGER
# ============================================================

class ProgressManager:

    def __init__(self):

        self._queues: dict[
            str,
            asyncio.Queue
        ] = defaultdict(
            asyncio.Queue
        )

        # Persistent per-session audit history.
        #
        # The existing SSE queue is consumed by the frontend.
        # History is separate so the audit timeline can still be
        # displayed after an event has been consumed.
        self._history: dict[
            str,
            list[dict[str, Any]]
        ] = defaultdict(list)

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    def create(
        self,
        session_id: str,
    ):

        if session_id not in self._queues:

            self._queues[
                session_id
            ] = asyncio.Queue()

        if session_id not in self._history:

            self._history[
                session_id
            ] = []

    def set_session(
        self,
        session_id: str,
    ):

        _current_session_id.set(
            session_id
        )

        self.create(
            session_id
        )

    def get_session(
        self,
    ) -> str | None:

        return _current_session_id.get()

    # --------------------------------------------------------
    # EMIT
    # --------------------------------------------------------

    async def emit(
        self,
        session_id: str,
        event: str,
        message: str,
        status: str = "active",
        data: dict[str, Any] | None = None,
    ):

        self.create(
            session_id
        )

        payload = {
            "event":
                event,

            "message":
                message,

            "status":
                status,

            "data":
                data or {},
        }

        # Store audit history separately from the SSE queue.
        history = self._history[
            session_id
        ]

        history.append(
            payload
        )

        # Keep memory bounded.
        if len(history) > 100:

            del history[
                :-100
            ]

        # Preserve the existing SSE behaviour.
        await self._queues[
            session_id
        ].put(
            payload
        )

    # --------------------------------------------------------
    # SYNC COMPATIBILITY
    # --------------------------------------------------------

    def emit_sync(
        self,
        event: str,
        message: str,
        status: str = "active",
        data: dict[str, Any] | None = None,
    ):
        """
        Synchronous wrapper used by the orchestrator and
        network guard.
        """

        session_id = (
            self.get_session()
        )

        if not session_id:
            return

        try:

            loop = (
                asyncio.get_running_loop()
            )

            loop.create_task(
                self.emit(
                    session_id,
                    event,
                    message,
                    status,
                    data,
                )
            )

        except RuntimeError:
            pass

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def reset_history(
        self,
        session_id: str,
    ) -> None:
        """
        Start a fresh audit timeline for a new agent request.

        The SSE queue is intentionally preserved so an already-open
        frontend EventSource continues to receive the new request's
        events. Only the persistent history used by the monitor is
        cleared.
        """
        self.create(session_id)
        self._history[session_id].clear()

    def get_history(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:

        return list(
            self._history.get(
                session_id,
                [],
            )
        )

    # --------------------------------------------------------
    # SSE
    # --------------------------------------------------------

    async def stream(
        self,
        session_id: str,
    ) -> AsyncGenerator[str, None]:

        self.create(
            session_id
        )

        queue = self._queues[
            session_id
        ]

        # Initial connection event.
        yield (
            "event: connected\n"
            "data: "
            + json.dumps(
                {
                    "event":
                        "CONNECTED",

                    "message":
                        "Connected to backend progress",

                    "status":
                        "active",

                    "data":
                        {},
                }
            )
            + "\n\n"
        )

        while True:

            payload = await queue.get()

            yield (
                "event: progress\n"
                "data: "
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            event_name = payload.get(
                "event"
            )

            event_data = (
                payload.get(
                    "data"
                )
                or {}
            )

            # ERROR ends the stream.
            if event_name == ERROR:

                break

            # Final COMPLETED ends the stream.
            if (
                event_name
                == COMPLETED
                and
                event_data.get(
                    "final"
                )
                is True
            ):

                break

        # The queue can be removed after completion.
        #
        # IMPORTANT:
        # History is intentionally retained so the audit endpoint
        # can still display the execution timeline.
        self._queues.pop(
            session_id,
            None,
        )


# ============================================================
# SINGLETON
# ============================================================

progress_manager = (
    ProgressManager()
)


# ============================================================
# ORCHESTRATOR COMPATIBILITY
# ============================================================

def emit_event(
    event: str,
    message: str,
    status: str = "active",
    data: dict[str, Any] | None = None,
):
    """
    Keeps the existing orchestrator API unchanged.

    Existing calls such as:

        emit_event(
            EVENT,
            "message"
        )

    continue to work.
    """

    progress_manager.emit_sync(
        event,
        message,
        status,
        data,
    )