from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional
import json
import sqlite3

from app.core.config import settings


# ============================================================
# DATABASE LOCATION
# ============================================================

DATA_DIR = (
    settings.BASE_DIR
    / "data"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DB_PATH = (
    DATA_DIR
    / "chat.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA journal_mode = WAL"
    )

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    connection = _connect()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                request_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,

                FOREIGN KEY (
                    conversation_id
                )
                REFERENCES conversations(id)
                ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation
            ON messages (
                conversation_id,
                id
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_request
            ON messages (
                request_id
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


initialize_database()


# ============================================================
# TIME
# ============================================================

def _now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# TITLE
# ============================================================

def _make_title(
    content: str,
) -> str:

    text = " ".join(
        str(content)
        .strip()
        .split()
    )

    if not text:
        return "New Chat"

    if len(text) <= 60:
        return text

    return (
        text[:57].rstrip()
        + "..."
    )


# ============================================================
# CREATE / ENSURE CONVERSATION
# ============================================================

def ensure_conversation(
    conversation_id: str,
    title: Optional[str] = None,
) -> dict[str, Any]:

    conversation_id = (
        str(conversation_id)
        .strip()
    )

    if not conversation_id:
        raise ValueError(
            "conversation_id cannot be empty."
        )

    now = _now()

    connection = _connect()

    try:
        existing = connection.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        ).fetchone()

        if existing:
            return dict(existing)

        final_title = (
            title
            or "New Chat"
        )

        connection.execute(
            """
            INSERT INTO conversations (
                id,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                final_title,
                now,
                now,
            ),
        )

        connection.commit()

        return {
            "id":
                conversation_id,
            "title":
                final_title,
            "created_at":
                now,
            "updated_at":
                now,
        }

    finally:
        connection.close()


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

def update_conversation_title(
    conversation_id: str,
    title: str,
):
    title = _make_title(
        title
    )

    connection = _connect()

    try:
        connection.execute(
            """
            UPDATE conversations
            SET
                title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                _now(),
                conversation_id,
            ),
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    conversation_id: str,
    role: str,
    content: str,
    request_id: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> int:

    conversation_id = (
        str(conversation_id)
        .strip()
    )

    role = (
        str(role)
        .strip()
        .lower()
    )

    content = str(content)

    if not conversation_id:
        raise ValueError(
            "conversation_id cannot be empty."
        )

    if role not in {
        "user",
        "assistant",
        "system",
    }:
        raise ValueError(
            f"Unsupported message role: {role}"
        )

    ensure_conversation(
        conversation_id
    )

    metadata_json = json.dumps(
        metadata or {},
        ensure_ascii=False,
        default=str,
    )

    now = _now()

    connection = _connect()

    try:
        cursor = connection.execute(
            """
            INSERT INTO messages (
                conversation_id,
                request_id,
                role,
                content,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                request_id,
                role,
                content,
                metadata_json,
                now,
            ),
        )

        # --------------------------------------------------------
        # AUTOMATIC CHAT TITLE
        #
        # The first meaningful user message becomes the title.
        # --------------------------------------------------------

        if role == "user":

            existing_user_message = connection.execute(
                """
                SELECT id
                FROM messages
                WHERE conversation_id = ?
                AND role = 'user'
                AND id != ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (
                    conversation_id,
                    cursor.lastrowid,
                ),
            ).fetchone()

            if (
                existing_user_message
                is None
            ):
                connection.execute(
                    """
                    UPDATE conversations
                    SET
                        title = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        _make_title(
                            content
                        ),
                        now,
                        conversation_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        conversation_id,
                    ),
                )

        else:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    conversation_id,
                ),
            )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:
        connection.close()


# ============================================================
# GET MESSAGES
# ============================================================

def get_messages(
    conversation_id: str,
    limit: Optional[int] = None,
    exclude_request_id: Optional[str] = None,
) -> list[dict[str, Any]]:

    connection = _connect()

    try:
        sql = """
            SELECT
                id,
                conversation_id,
                request_id,
                role,
                content,
                metadata_json,
                created_at
            FROM messages
            WHERE conversation_id = ?
        """

        parameters: list[Any] = [
            conversation_id
        ]

        if exclude_request_id:
            sql += """
                AND (
                    request_id IS NULL
                    OR request_id != ?
                )
            """

            parameters.append(
                exclude_request_id
            )

        sql += """
            ORDER BY id ASC
        """

        if limit is not None:
            safe_limit = max(
                1,
                min(
                    int(limit),
                    100,
                ),
            )

            sql += """
                LIMIT ?
            """

            parameters.append(
                safe_limit
            )

        rows = connection.execute(
            sql,
            parameters,
        ).fetchall()

        messages = []

        for row in rows:

            try:
                metadata = json.loads(
                    row[
                        "metadata_json"
                    ]
                    or "{}"
                )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    metadata = {}

            except Exception:
                metadata = {}

            messages.append(
                {
                    "id":
                        row["id"],

                    "conversation_id":
                        row[
                            "conversation_id"
                        ],

                    "request_id":
                        row[
                            "request_id"
                        ],

                    "role":
                        row["role"],

                    "content":
                        row["content"],

                    "created_at":
                        row[
                            "created_at"
                        ],

                    **metadata,
                }
            )

        return messages

    finally:
        connection.close()


# ============================================================
# RECENT CONTEXT
# ============================================================

def get_recent_messages(
    conversation_id: str,
    limit: int = 12,
    exclude_request_id: Optional[str] = None,
) -> list[dict[str, Any]]:

    connection = _connect()

    try:
        safe_limit = max(
            1,
            min(
                int(limit),
                50,
            ),
        )

        if exclude_request_id:

            rows = connection.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    request_id,
                    role,
                    content,
                    metadata_json,
                    created_at
                FROM messages
                WHERE conversation_id = ?
                AND (
                    request_id IS NULL
                    OR request_id != ?
                )
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    conversation_id,
                    exclude_request_id,
                    safe_limit,
                ),
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    request_id,
                    role,
                    content,
                    metadata_json,
                    created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    conversation_id,
                    safe_limit,
                ),
            ).fetchall()

        rows = list(
            reversed(rows)
        )

        messages = []

        for row in rows:

            try:
                metadata = json.loads(
                    row[
                        "metadata_json"
                    ]
                    or "{}"
                )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    metadata = {}

            except Exception:
                metadata = {}

            messages.append(
                {
                    "id":
                        row["id"],

                    "conversation_id":
                        row[
                            "conversation_id"
                        ],

                    "request_id":
                        row[
                            "request_id"
                        ],

                    "role":
                        row["role"],

                    "content":
                        row["content"],

                    "created_at":
                        row[
                            "created_at"
                        ],

                    **metadata,
                }
            )

        return messages

    finally:
        connection.close()


# ============================================================
# GET CONVERSATIONS
# ============================================================

def get_conversations(
    limit: int = 50,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    connection = _connect()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (
                safe_limit,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


# ============================================================
# DELETE CONVERSATION
# ============================================================

def delete_conversation(
    conversation_id: str,
) -> bool:

    connection = _connect()

    try:
        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        )

        connection.commit()

        return (
            cursor.rowcount > 0
        )

    finally:
        connection.close()