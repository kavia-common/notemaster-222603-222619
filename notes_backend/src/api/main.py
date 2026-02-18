"""notes_backend FastAPI application.

Provides a REST API for a notes application:
- Notes CRUD
- Search by query and tags
- Tag management
- Pin/Favorite toggles

The backend stores data in SQLite. Configure via environment variable:
- SQLITE_DB: absolute/relative path to the SQLite .db file

This app enables CORS for local development (frontend container).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

openapi_tags = [
    {"name": "Health", "description": "Service health and basic info."},
    {"name": "Notes", "description": "Create, read, update, delete, and search notes."},
    {"name": "Tags", "description": "List tags and manage note-tag assignments."},
]

app = FastAPI(
    title="Notemaster Notes API",
    description=(
        "FastAPI backend for the Notemaster notes app. "
        "Supports notes CRUD, tagging, pin/favorite, and simple search."
    ),
    version="0.1.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db_path() -> str:
    """Resolve database path from env with a safe default for dev."""
    return os.environ.get("SQLITE_DB", "myapp.db")


def _connect() -> sqlite3.Connection:
    """Create a SQLite connection with sane defaults."""
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the schema exists (idempotent).

    We create tables here as a safety net so backend can run even if the DB
    container wasn't initialized yet.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            is_pinned INTEGER NOT NULL DEFAULT 0,
            is_favorite INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_tags (
            note_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (note_id, tag_id),
            FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_pinned ON notes(is_pinned)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_favorite ON notes(is_favorite)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_note_tags_note ON note_tags(note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(tag_id)")
    conn.commit()


def get_db() -> sqlite3.Connection:
    """FastAPI dependency that yields a DB connection."""
    conn = _connect()
    try:
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


def _utc_iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _row_to_note(conn: sqlite3.Connection, note_row: sqlite3.Row) -> dict[str, Any]:
    """Convert a notes row to API dict including tags."""
    note_id = int(note_row["id"])
    tag_rows = conn.execute(
        """
        SELECT t.name
        FROM tags t
        INNER JOIN note_tags nt ON nt.tag_id = t.id
        WHERE nt.note_id = ?
        ORDER BY t.name
        """,
        (note_id,),
    ).fetchall()
    tags = [r["name"] for r in tag_rows]
    return {
        "id": note_id,
        "title": note_row["title"],
        "content": note_row["content"],
        "tags": tags,
        "is_pinned": bool(note_row["is_pinned"]),
        "is_favorite": bool(note_row["is_favorite"]),
        "created_at": note_row["created_at"],
        "updated_at": note_row["updated_at"],
    }


class NoteCreate(BaseModel):
    title: str = Field("", description="Short title of the note.")
    content: str = Field("", description="Markdown text content of the note.")
    tags: List[str] = Field(default_factory=list, description="List of tag names.")


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, description="New note title.")
    content: Optional[str] = Field(None, description="New markdown content.")
    tags: Optional[List[str]] = Field(None, description="Replace tags with this list.")
    is_pinned: Optional[bool] = Field(None, description="Whether the note is pinned.")
    is_favorite: Optional[bool] = Field(None, description="Whether the note is favorited.")


class NoteOut(BaseModel):
    id: int = Field(..., description="Note ID.")
    title: str = Field(..., description="Note title.")
    content: str = Field(..., description="Note markdown content.")
    tags: List[str] = Field(..., description="Tag names attached to the note.")
    is_pinned: bool = Field(..., description="Pinned flag.")
    is_favorite: bool = Field(..., description="Favorite flag.")
    created_at: str = Field(..., description="Creation timestamp (SQLite text).")
    updated_at: str = Field(..., description="Update timestamp (SQLite text).")


class TagOut(BaseModel):
    name: str = Field(..., description="Tag name.")
    note_count: int = Field(..., description="Number of notes using this tag.")


# PUBLIC_INTERFACE
@app.get("/", tags=["Health"], summary="Health check", description="Simple health check endpoint.")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        JSON with a human-readable status message.
    """
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.get(
    "/notes",
    response_model=List[NoteOut],
    tags=["Notes"],
    summary="List/search notes",
    description=(
        "List notes sorted by pinned desc then updated_at desc. "
        "Optional query searches title+content (case-insensitive). "
        "Optional tag filters to notes containing ALL specified tags."
    ),
)
def list_notes(
    q: Optional[str] = Query(None, description="Search query for title/content."),
    tags: Optional[List[str]] = Query(None, description="Filter: notes that contain ALL of these tags."),
    pinned: Optional[bool] = Query(None, description="Filter by pinned flag."),
    favorite: Optional[bool] = Query(None, description="Filter by favorite flag."),
    db: sqlite3.Connection = Depends(get_db),
) -> List[dict[str, Any]]:
    """List notes with optional search/filter."""
    where_clauses: list[str] = []
    params: list[Any] = []

    if q:
        where_clauses.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)")
        like = f"%{q.lower()}%"
        params.extend([like, like])

    if pinned is not None:
        where_clauses.append("is_pinned = ?")
        params.append(1 if pinned else 0)

    if favorite is not None:
        where_clauses.append("is_favorite = ?")
        params.append(1 if favorite else 0)

    base_sql = "SELECT * FROM notes"
    if where_clauses:
        base_sql += " WHERE " + " AND ".join(where_clauses)

    base_sql += " ORDER BY is_pinned DESC, updated_at DESC, id DESC"

    note_rows = db.execute(base_sql, tuple(params)).fetchall()

    # If tag filter present: keep notes that contain all tags
    if tags:
        wanted = sorted({t.strip() for t in tags if t.strip()})
        if wanted:
            filtered: list[sqlite3.Row] = []
            for r in note_rows:
                existing = db.execute(
                    """
                    SELECT t.name
                    FROM tags t
                    INNER JOIN note_tags nt ON nt.tag_id = t.id
                    WHERE nt.note_id = ?
                    """,
                    (int(r["id"]),),
                ).fetchall()
                existing_set = {x["name"] for x in existing}
                if all(t in existing_set for t in wanted):
                    filtered.append(r)
            note_rows = filtered

    return [_row_to_note(db, r) for r in note_rows]


def _get_or_create_tag_id(conn: sqlite3.Connection, name: str) -> int:
    """Return tag id for name, creating if missing."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Empty tag name")
    row = conn.execute("SELECT id FROM tags WHERE name = ?", (cleaned,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute("INSERT INTO tags (name) VALUES (?)", (cleaned,))
    conn.commit()
    return int(cur.lastrowid)


def _replace_note_tags(conn: sqlite3.Connection, note_id: int, tags: List[str]) -> None:
    """Replace note tags with the given list."""
    # Normalize unique, non-empty
    normalized = []
    seen = set()
    for t in tags:
        tt = t.strip()
        if tt and tt not in seen:
            normalized.append(tt)
            seen.add(tt)

    conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
    for t in normalized:
        tag_id = _get_or_create_tag_id(conn, t)
        conn.execute("INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)", (note_id, tag_id))
    conn.commit()


# PUBLIC_INTERFACE
@app.post(
    "/notes",
    response_model=NoteOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Notes"],
    summary="Create note",
    description="Create a new note with optional tags.",
)
def create_note(payload: NoteCreate, db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    """Create a new note."""
    now = _utc_iso_now()
    cur = db.execute(
        """
        INSERT INTO notes (title, content, is_pinned, is_favorite, created_at, updated_at)
        VALUES (?, ?, 0, 0, ?, ?)
        """,
        (payload.title or "", payload.content or "", now, now),
    )
    note_id = int(cur.lastrowid)
    db.commit()

    _replace_note_tags(db, note_id, payload.tags)

    row = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    return _row_to_note(db, row)


def _get_note_row_or_404(conn: sqlite3.Connection, note_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    return row


# PUBLIC_INTERFACE
@app.get(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Get note",
    description="Fetch a single note by ID (including tags).",
)
def get_note(note_id: int, db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    """Get a note by ID."""
    row = _get_note_row_or_404(db, note_id)
    return _row_to_note(db, row)


# PUBLIC_INTERFACE
@app.put(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Update note",
    description="Update note fields (partial update supported) and optionally replace tags.",
)
def update_note(note_id: int, payload: NoteUpdate, db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    """Update a note."""
    existing = _get_note_row_or_404(db, note_id)

    new_title = payload.title if payload.title is not None else existing["title"]
    new_content = payload.content if payload.content is not None else existing["content"]
    new_pinned = (1 if payload.is_pinned else 0) if payload.is_pinned is not None else int(existing["is_pinned"])
    new_fav = (1 if payload.is_favorite else 0) if payload.is_favorite is not None else int(existing["is_favorite"])

    now = _utc_iso_now()
    db.execute(
        """
        UPDATE notes
        SET title = ?, content = ?, is_pinned = ?, is_favorite = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_title, new_content, new_pinned, new_fav, now, note_id),
    )
    db.commit()

    if payload.tags is not None:
        _replace_note_tags(db, note_id, payload.tags)

    row = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    return _row_to_note(db, row)


# PUBLIC_INTERFACE
@app.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Notes"],
    summary="Delete note",
    description="Delete a note by ID (tag mappings are removed via cascade).",
)
def delete_note(note_id: int, db: sqlite3.Connection = Depends(get_db)) -> Response:
    """Delete a note."""
    _get_note_row_or_404(db, note_id)
    db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# PUBLIC_INTERFACE
@app.get(
    "/tags",
    response_model=List[TagOut],
    tags=["Tags"],
    summary="List tags",
    description="List tags with note usage counts.",
)
def list_tags(db: sqlite3.Connection = Depends(get_db)) -> List[dict[str, Any]]:
    """List tags with usage counts."""
    rows = db.execute(
        """
        SELECT t.name AS name, COUNT(nt.note_id) AS note_count
        FROM tags t
        LEFT JOIN note_tags nt ON nt.tag_id = t.id
        GROUP BY t.id
        ORDER BY LOWER(t.name) ASC
        """
    ).fetchall()
    return [{"name": r["name"], "note_count": int(r["note_count"])} for r in rows]
