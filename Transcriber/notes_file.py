"""Turn one transcribed call into a saved notes file.

run_pipeline() gives back segments and highlights. This writes them to the
call's folder as two files with the same content:

    notes.json  - for code: the website, the database, a video player seeking
    notes.md    - for people: open it, print it, send it to the student

Both use the same three data types:

    Segment  - one stretch of speech:   start/end timestamps + what was said
    KeyPart  - one key teaching moment: start/end timestamps + title + note
    timestamps are always given twice - "MM:SS" to read, seconds to seek to
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pydantic import BaseModel

from Transcriber.Transcribe import Segment, format_ts


class KeyPart(BaseModel):
    start_sec: float
    end_sec: float
    start_ts: str
    end_ts: str
    title: str
    note: str


class SessionNotes(BaseModel):
    session_id: str
    recorded_at: str          # ISO 8601, UTC
    tutor: str
    student: str
    duration_sec: float
    duration_ts: str
    key_parts: List[KeyPart]
    segments: List[Segment]


def _seconds(ts: str) -> float:
    """"MM:SS" or "HH:MM:SS" -> seconds."""
    total = 0.0
    for part in str(ts).strip().split(":"):
        total = total * 60 + float(part or 0)
    return total


def _key_part(highlight: dict, segments: List[Segment]) -> KeyPart:
    """Gemini only returns "MM:SS" for highlights. Borrow the exact seconds from
    the segments they point at, so seeking lands on the start of the sentence."""
    start = next((s.start_sec for s in segments if s.start_ts == highlight["start_ts"]),
                 _seconds(highlight["start_ts"]))
    end = next((s.end_sec for s in segments if s.end_ts == highlight["end_ts"]),
               _seconds(highlight["end_ts"]))
    return KeyPart(
        start_sec=start,
        end_sec=max(end, start),
        start_ts=format_ts(start),
        end_ts=format_ts(max(end, start)),
        title=highlight.get("title", ""),
        note=highlight.get("note", ""),
    )


def build_notes(result: dict, *, session_id: str, tutor: str, student: str,
                recorded_at: datetime | None = None) -> SessionNotes:
    segments = [Segment.model_validate(s) for s in result.get("segments") or []]
    key_parts = sorted((_key_part(h, segments) for h in result.get("highlights") or []),
                       key=lambda k: k.start_sec)
    duration = max((s.end_sec for s in segments), default=0.0)
    return SessionNotes(
        session_id=session_id,
        recorded_at=(recorded_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        tutor=tutor,
        student=student,
        duration_sec=duration,
        duration_ts=format_ts(duration),
        key_parts=key_parts,
        segments=segments,
    )


def to_markdown(notes: SessionNotes) -> str:
    when = datetime.fromisoformat(notes.recorded_at).strftime("%B %d, %Y at %H:%M UTC")
    lines = [
        "# Tutoring session notes",
        "",
        f"**Tutor:** {notes.tutor}  ",
        f"**Student:** {notes.student}  ",
        f"**Recorded:** {when}  ",
        f"**Length:** {notes.duration_ts}",
        "",
        "## Key parts",
        "",
    ]
    if notes.key_parts:
        for k in notes.key_parts:
            lines.append(f"- **[{k.start_ts}-{k.end_ts}] {k.title}**  ")
            lines.append(f"  {k.note}")
    else:
        lines.append("_No key parts were found in this session._")
    lines += ["", "## Full transcript", ""]
    if notes.segments:
        lines += [f"`[{s.start_ts}-{s.end_ts}]` {s.text}  " for s in notes.segments]
    else:
        lines.append("_No speech was detected in the recording._")
    return "\n".join(lines) + "\n"


def write_notes_files(notes: SessionNotes, folder: Path) -> dict[str, Path]:
    """Write notes.json and notes.md into the call's folder. Returns both paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"json": folder / "notes.json", "md": folder / "notes.md"}
    # newline="\n" so the files are byte-for-byte the same on Windows and the Linux server.
    paths["json"].write_text(json.dumps(notes.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    paths["md"].write_text(to_markdown(notes), encoding="utf-8", newline="\n")
    return paths
