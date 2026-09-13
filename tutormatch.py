"""TutorMatch - user data, storage, and learning-style matching.

Everything for the data layer lives in this one file. Owned by Caden.

    USING IT FROM THE WEBSITE
    -------------------------
        from tutormatch import QUIZ_QUESTIONS, upsert_user, save_learner_profile
        from tutormatch import match_for_answers, build_profile_sentence

        upsert_user(auth_sub=..., name=..., email=...)      # after Auth0 login
        sentence, matches = match_for_answers(answers, ["calculus"])

    COMMAND LINE
    ------------
        python tutormatch.py schema    create the database tables
        python tutormatch.py seed      load the 16 tutors
        python tutormatch.py check     show match results for 5 test students

    SETUP
    -----
        pip install -r requirements.txt
        put TIGER_URL=<connection string> in .env

    HOW MATCHING WORKS
    ------------------
    The quiz asks how a student learns across five axes. Every tutor is scored
    0..1 on those same axes. A match is a tutor who scores high on the answers
    this student actually gave. The scoring is a weighted sum computed by the
    database - no AI, so it's fast, deterministic, and explainable.

    SECTIONS
    --------
        1. CONFIG      reading .env
        2. SCHEMAS     the data shapes
        3. DATABASE    connection, tables, user data
        4. QUIZ        the six questions
        5. TUTORS      the 16 seeded tutors and their style scores
        6. MATCHING    scoring and explanations
        7. CLI         schema / seed / check commands
"""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

# =====================================================================
# 1. CONFIG
# =====================================================================

# Load .env from beside this file, regardless of where python was started.
load_dotenv(Path(__file__).resolve().parent / ".env")


@dataclass(frozen=True)
class Settings:
    """TigerData connection string, from .env.

    Auth0, Solana and storage settings belong to whoever owns those tracks -
    read them in your own module rather than adding them here.
    """

    tiger_url: str = os.getenv("TIGER_URL", "")

    @property
    def has_db(self) -> bool:
        return bool(self.tiger_url)


settings = Settings()


# =====================================================================
# 2. SCHEMAS - the data shapes
# =====================================================================

Role = Literal["student", "tutor"]


class _Dictable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


@dataclass
class User(_Dictable):
    auth_sub: str
    role: Role
    name: str
    email: str
    avatar_url: str | None = None
    wallet_address: str | None = None


@dataclass
class QuizOption(_Dictable):
    value: str
    label: str


@dataclass
class QuizQuestion(_Dictable):
    id: str
    prompt: str
    options: list[QuizOption] = field(default_factory=list)
    free_text: bool = False


@dataclass
class TutorMatch(_Dictable):
    user: User
    bio: str
    subjects: list[str]
    hourly_rate_sol: float
    rating: float
    # 0..1 weighted blend of style fit, subject overlap and rating.
    score: float
    # Why this tutor fits, built from the scores that ranked them.
    rationale: str = ""
    # Component scores, kept for debugging the ranking.
    style_similarity: float = 0.0
    subject_overlap: float = 0.0
    teaching_levels: list[str] = field(default_factory=list)
    # For linking to the tutor's profile page.
    public_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["user"] = self.user.to_dict()
        return d


@dataclass
class TutorFixture:
    """A seeded tutor.

    `style` scores them 0..1 on every possible quiz answer, keyed by question
    id then answer value. Those numbers are what matching runs on, so they
    should agree with what the bio says - if they drift apart, the ranking
    stops matching the description.
    """

    auth_sub: str
    name: str
    email: str
    avatar_url: str
    bio: str
    subjects: list[str]
    hourly_rate_sol: float
    rating: float
    style: dict[str, dict[str, float]] = field(default_factory=dict)
    teaching_levels: list[str] = field(default_factory=list)


# =====================================================================
# 3. DATABASE
# =====================================================================

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()

CREATE TABLE IF NOT EXISTS users (
  auth_sub        text PRIMARY KEY,
  -- Opaque ID for profile URLs, so Auth0 IDs never appear in addresses.
  -- md5 is just a stable slug here, not security.
  public_id       text GENERATED ALWAYS AS (substr(md5(auth_sub), 1, 16)) STORED,
  role            text NOT NULL DEFAULT 'student',
  -- Chosen at onboarding. Auth0's `name` is the email address for
  -- email/password sign-ups, so it must never be shown on a public profile.
  display_name    text,
  name            text,
  email           text,
  avatar_url      text,
  wallet_address  text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learner_profiles (
  user_id           text PRIMARY KEY REFERENCES users(auth_sub) ON DELETE CASCADE,
  pace              text,
  goals             text,
  subjects          text[],
  -- The quiz answers themselves, keyed by question id. This is what matching
  -- scores against, so keep it readable: {"intake": "visual", "pace": "slow"}
  raw_answers       jsonb,
  profile_sentence  text,
  grade_level       text,           -- one of GRADE_LEVELS, e.g. 'high'
  updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tutor_profiles (
  user_id           text PRIMARY KEY REFERENCES users(auth_sub) ON DELETE CASCADE,
  bio               text,
  subjects          text[],
  -- {"intake": {"visual": 1.0, "verbal": 0.3, ...}, "pace": {...}, ...}
  style_affinity    jsonb,
  hourly_rate_sol   numeric(10,4),
  rating            numeric(3,2) DEFAULT 4.5,
  teaching_levels   text[]          -- GRADE_LEVELS values this tutor teaches
);

CREATE TABLE IF NOT EXISTS sessions (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  -- Whatever string the video app uses to name a room, e.g. "call-42".
  -- Lets other parts of the system reference a session by the id they
  -- already have instead of having to know our uuid.
  room_key       text UNIQUE,
  student_id     text REFERENCES users(auth_sub),
  tutor_id       text REFERENCES users(auth_sub),
  subject        text,
  mode           text NOT NULL DEFAULT 'video',
  status         text NOT NULL DEFAULT 'pending',
  room_url       text,
  recording_url  text,
  recording_key  text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  started_at     timestamptz,
  ended_at       timestamptz
);

CREATE TABLE IF NOT EXISTS payments (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      uuid REFERENCES sessions(id) ON DELETE CASCADE,
  tx_signature    text UNIQUE NOT NULL,
  amount_lamports bigint,
  status          text NOT NULL DEFAULT 'pending',
  confirmed_at    timestamptz
);

CREATE TABLE IF NOT EXISTS session_notes (
  session_id    uuid PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  summary       text,
  key_moments   jsonb,           -- [{ tMs, title, why }]  tMs is MILLISECONDS
  concepts      text[],
  action_items  text[],
  generated_at  timestamptz NOT NULL DEFAULT now()
);

-- These two are genuinely time-series, which is what makes TimescaleDB
-- load-bearing here rather than decorative.
--
-- NOTE: a hypertable's PRIMARY KEY / UNIQUE index must include the
-- partitioning column. That is why neither table has a plain `id PRIMARY KEY`.

CREATE TABLE IF NOT EXISTS transcript_segments (
  session_id  uuid NOT NULL,
  ts          timestamptz NOT NULL DEFAULT now(),
  start_ms    integer NOT NULL,
  end_ms      integer NOT NULL,
  speaker     text,
  text        text NOT NULL,
  embedding   vector(768)
);

CREATE TABLE IF NOT EXISTS messages (
  session_id  uuid NOT NULL,
  ts          timestamptz NOT NULL DEFAULT now(),
  sender_id   text,
  body        text NOT NULL
);

SELECT create_hypertable('transcript_segments', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('messages',            'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS transcript_embedding_hnsw
  ON transcript_segments USING hnsw (embedding vector_cosine_ops);

-- Subject filtering uses the && array-overlap operator.
CREATE INDEX IF NOT EXISTS tutor_subjects_gin
  ON tutor_profiles USING gin (subjects);

CREATE INDEX IF NOT EXISTS transcript_session_ts
  ON transcript_segments (session_id, ts DESC);

CREATE INDEX IF NOT EXISTS messages_session_ts
  ON messages (session_id, ts DESC);

CREATE INDEX IF NOT EXISTS sessions_student
  ON sessions (student_id, created_at DESC);

-- Migration for databases created before room_key existed. No-op otherwise.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS room_key text;
CREATE UNIQUE INDEX IF NOT EXISTS sessions_room_key ON sessions (room_key);

-- Migration for databases created before grade/teaching levels existed.
ALTER TABLE learner_profiles ADD COLUMN IF NOT EXISTS grade_level text;
ALTER TABLE tutor_profiles   ADD COLUMN IF NOT EXISTS teaching_levels text[];

-- Migration for databases created before profiles had names and public IDs.
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name text;
ALTER TABLE users ADD COLUMN IF NOT EXISTS public_id text
  GENERATED ALWAYS AS (substr(md5(auth_sub), 1, 16)) STORED;
CREATE UNIQUE INDEX IF NOT EXISTS users_public_id ON users (public_id);
"""

_pool: ConnectionPool | None = None

# Managed TigerData tiers cap connections, and four developers plus a deployed
# app add up fast. Keep this small.
_MAX_POOL = 8


def get_pool() -> ConnectionPool:
    global _pool
    if not settings.tiger_url:
        raise RuntimeError(
            "TIGER_URL is not set. Put your TigerData connection string in .env"
        )
    if _pool is None:
        _pool = ConnectionPool(
            settings.tiger_url,
            min_size=1,
            max_size=_MAX_POOL,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        # Close cleanly at exit. Without this the pool's worker threads are
        # still alive during interpreter shutdown and Python 3.14 raises a
        # noisy PythonFinalizationError after every script.
        atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """Yield a cursor inside a transaction that commits on clean exit."""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            yield cur


def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(sql: str, params: tuple | dict | None = None) -> dict[str, Any] | None:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(sql: str, params: tuple | dict | None = None) -> int:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def ping() -> dict[str, Any]:
    """Connectivity + extension check."""
    return fetch_one(
        """
        SELECT version() AS pg_version,
               (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') AS timescaledb,
               (SELECT extversion FROM pg_extension WHERE extname = 'vector')      AS pgvector
        """
    ) or {}


def apply_schema() -> None:
    """Create the tables. Idempotent - safe to re-run."""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------- user data


def upsert_user(
    auth_sub: str,
    *,
    name: str | None = None,
    email: str | None = None,
    avatar_url: str | None = None,
    role: str = "student",
    wallet_address: str | None = None,
) -> dict[str, Any]:
    """Create the user row if it's their first time, otherwise refresh it.

    This is the Auth0 -> TigerData bridge. Call it after login with Auth0's
    `sub` claim. Safe to call every time - it updates rather than duplicating.

    `auth_sub` MUST be the same string every time for the same person. If it
    varies, users silently duplicate and their quiz answers get orphaned.
    """
    return fetch_one(
        """
        INSERT INTO users (auth_sub, role, name, email, avatar_url, wallet_address)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (auth_sub) DO UPDATE SET
            name           = COALESCE(EXCLUDED.name, users.name),
            email          = COALESCE(EXCLUDED.email, users.email),
            avatar_url     = COALESCE(EXCLUDED.avatar_url, users.avatar_url),
            wallet_address = COALESCE(EXCLUDED.wallet_address, users.wallet_address)
        RETURNING *
        """,
        (auth_sub, role, name, email, avatar_url, wallet_address),
    ) or {}


def get_user(auth_sub: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM users WHERE auth_sub = %s", (auth_sub,))


def save_learner_profile(
    user_id: str,
    *,
    subjects: list[str],
    pace: str,
    goals: str,
    raw_answers: dict[str, Any],
    profile_sentence: str,
    grade_level: str | None = None,
) -> None:
    """Store the quiz answers. raw_answers is what matching scores against."""
    execute(
        """
        INSERT INTO learner_profiles
            (user_id, subjects, pace, goals, raw_answers, profile_sentence, grade_level, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (user_id) DO UPDATE SET
            subjects         = EXCLUDED.subjects,
            pace             = EXCLUDED.pace,
            goals            = EXCLUDED.goals,
            raw_answers      = EXCLUDED.raw_answers,
            profile_sentence = EXCLUDED.profile_sentence,
            grade_level      = EXCLUDED.grade_level,
            updated_at       = now()
        """,
        (user_id, subjects, pace, goals, Json(raw_answers), profile_sentence, grade_level),
    )


def get_learner_profile(user_id: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM learner_profiles WHERE user_id = %s", (user_id,))


def set_user_role(auth_sub: str, role: str) -> None:
    """Make someone a student or a tutor.

    Becoming a student removes any tutor profile, so a person who switches
    roles stops appearing in other students' matches.
    """
    if role not in ("student", "tutor"):
        raise ValueError(f"role must be 'student' or 'tutor', not {role!r}")
    execute("UPDATE users SET role = %s WHERE auth_sub = %s", (role, auth_sub))
    if role == "student":
        execute("DELETE FROM tutor_profiles WHERE user_id = %s", (auth_sub,))


def save_tutor_profile(
    user_id: str,
    *,
    bio: str,
    subjects: list[str],
    teaching_levels: list[str],
    style_affinity: dict[str, dict[str, float]],
    hourly_rate_sol: float,
) -> None:
    """Make a real account matchable as a tutor. Rating keeps its existing value."""
    execute(
        """
        INSERT INTO tutor_profiles
            (user_id, bio, subjects, teaching_levels, style_affinity, hourly_rate_sol)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            bio             = EXCLUDED.bio,
            subjects        = EXCLUDED.subjects,
            teaching_levels = EXCLUDED.teaching_levels,
            style_affinity  = EXCLUDED.style_affinity,
            hourly_rate_sol = EXCLUDED.hourly_rate_sol
        """,
        (user_id, bio, subjects, teaching_levels, Json(style_affinity), hourly_rate_sol),
    )


def get_tutor_profile(user_id: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM tutor_profiles WHERE user_id = %s", (user_id,))


def get_profile(auth_sub: str) -> dict[str, Any] | None:
    """Everything about one person: their user row and whichever profiles exist."""
    user = get_user(auth_sub)
    if user is None:
        return None
    return {
        "user": user,
        "learner": get_learner_profile(auth_sub),
        "tutor": get_tutor_profile(auth_sub),
    }


def is_onboarded(auth_sub: str) -> bool:
    """True once someone has a profile for their current role."""
    profile = get_profile(auth_sub)
    if profile is None:
        return False
    key = "tutor" if profile["user"]["role"] == "tutor" else "learner"
    return profile[key] is not None


def set_display_name(auth_sub: str, display_name: str) -> None:
    execute("UPDATE users SET display_name = %s WHERE auth_sub = %s", (display_name, auth_sub))


def get_user_by_public_id(public_id: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM users WHERE public_id = %s", (public_id,))


def public_name(user: dict[str, Any]) -> str:
    """The name to show other people. Never an email address.

    Auth0 fills `name` with the email for email/password sign-ups, so it is
    only used when it doesn't look like one - which covers seeded tutors and
    social logins that provide a real name.
    """
    if user.get("display_name"):
        return user["display_name"]
    name = user.get("name") or ""
    return name if name and "@" not in name else "TutorMatch member"


def pair_room_key(sub_a: str, sub_b: str) -> str:
    """The chat/call room for two people - the same whichever of them asks.

    Sorting makes it order-independent, so a student opening a chat with a
    tutor and that tutor opening a chat with the student land in one room.
    Hashed so the room name doesn't reveal either Auth0 ID.
    """
    pair = "\n".join(sorted((sub_a, sub_b)))
    return "pair-" + hashlib.sha256(pair.encode("utf-8")).hexdigest()[:24]


# ----------------------------------------------------------------- sessions


def create_session(
    student_id: str,
    tutor_id: str,
    subject: str,
    *,
    mode: str = "video",
    room_url: str | None = None,
    room_key: str | None = None,
) -> str:
    """Start a tutoring session. Returns the session id everything else keys off."""
    row = fetch_one(
        """
        INSERT INTO sessions (student_id, tutor_id, subject, mode, room_url, room_key, status)
        VALUES (%s, %s, %s, %s, %s, %s, 'pending')
        RETURNING id
        """,
        (student_id, tutor_id, subject, mode, room_url, room_key),
    )
    return str(row["id"])


def resolve_session(key: str) -> str:
    """Turn any session identifier into the uuid the tables key off.

    The video call names rooms with whatever string is in the URL, e.g.
    "call-42". Rather than make that side adopt our uuids, anything that
    isn't already one is treated as a room_key and looked up - created on
    first sight if it doesn't exist yet.

    So a recording can be saved against a room nobody formally booked, and
    it still lands in the right place if a booking is created later.
    """
    key = str(key)

    # Already one of our uuids?
    row = fetch_one("SELECT id FROM sessions WHERE id::text = %s", (key,))
    if row:
        return str(row["id"])

    row = fetch_one("SELECT id FROM sessions WHERE room_key = %s", (key,))
    if row:
        return str(row["id"])

    row = fetch_one(
        """
        INSERT INTO sessions (room_key, status) VALUES (%s, 'live')
        ON CONFLICT (room_key) DO UPDATE SET room_key = EXCLUDED.room_key
        RETURNING id
        """,
        (key,),
    )
    return str(row["id"])

def attach_participant(session_id: str, role: str, auth_sub: str) -> None:
    """Record who actually joined a video-call session, keyed by their Auth0 sub.

    Call this when someone joins /call - it fills in the sessions row's
    student_id or tutor_id, even for ad-hoc sessions created by resolve_session
    that were never formally booked.
    """
    session_id = resolve_session(session_id)
    if role == "tutor":
        execute("UPDATE sessions SET tutor_id = %s WHERE id = %s", (auth_sub, session_id))
    else:
        execute("UPDATE sessions SET student_id = %s WHERE id = %s", (auth_sub, session_id))


def get_session_participants(session_id: str) -> dict[str, Any] | None:
    """The two Auth0 subs (student_id, tutor_id) attached to a session, for access checks."""
    return fetch_one(
        "SELECT student_id, tutor_id FROM sessions WHERE id::text = %s OR room_key = %s",
        (str(session_id), str(session_id)),
    )

def get_session(session_id: str) -> dict[str, Any] | None:
    """Look up a session by uuid or by room_key."""
    return fetch_one(
        "SELECT * FROM sessions WHERE id::text = %s OR room_key = %s",
        (str(session_id), str(session_id)),
    )


def set_session_status(
    session_id: str,
    status: str,
    *,
    recording_url: str | None = None,
) -> None:
    """Move a session along: pending -> paid -> live -> ended -> processed."""
    session_id = resolve_session(session_id)
    execute(
        """
        UPDATE sessions SET
            status        = %s,
            recording_url = COALESCE(%s, recording_url),
            started_at    = CASE WHEN %s = 'live'  THEN now() ELSE started_at END,
            ended_at      = CASE WHEN %s = 'ended' THEN now() ELSE ended_at   END
        WHERE id = %s
        """,
        (status, recording_url, status, status, session_id),
    )


# ------------------------------------------------------- transcripts + notes


def ts_to_ms(value: Any) -> int:
    """Normalise a timestamp into integer milliseconds.

    The AI pipeline emits "MM:SS" (and sometimes "HH:MM:SS") strings because
    they read well on screen. The video player needs a number to seek to.
    Everything stored in the database is milliseconds, so the conversion
    happens here, once, rather than in every caller.

        ts_to_ms("02:34")   -> 154000
        ts_to_ms(154.0)     -> 154000      (seconds as a number)
    """
    if isinstance(value, (int, float)):
        return int(round(float(value) * 1000))

    text = str(value).strip()
    if not text:
        return 0
    if ":" not in text:
        return int(round(float(text) * 1000))

    parts = [float(p) for p in text.split(":")]
    seconds = 0.0
    for part in parts:            # handles MM:SS and HH:MM:SS alike
        seconds = seconds * 60 + part
    return int(round(seconds * 1000))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field off either a dict or an object (e.g. a pydantic model)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def save_session_notes(
    session_id: str,
    highlights: list[Any],
    *,
    summary: str = "",
    concepts: list[str] | None = None,
    action_items: list[str] | None = None,
) -> None:
    """Store the AI-generated notes for a session.

    `highlights` accepts the AI pipeline's own shape directly - dicts or
    pydantic models with start_ts / title / note. Timestamps are converted to
    integer milliseconds on the way in, so the website can feed them straight
    to the player's seek call.

        from tutormatch import save_session_notes
        save_session_notes(session_id, highlights, summary=...)
    """
    session_id = resolve_session(session_id)
    key_moments = [
        {
            "tMs": ts_to_ms(_field(h, "start_ts", _field(h, "start_sec", 0))),
            "title": _field(h, "title", "") or "",
            "why": _field(h, "note", _field(h, "why", "")) or "",
        }
        for h in highlights or []
    ]

    execute(
        """
        INSERT INTO session_notes
            (session_id, summary, key_moments, concepts, action_items, generated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (session_id) DO UPDATE SET
            summary      = EXCLUDED.summary,
            key_moments  = EXCLUDED.key_moments,
            concepts     = EXCLUDED.concepts,
            action_items = EXCLUDED.action_items,
            generated_at = now()
        """,
        (session_id, summary, Json(key_moments), concepts or [], action_items or []),
    )


def get_session_notes(session_id: str) -> dict[str, Any] | None:
    """Read notes back for the recap page. key_moments[].tMs is milliseconds."""
    return fetch_one(
        "SELECT * FROM session_notes WHERE session_id = %s", (resolve_session(session_id),)
    )


def save_transcript_segments(session_id: str, segments: list[Any]) -> int:
    """Store the transcript. Accepts the AI pipeline's segment shape directly.

    Replaces any existing transcript for the session, so re-running the
    pipeline doesn't leave two copies interleaved.
    """
    session_id = resolve_session(session_id)
    rows = [
        (
            session_id,
            ts_to_ms(_field(s, "start_sec", _field(s, "start_ts", 0))),
            ts_to_ms(_field(s, "end_sec", _field(s, "end_ts", 0))),
            _field(s, "speaker", "") or "",
            _field(s, "text", "") or "",
        )
        for s in segments or []
    ]
    if not rows:
        return 0

    with cursor() as cur:
        cur.execute("DELETE FROM transcript_segments WHERE session_id = %s", (session_id,))
        cur.executemany(
            """
            INSERT INTO transcript_segments (session_id, start_ms, end_ms, speaker, text)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows,
        )
    return len(rows)


def get_transcript(session_id: str) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT start_ms, end_ms, speaker, text
        FROM transcript_segments
        WHERE session_id = %s
        ORDER BY start_ms
        """,
        (resolve_session(session_id),),
    )


def save_pipeline_result(
    session_id: str,
    result: dict[str, Any],
    *,
    recording_url: str | None = None,
) -> dict[str, int]:
    """Store a whole transcription run in one call.

    Takes `run_pipeline()`'s return value as-is - {"segments", "highlights"} -
    so wiring the video app to the database is a single line:

        from tutormatch import save_pipeline_result
        save_pipeline_result(session_id, transcription, recording_url=str(save_path))

    `session_id` can be the room id the video call already uses; it does not
    have to be one of our uuids.
    """
    session_id = resolve_session(session_id)
    segments = result.get("segments") or []
    highlights = result.get("highlights") or []

    n_segments = save_transcript_segments(session_id, segments)
    save_session_notes(session_id, highlights)
    set_session_status(session_id, "processed", recording_url=recording_url)

    return {"segments": n_segments, "highlights": len(highlights)}


# ------------------------------------------------------------------ messages


def save_message(session_id: str, sender_id: str, body: str) -> None:
    """Store one chat message. Call this from the websocket handler."""
    execute(
        "INSERT INTO messages (session_id, sender_id, body) VALUES (%s, %s, %s)",
        (resolve_session(session_id), sender_id, body),
    )


def get_messages(session_id: str, limit: int = 500) -> list[dict[str, Any]]:
    """Chat history for a session, oldest first."""
    rows = fetch_all(
        """
        SELECT ts, sender_id, body FROM messages
        WHERE session_id = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (resolve_session(session_id), limit),
    )
    return list(reversed(rows))


# =====================================================================
# 4. QUIZ - the six questions
# =====================================================================

# The website renders these. Don't write your own - matching scores
# against these exact answer values.

QUIZ_QUESTIONS: list[QuizQuestion] = [
    QuizQuestion(
        id="intake",
        prompt="When you meet a new idea, what makes it click fastest?",
        options=[
            QuizOption("visual", "A diagram or picture of how it fits together"),
            QuizOption("verbal", "Someone talking me through it out loud"),
            QuizOption("reading", "Reading a clear written explanation"),
            QuizOption("kinesthetic", "Trying it myself and seeing what breaks"),
        ],
    ),
    QuizQuestion(
        id="explanation",
        prompt="Which explanation style do you prefer?",
        options=[
            QuizOption("examples_first", "Show me a worked example, then the rule"),
            QuizOption("theory_first", "Give me the rule, then we apply it"),
            QuizOption("analogy", "Compare it to something I already understand"),
            QuizOption("discovery", "Let me poke at it until I figure it out"),
        ],
    ),
    QuizQuestion(
        id="pace",
        prompt="What pace suits you?",
        options=[
            QuizOption("slow", "Slow and thorough - I want every step"),
            QuizOption("moderate", "Moderate - steady with room for questions"),
            QuizOption("fast", "Fast - hit the highlights, I fill gaps myself"),
        ],
    ),
    QuizQuestion(
        id="when_stuck",
        prompt="When you're stuck, what do you want your tutor to do?",
        options=[
            QuizOption("hint", "Drop a small hint and let me keep trying"),
            QuizOption("walkthrough", "Walk me through the whole thing start to finish"),
            QuizOption("socratic", "Ask me questions until I find it myself"),
            QuizOption("similar_example", "Show a similar problem solved, then I retry"),
        ],
    ),
    QuizQuestion(
        id="structure",
        prompt="How should a session be structured?",
        options=[
            QuizOption("agenda", "A clear agenda we work through"),
            QuizOption("freeform", "Freeform - I bring questions as they come"),
            QuizOption("drilling", "Lots of practice problems back to back"),
            QuizOption("discussion", "Open discussion of the underlying concepts"),
        ],
    ),
    QuizQuestion(
        id="goal",
        prompt="What are you working toward right now?",
        free_text=True,
    ),
]

# Students pick one; tutors pick every level they teach. Matching only pairs a
# student with tutors who teach their level.
GRADE_LEVELS: list[QuizOption] = [
    QuizOption("elementary", "Elementary (K-5)"),
    QuizOption("middle", "Middle school (6-8)"),
    QuizOption("high", "High school (9-12)"),
    QuizOption("college", "College"),
    QuizOption("adult", "Adult / professional"),
]

SUBJECTS: list[QuizOption] = [
    QuizOption("calculus", "Calculus"),
    QuizOption("physics", "Physics"),
    QuizOption("chemistry", "Chemistry"),
    QuizOption("biology", "Biology"),
    QuizOption("computer_science", "Computer science"),
]

# The same questions asked from the tutor's side. Answer values are identical
# to QUIZ_QUESTIONS on purpose: a tutor's "how I teach" answer is scored
# directly against a student's "how I learn" answer.
TEACHING_QUESTIONS: list[QuizQuestion] = [
    QuizQuestion(
        id="intake",
        prompt="How do you usually get a new idea across?",
        options=[
            QuizOption("visual", "Draw it - diagrams and pictures"),
            QuizOption("verbal", "Talk it through out loud"),
            QuizOption("reading", "Write out a clear explanation"),
            QuizOption("kinesthetic", "Get them trying it hands-on"),
        ],
    ),
    QuizQuestion(
        id="explanation",
        prompt="How do you usually explain something new?",
        options=[
            QuizOption("examples_first", "A worked example first, then the rule"),
            QuizOption("theory_first", "The rule first, then we apply it"),
            QuizOption("analogy", "An analogy to something familiar"),
            QuizOption("discovery", "Let them explore until they work it out"),
        ],
    ),
    QuizQuestion(
        id="pace",
        prompt="What pace do you usually teach at?",
        options=[
            QuizOption("slow", "Slow and thorough - every step"),
            QuizOption("moderate", "Steady, with room for questions"),
            QuizOption("fast", "Fast - highlights, skip the basics"),
        ],
    ),
    QuizQuestion(
        id="when_stuck",
        prompt="When a student is stuck, what do you do?",
        options=[
            QuizOption("hint", "Give a small hint and let them keep trying"),
            QuizOption("walkthrough", "Walk them through it start to finish"),
            QuizOption("socratic", "Ask questions until they find it"),
            QuizOption("similar_example", "Solve a similar problem, then hand it back"),
        ],
    ),
    QuizQuestion(
        id="structure",
        prompt="How do you run a session?",
        options=[
            QuizOption("agenda", "A clear agenda we work through"),
            QuizOption("freeform", "Freeform - they bring the questions"),
            QuizOption("drilling", "Lots of practice problems"),
            QuizOption("discussion", "Discussion of the underlying concepts"),
        ],
    ),
]


def clean_answers(submitted: dict[str, Any], questions: list[QuizQuestion]) -> dict[str, str]:
    """Keep only answers that are real options for these questions.

    Form input can't be trusted, and matching looks answers up by value, so an
    unknown value would silently score zero instead of failing loudly.
    """
    cleaned: dict[str, str] = {}
    for q in questions:
        value = str(submitted.get(q.id) or "").strip()
        if q.free_text:
            if value:
                cleaned[q.id] = value[:500]
        elif value in {o.value for o in q.options}:
            cleaned[q.id] = value
    return cleaned


def style_from_teaching_answers(answers: dict[str, str]) -> dict[str, dict[str, float]]:
    """Turn a tutor's quiz answers into the affinity scores matching runs on.

    The answer they picked scores 1.0 on each axis; the alternatives 0.25, since
    most tutors can still flex toward other styles. Seeded tutors have
    hand-tuned scores instead - this is the version real accounts get.
    """
    return {
        q.id: {o.value: (1.0 if o.value == answers.get(q.id) else 0.25) for o in q.options}
        for q in TEACHING_QUESTIONS
    }


# =====================================================================
# 5. TUTORS - the seeded pool
# =====================================================================

# The bios are deliberately DISTINCT from one another, and each tutor's
# `style` scores should agree with what their bio says. To retune a match,
# edit the numbers here and re-run `python tutormatch.py seed`.

TUTORS: list[TutorFixture] = [
    TutorFixture(
        "seed|tutor-01", "Maya Okonkwo", "maya@tutormatch.tech",
        "https://i.pravatar.cc/160?img=47",
        "Teaches almost entirely through diagrams. Draws the shape of a problem on a "
        "shared whiteboard before touching any algebra, and will redraw it three "
        "different ways until the picture makes the answer obvious. Best for people "
        "who need to see a thing to believe it.",
        ["calculus"], 0.18, 4.9,
        style={
            "intake": {"visual": 1.0, "kinesthetic": 0.4, "verbal": 0.3, "reading": 0.1},
            "explanation": {"examples_first": 0.8, "analogy": 0.6, "discovery": 0.3, "theory_first": 0.2},
            "pace": {"moderate": 0.8, "slow": 0.7, "fast": 0.2},
            "when_stuck": {"similar_example": 0.8, "walkthrough": 0.7, "hint": 0.4, "socratic": 0.3},
            "structure": {"agenda": 0.5, "discussion": 0.5, "freeform": 0.5, "drilling": 0.3},
        },
        teaching_levels=["high", "college"],
    ),
    TutorFixture(
        "seed|tutor-02", "Dev Raghunathan", "dev@tutormatch.tech",
        "https://i.pravatar.cc/160?img=12",
        "Relentless problem-drilling. Expect twenty short problems in an hour, graded "
        "live, with the pattern named after each one. Not much theory talk - the "
        "belief here is that fluency comes from reps and that speed removes fear.",
        ["calculus"], 0.12, 4.6,
        style={
            "intake": {"kinesthetic": 0.9, "visual": 0.3, "verbal": 0.3, "reading": 0.2},
            "explanation": {"examples_first": 0.9, "theory_first": 0.3, "discovery": 0.3, "analogy": 0.2},
            "pace": {"fast": 1.0, "moderate": 0.5, "slow": 0.1},
            "when_stuck": {"similar_example": 0.9, "hint": 0.6, "walkthrough": 0.4, "socratic": 0.2},
            "structure": {"drilling": 1.0, "agenda": 0.6, "freeform": 0.2, "discussion": 0.1},
        },
        teaching_levels=["high", "college"],
    ),
    TutorFixture(
        "seed|tutor-03", "Priya Venkatesan", "priya@tutormatch.tech",
        "https://i.pravatar.cc/160?img=32",
        "Socratic to a fault. Almost never gives a direct answer, instead asking "
        "narrowing questions until the student says the thing out loud themselves. "
        "Slow going at first and enormously sticky afterwards. Suits people who "
        "resent being handed answers.",
        ["calculus", "physics"], 0.20, 4.8,
        style={
            "intake": {"verbal": 0.9, "kinesthetic": 0.4, "visual": 0.3, "reading": 0.2},
            "explanation": {"discovery": 1.0, "analogy": 0.5, "examples_first": 0.2, "theory_first": 0.2},
            "pace": {"slow": 0.9, "moderate": 0.6, "fast": 0.1},
            "when_stuck": {"socratic": 1.0, "hint": 0.8, "similar_example": 0.2, "walkthrough": 0.0},
            "structure": {"discussion": 0.9, "freeform": 0.7, "agenda": 0.3, "drilling": 0.1},
        },
        teaching_levels=["high", "college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-04", "Tomas Lindqvist", "tomas@tutormatch.tech",
        "https://i.pravatar.cc/160?img=52",
        "Formal and theory-first. Starts from the definition, states the theorem, "
        "proves it, and only then works an example. Dry by design. Students who want "
        "to know why a rule is true, rather than just how to apply it, tend to stay "
        "for months.",
        ["calculus"], 0.22, 4.5,
        style={
            "intake": {"reading": 0.9, "verbal": 0.5, "visual": 0.3, "kinesthetic": 0.1},
            "explanation": {"theory_first": 1.0, "examples_first": 0.3, "discovery": 0.2, "analogy": 0.1},
            "pace": {"slow": 0.8, "moderate": 0.6, "fast": 0.3},
            "when_stuck": {"walkthrough": 0.7, "socratic": 0.4, "similar_example": 0.4, "hint": 0.3},
            "structure": {"agenda": 0.8, "discussion": 0.8, "drilling": 0.3, "freeform": 0.3},
        },
        teaching_levels=["college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-05", "Rosa Delgado", "rosa@tutormatch.tech",
        "https://i.pravatar.cc/160?img=45",
        "Explains everything by analogy to ordinary life - derivatives as "
        "speedometers, integrals as filling a bathtub. Warm, chatty, low-pressure "
        "sessions aimed at students who have decided they are bad at math and need "
        "that belief dismantled before anything else.",
        ["calculus"], 0.14, 4.7,
        style={
            "intake": {"verbal": 0.9, "visual": 0.5, "kinesthetic": 0.3, "reading": 0.2},
            "explanation": {"analogy": 1.0, "examples_first": 0.7, "discovery": 0.2, "theory_first": 0.1},
            "pace": {"slow": 1.0, "moderate": 0.6, "fast": 0.0},
            "when_stuck": {"walkthrough": 0.9, "similar_example": 0.6, "hint": 0.4, "socratic": 0.2},
            "structure": {"freeform": 0.8, "discussion": 0.7, "agenda": 0.4, "drilling": 0.2},
        },
        teaching_levels=["middle", "high", "college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-06", "Ken Arai", "ken@tutormatch.tech",
        "https://i.pravatar.cc/160?img=13",
        "Structured agenda every time: five minutes reviewing last session, thirty on "
        "the new topic, twenty on mixed practice, five setting homework. Sends written "
        "notes afterward. Ideal for students who are behind and need a plan more than "
        "inspiration.",
        ["calculus", "chemistry"], 0.16, 4.8,
        style={
            "intake": {"reading": 0.7, "verbal": 0.6, "visual": 0.5, "kinesthetic": 0.3},
            "explanation": {"examples_first": 0.7, "theory_first": 0.6, "analogy": 0.3, "discovery": 0.1},
            "pace": {"moderate": 1.0, "slow": 0.7, "fast": 0.3},
            "when_stuck": {"walkthrough": 0.8, "similar_example": 0.7, "hint": 0.3, "socratic": 0.2},
            "structure": {"agenda": 1.0, "drilling": 0.6, "discussion": 0.3, "freeform": 0.1},
        },
        teaching_levels=["middle", "high", "college"],
    ),
    TutorFixture(
        "seed|tutor-07", "Amara Boateng", "amara@tutormatch.tech",
        "https://i.pravatar.cc/160?img=26",
        "Runs chemistry as a visual molecular story - builds 3D models on screen and "
        "rotates them so students can see why a reaction goes one way and not the "
        "other. Very little rote memorisation; heavy emphasis on seeing the mechanism.",
        ["chemistry"], 0.19, 4.9,
        style={
            "intake": {"visual": 1.0, "kinesthetic": 0.5, "verbal": 0.4, "reading": 0.2},
            "explanation": {"analogy": 0.6, "examples_first": 0.6, "theory_first": 0.4, "discovery": 0.4},
            "pace": {"moderate": 0.8, "slow": 0.6, "fast": 0.3},
            "when_stuck": {"similar_example": 0.6, "walkthrough": 0.6, "hint": 0.5, "socratic": 0.4},
            "structure": {"discussion": 0.8, "agenda": 0.5, "freeform": 0.5, "drilling": 0.2},
        },
        teaching_levels=["high", "college"],
    ),
    TutorFixture(
        "seed|tutor-08", "Colin Mbeki", "colin@tutormatch.tech",
        "https://i.pravatar.cc/160?img=14",
        "Exam-focused chemistry drilling. Works from past papers exclusively, times "
        "every question, and teaches the specific tricks markers reward. Blunt "
        "feedback. Students cramming for a date circled on the calendar do well here.",
        ["chemistry"], 0.13, 4.4,
        style={
            "intake": {"kinesthetic": 0.7, "reading": 0.6, "visual": 0.3, "verbal": 0.3},
            "explanation": {"examples_first": 0.9, "theory_first": 0.3, "analogy": 0.1, "discovery": 0.1},
            "pace": {"fast": 1.0, "moderate": 0.5, "slow": 0.1},
            "when_stuck": {"similar_example": 0.9, "walkthrough": 0.8, "hint": 0.3, "socratic": 0.1},
            "structure": {"drilling": 1.0, "agenda": 0.7, "freeform": 0.2, "discussion": 0.1},
        },
        teaching_levels=["high", "college"],
    ),
    TutorFixture(
        "seed|tutor-09", "Hannah Weiss", "hannah@tutormatch.tech",
        "https://i.pravatar.cc/160?img=44",
        "Lab-first chemistry. Every concept arrives attached to an experiment, often "
        "demonstrated live on camera. Encourages students to predict the outcome "
        "before seeing it, then explains the gap between guess and result. Messy, "
        "memorable sessions.",
        ["chemistry"], 0.21, 4.7,
        style={
            "intake": {"kinesthetic": 1.0, "visual": 0.7, "verbal": 0.4, "reading": 0.2},
            "explanation": {"discovery": 0.9, "examples_first": 0.5, "analogy": 0.4, "theory_first": 0.2},
            "pace": {"moderate": 0.7, "slow": 0.5, "fast": 0.4},
            "when_stuck": {"hint": 0.8, "socratic": 0.6, "similar_example": 0.4, "walkthrough": 0.3},
            "structure": {"freeform": 0.7, "discussion": 0.6, "agenda": 0.4, "drilling": 0.2},
        },
        teaching_levels=["elementary", "middle", "high"],
    ),
    TutorFixture(
        "seed|tutor-10", "Yusuf Karim", "yusuf@tutormatch.tech",
        "https://i.pravatar.cc/160?img=59",
        "Reads like a textbook in the best way - precise written explanations shared "
        "in the chat as the session goes, so the student leaves with a clean document. "
        "Quiet, unhurried, and excellent for learners who process by reading rather "
        "than listening.",
        ["chemistry", "biology"], 0.15, 4.6,
        style={
            "intake": {"reading": 1.0, "verbal": 0.3, "visual": 0.3, "kinesthetic": 0.1},
            "explanation": {"theory_first": 0.6, "examples_first": 0.6, "analogy": 0.4, "discovery": 0.2},
            "pace": {"slow": 0.8, "moderate": 0.7, "fast": 0.2},
            "when_stuck": {"walkthrough": 0.8, "similar_example": 0.6, "hint": 0.4, "socratic": 0.3},
            "structure": {"agenda": 0.7, "discussion": 0.5, "freeform": 0.5, "drilling": 0.3},
        },
        teaching_levels=["high", "college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-11", "Grace Sullivan", "grace@tutormatch.tech",
        "https://i.pravatar.cc/160?img=31",
        "Project-based computer science. No lectures - the student builds something "
        "small and real from minute one, and concepts get introduced only when the "
        "build demands them. Suits people who lose interest in abstractions with no "
        "payoff.",
        ["computer_science"], 0.20, 4.9,
        style={
            "intake": {"kinesthetic": 1.0, "visual": 0.4, "verbal": 0.4, "reading": 0.2},
            "explanation": {"discovery": 1.0, "examples_first": 0.4, "analogy": 0.3, "theory_first": 0.1},
            "pace": {"fast": 0.8, "moderate": 0.6, "slow": 0.2},
            "when_stuck": {"hint": 0.9, "socratic": 0.6, "similar_example": 0.4, "walkthrough": 0.2},
            "structure": {"freeform": 1.0, "discussion": 0.4, "drilling": 0.2, "agenda": 0.2},
        },
        teaching_levels=["high", "college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-12", "Emeka Nwosu", "emeka@tutormatch.tech",
        "https://i.pravatar.cc/160?img=68",
        "Whiteboards data structures as pictures - boxes, arrows, and memory laid out "
        "visually before a single line of code. Especially good at making pointers and "
        "recursion stop being terrifying. Interview-prep heavy.",
        ["computer_science"], 0.23, 4.8,
        style={
            "intake": {"visual": 1.0, "verbal": 0.5, "kinesthetic": 0.3, "reading": 0.2},
            "explanation": {"examples_first": 0.7, "analogy": 0.6, "theory_first": 0.4, "discovery": 0.3},
            "pace": {"moderate": 0.8, "fast": 0.5, "slow": 0.5},
            "when_stuck": {"similar_example": 0.7, "walkthrough": 0.7, "hint": 0.4, "socratic": 0.3},
            "structure": {"agenda": 0.7, "drilling": 0.6, "discussion": 0.4, "freeform": 0.3},
        },
        teaching_levels=["college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-13", "Sofia Marchetti", "sofia@tutormatch.tech",
        "https://i.pravatar.cc/160?img=49",
        "Pair-programs the entire session with the student driving and never touches "
        "the keyboard. Asks what do you think that error means more than anything "
        "else. Frustrating for people in a hurry, transformative for people who want "
        "independence.",
        ["computer_science"], 0.17, 4.7,
        style={
            "intake": {"kinesthetic": 1.0, "verbal": 0.6, "visual": 0.3, "reading": 0.2},
            "explanation": {"discovery": 0.9, "examples_first": 0.3, "analogy": 0.3, "theory_first": 0.1},
            "pace": {"slow": 0.7, "moderate": 0.7, "fast": 0.2},
            "when_stuck": {"socratic": 1.0, "hint": 0.8, "similar_example": 0.3, "walkthrough": 0.1},
            "structure": {"freeform": 0.8, "discussion": 0.5, "agenda": 0.3, "drilling": 0.2},
        },
        teaching_levels=["high", "college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-14", "Arjun Malhotra", "arjun@tutormatch.tech",
        "https://i.pravatar.cc/160?img=56",
        "Theory-heavy computer science - complexity analysis, formal correctness, why "
        "an algorithm is optimal rather than merely working. Moves fast and assumes "
        "the student wants depth. Not the right fit for someone with a deadline "
        "tomorrow.",
        ["computer_science"], 0.25, 4.5,
        style={
            "intake": {"reading": 0.8, "verbal": 0.6, "visual": 0.4, "kinesthetic": 0.2},
            "explanation": {"theory_first": 1.0, "discovery": 0.4, "examples_first": 0.3, "analogy": 0.2},
            "pace": {"fast": 1.0, "moderate": 0.4, "slow": 0.1},
            "when_stuck": {"socratic": 0.5, "walkthrough": 0.5, "hint": 0.5, "similar_example": 0.3},
            "structure": {"discussion": 1.0, "agenda": 0.5, "drilling": 0.3, "freeform": 0.3},
        },
        teaching_levels=["college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-15", "Nadia Haddad", "nadia@tutormatch.tech",
        "https://i.pravatar.cc/160?img=24",
        "Teaches programming through analogy and story - a queue is a checkout line, a "
        "hash map is a coat check. Patient with absolute beginners and deliberately "
        "avoids jargon until the idea has landed. Gentle pace, lots of encouragement.",
        ["computer_science"], 0.14, 4.8,
        style={
            "intake": {"verbal": 0.9, "visual": 0.5, "kinesthetic": 0.3, "reading": 0.3},
            "explanation": {"analogy": 1.0, "examples_first": 0.6, "discovery": 0.2, "theory_first": 0.1},
            "pace": {"slow": 1.0, "moderate": 0.5, "fast": 0.0},
            "when_stuck": {"walkthrough": 0.9, "similar_example": 0.6, "hint": 0.4, "socratic": 0.2},
            "structure": {"freeform": 0.8, "discussion": 0.6, "agenda": 0.4, "drilling": 0.2},
        },
        teaching_levels=["elementary", "middle", "high", "college", "adult"],
    ),
    TutorFixture(
        "seed|tutor-16", "Leo Fitzgerald", "leo@tutormatch.tech",
        "https://i.pravatar.cc/160?img=51",
        "Fast, high-level sessions for students who are already competent and want the "
        "last twenty percent. Skips fundamentals, talks in shorthand, and focuses on "
        "edge cases and elegance. Explicitly not for beginners.",
        ["computer_science", "calculus"], 0.26, 4.6,
        style={
            "intake": {"verbal": 0.7, "reading": 0.6, "visual": 0.4, "kinesthetic": 0.4},
            "explanation": {"theory_first": 0.7, "discovery": 0.6, "examples_first": 0.3, "analogy": 0.2},
            "pace": {"fast": 1.0, "moderate": 0.3, "slow": 0.0},
            "when_stuck": {"hint": 0.8, "socratic": 0.6, "similar_example": 0.3, "walkthrough": 0.2},
            "structure": {"discussion": 0.8, "freeform": 0.7, "agenda": 0.3, "drilling": 0.3},
        },
        teaching_levels=["college", "adult"],
    ),
]


# =====================================================================
# 6. MATCHING
# =====================================================================

# The five quiz axes that carry style signal. `goal` is free text and is not
# scored - it's kept for display and for the tutor to read before a session.
STYLE_AXES = ("intake", "explanation", "pace", "when_stuck", "structure")

# Style dominates on purpose: matching on how someone learns is the whole
# premise. Subject is a strong secondary signal, rating a tiebreak.
W_STYLE = 0.55
W_SUBJECT = 0.30
W_RATING = 0.15

# Human-readable phrasing for each answer, from the student's side.
_LEARNER_PHRASES: dict[str, dict[str, str]] = {
    "intake": {
        "visual": "learns best from diagrams and pictures",
        "verbal": "learns best by being talked through ideas",
        "reading": "learns best from clear written explanations",
        "kinesthetic": "learns best by trying things hands-on",
    },
    "explanation": {
        "examples_first": "wants a worked example before the rule",
        "theory_first": "wants the rule stated first, then applied",
        "analogy": "understands ideas through analogies",
        "discovery": "prefers working things out independently",
    },
    "pace": {
        "slow": "wants a slow, thorough pace",
        "moderate": "wants a steady, moderate pace",
        "fast": "wants a fast pace that skips basics",
    },
    "when_stuck": {
        "hint": "wants a hint and space to keep trying when stuck",
        "walkthrough": "wants a full walkthrough when stuck",
        "socratic": "wants guiding questions rather than answers",
        "similar_example": "wants to see a similar problem solved first",
    },
    "structure": {
        "agenda": "wants sessions run to a clear agenda",
        "freeform": "wants freeform, question-driven sessions",
        "drilling": "wants lots of back-to-back practice",
        "discussion": "wants to discuss the underlying concepts",
    },
}

# The same answers phrased from the tutor's side, used to build the
# "why you matched" line without an AI.
_TUTOR_PHRASES: dict[str, dict[str, str]] = {
    "intake": {
        "visual": "teaches through diagrams and visuals",
        "verbal": "talks students through ideas out loud",
        "reading": "leaves you with clear written explanations",
        "kinesthetic": "gets you doing rather than watching",
    },
    "explanation": {
        "examples_first": "shows a worked example before the rule",
        "theory_first": "starts from the rule and then applies it",
        "analogy": "explains through everyday analogies",
        "discovery": "lets you work things out yourself",
    },
    "pace": {
        "slow": "works at a slow, thorough pace",
        "moderate": "keeps a steady pace with room for questions",
        "fast": "moves fast and skips the basics",
    },
    "when_stuck": {
        "hint": "gives a nudge rather than the answer",
        "walkthrough": "walks you through the whole problem",
        "socratic": "asks questions until you find it yourself",
        "similar_example": "solves a similar problem first, then hands it back",
    },
    "structure": {
        "agenda": "runs every session to a clear agenda",
        "freeform": "lets you drive the session with your own questions",
        "drilling": "runs lots of practice problems back to back",
        "discussion": "digs into the concepts underneath",
    },
}


def build_profile_sentence(answers: dict[str, str], subjects: list[str]) -> str:
    """A readable summary of the learner, for display and for debugging matches."""
    parts = [
        _LEARNER_PHRASES[ax][answers[ax]]
        for ax in STYLE_AXES
        if answers.get(ax) in _LEARNER_PHRASES.get(ax, {})
    ]
    sentence = "A student who " + ", ".join(parts) if parts else "A student"
    if subjects:
        sentence += f". Studying {', '.join(s.replace('_', ' ') for s in subjects)}"
    goal = (answers.get("goal") or "").strip()
    if goal:
        sentence += f". Their goal: {goal}"
    return sentence + "."


def explain(answers: dict[str, str], affinity: dict[str, Any], limit: int = 2) -> str:
    """Say why this tutor fits, naming the answers that actually drove the score.

    Picks the axes where the tutor scores highest on this student's answers, so
    the explanation is true by construction - it reads the same numbers the
    ranking used.
    """
    scored: list[tuple[float, str]] = []
    for ax in STYLE_AXES:
        ans = answers.get(ax)
        if not ans:
            continue
        score = float((affinity.get(ax) or {}).get(ans, 0.0))
        phrase = _TUTOR_PHRASES.get(ax, {}).get(ans)
        if phrase and score >= 0.6:
            scored.append((score, phrase))

    scored.sort(reverse=True)
    picked = [p for _, p in scored[:limit]]
    if not picked:
        return "Covers your subject, though their teaching style is a looser fit."
    if len(picked) == 1:
        return f"This tutor {picked[0]} - which is what you asked for."
    return f"This tutor {picked[0]} and {picked[1]} - both things you asked for."


def describe_teaching_style(affinity: dict[str, Any]) -> list[str]:
    """How a tutor teaches, one trait per axis, for their profile page.

    Uses each axis's highest-scored answer, which is the tutor's own pick for
    real accounts and the strongest trait for hand-tuned seeded tutors.
    """
    traits = []
    for ax in STYLE_AXES:
        scores = affinity.get(ax) or {}
        if scores:
            top = max(scores, key=lambda k: float(scores[k]))
            if phrase := _TUTOR_PHRASES.get(ax, {}).get(top):
                traits.append(phrase)
    return traits


def describe_learning_style(answers: dict[str, Any]) -> list[str]:
    """How a student learns, one trait per answered axis, for their profile page."""
    return [
        _LEARNER_PHRASES[ax][answers[ax]]
        for ax in STYLE_AXES
        if answers.get(ax) in _LEARNER_PHRASES.get(ax, {})
    ]


# Scoring happens in SQL so TigerData does the work. Each axis contributes the
# tutor's stored affinity for the answer this student gave; missing entries
# score 0 via COALESCE.
_STYLE_SUM = " + ".join(
    f"COALESCE((tp.style_affinity -> '{ax}' ->> %({ax})s)::float, 0)" for ax in STYLE_AXES
)

MATCH_SQL = f"""
WITH scored AS (
    SELECT
        u.auth_sub, u.name, u.display_name, u.public_id, u.email, u.avatar_url,
        tp.bio, tp.subjects, tp.hourly_rate_sol, tp.rating, tp.style_affinity,
        tp.teaching_levels,
        ({_STYLE_SUM}) / {len(STYLE_AXES)}.0 AS style_score,
        CASE WHEN cardinality(%(subjects)s::text[]) = 0 THEN 1.0
             ELSE cardinality(ARRAY(SELECT unnest(tp.subjects)
                                    INTERSECT SELECT unnest(%(subjects)s::text[])))::float
                  / cardinality(%(subjects)s::text[])
        END AS subject_overlap
    FROM tutor_profiles tp
    JOIN users u ON u.auth_sub = tp.user_id
    WHERE tp.style_affinity IS NOT NULL
      AND (cardinality(%(subjects)s::text[]) = 0 OR tp.subjects && %(subjects)s::text[])
      AND (%(max_rate)s::numeric IS NULL OR tp.hourly_rate_sol <= %(max_rate)s::numeric)
      -- Only tutors who teach the student's level. A tutor with no levels set
      -- isn't excluded, so older profiles still appear until they're updated.
      AND (%(grade)s::text IS NULL
           OR coalesce(cardinality(tp.teaching_levels), 0) = 0
           OR %(grade)s::text = ANY(tp.teaching_levels))
)
SELECT *,
       {W_STYLE} * style_score
     + {W_SUBJECT} * subject_overlap
     + {W_RATING} * (rating / 5.0) AS score
FROM scored
ORDER BY score DESC
LIMIT %(limit)s
"""


def find_matches(
    answers: dict[str, str],
    subjects: list[str],
    *,
    max_rate: float | None = None,
    grade_level: str | None = None,
    limit: int = 3,
) -> list[TutorMatch]:
    """Rank tutors for a learner. Hard-filters on subject, price and level, then scores."""
    params: dict[str, Any] = {
        "subjects": subjects,
        "max_rate": max_rate,
        "grade": grade_level,
        "limit": limit,
    }
    # Answers the student didn't give become a sentinel that matches no key,
    # so that axis contributes 0 rather than erroring.
    for ax in STYLE_AXES:
        params[ax] = answers.get(ax) or "__none__"

    rows = fetch_all(MATCH_SQL, params)

    return [
        TutorMatch(
            user=User(
                auth_sub=r["auth_sub"],
                role="tutor",
                name=public_name(r),
                email=r["email"],
                avatar_url=r["avatar_url"],
            ),
            public_id=r["public_id"],
            bio=r["bio"],
            subjects=r["subjects"],
            hourly_rate_sol=float(r["hourly_rate_sol"]),
            rating=float(r["rating"]),
            score=round(float(r["score"]), 4),
            style_similarity=round(float(r["style_score"]), 4),
            subject_overlap=round(float(r["subject_overlap"]), 4),
            rationale=explain(answers, r["style_affinity"] or {}),
            teaching_levels=list(r["teaching_levels"] or []),
        )
        for r in rows
    ]


def match_for_answers(
    answers: dict[str, str],
    subjects: list[str],
    *,
    max_rate: float | None = None,
    grade_level: str | None = None,
    limit: int = 3,
) -> tuple[str, list[TutorMatch]]:
    """Quiz answers straight to ranked tutors, plus the readable profile summary.

    This is the main entry point for the website.
    """
    sentence = build_profile_sentence(answers, subjects)
    return sentence, find_matches(
        answers, subjects, max_rate=max_rate, grade_level=grade_level, limit=limit
    )


# =====================================================================
# 7. CLI
# =====================================================================

# Deliberately different learner personas, with who we'd expect near the top.
# Used by `python tutormatch.py check`.
_PERSONAS = [
    (
        "Visual calculus learner",
        {"intake": "visual", "explanation": "examples_first", "pace": "moderate",
         "when_stuck": "similar_example", "structure": "agenda",
         "goal": "understand the chain rule before my midterm"},
        ["calculus"],
        "Maya Okonkwo (diagram-driven)",
    ),
    (
        "Wants to be questioned, not told",
        {"intake": "verbal", "explanation": "discovery", "pace": "slow",
         "when_stuck": "socratic", "structure": "discussion",
         "goal": "actually understand calculus instead of memorising it"},
        ["calculus"],
        "Priya Venkatesan (socratic)",
    ),
    (
        "Cramming for a chemistry exam",
        {"intake": "kinesthetic", "explanation": "examples_first", "pace": "fast",
         "when_stuck": "walkthrough", "structure": "drilling",
         "goal": "pass my chemistry final in two weeks"},
        ["chemistry"],
        "Colin Mbeki (exam drilling)",
    ),
    (
        "Nervous beginner programmer",
        {"intake": "verbal", "explanation": "analogy", "pace": "slow",
         "when_stuck": "walkthrough", "structure": "freeform",
         "goal": "learn to code from scratch, I have never programmed"},
        ["computer_science"],
        "Nadia Haddad (analogy, beginner-friendly)",
    ),
    (
        "Hands-on builder",
        {"intake": "kinesthetic", "explanation": "discovery", "pace": "fast",
         "when_stuck": "hint", "structure": "freeform",
         "goal": "build a real project and learn as I go"},
        ["computer_science"],
        "Grace Sullivan (project-based)",
    ),
]


def cmd_schema() -> int:
    """Create the database tables. Safe to re-run."""
    print("Connecting to TigerData...")
    info = ping()
    print(f"  postgres    : {str(info.get('pg_version', '?')).split(',')[0]}")
    print(f"  timescaledb : {info.get('timescaledb') or 'NOT INSTALLED'}")
    print(f"  pgvector    : {info.get('pgvector') or 'NOT INSTALLED'}")

    print("\nCreating tables...")
    apply_schema()

    tables = fetch_all(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    print(f"\nTables ({len(tables)}):")
    for t in tables:
        print(f"  - {t['table_name']}")

    hyper = fetch_all("SELECT hypertable_name FROM timescaledb_information.hypertables")
    print(f"\nHypertables ({len(hyper)}):")
    for h in hyper:
        print(f"  - {h['hypertable_name']}")

    print("\nDone. Next: python tutormatch.py seed")
    return 0


def cmd_seed() -> int:
    """Load the tutor pool and their style scores. Safe to re-run."""
    print(f"Seeding {len(TUTORS)} tutors...\n")

    for t in TUTORS:
        upsert_user(t.auth_sub, name=t.name, email=t.email,
                    avatar_url=t.avatar_url, role="tutor")
        execute(
            """
            INSERT INTO tutor_profiles
                (user_id, bio, subjects, style_affinity, hourly_rate_sol, rating, teaching_levels)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                bio             = EXCLUDED.bio,
                subjects        = EXCLUDED.subjects,
                style_affinity  = EXCLUDED.style_affinity,
                hourly_rate_sol = EXCLUDED.hourly_rate_sol,
                rating          = EXCLUDED.rating,
                teaching_levels = EXCLUDED.teaching_levels
            """,
            (t.auth_sub, t.bio, t.subjects, Json(t.style), t.hourly_rate_sol, t.rating,
             t.teaching_levels),
        )
        print(f"  {t.auth_sub}  {t.name:<20} {', '.join(t.subjects)}")

    row = fetch_one(
        "SELECT count(*) AS n FROM tutor_profiles WHERE style_affinity IS NOT NULL"
    )
    print(f"\n{row['n']} tutors have style scores.")
    print("Next: python tutormatch.py check")
    return 0


def cmd_check() -> int:
    """Show match results for five deliberately different students.

    This matters more than it looks. Every other part of the stack either works
    or throws an exception - matching can silently return plausible-looking
    garbage. If a visual learner isn't ranking the diagram-heavy tutor first,
    that tutor's style scores need adjusting.
    """
    for label, answers, subjects, expected in _PERSONAS:
        sentence, matches = match_for_answers(answers, subjects, limit=3)
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        print(f"profile: {sentence}")
        print(f"expect near top: {expected}\n")
        if not matches:
            print("  NO MATCHES - did you run `python tutormatch.py seed`?")
            continue
        for i, m in enumerate(matches, 1):
            print(f"  {i}. {m.user.name:<20} score={m.score:.3f}  "
                  f"(style={m.style_similarity:.3f} subject={m.subject_overlap:.2f} "
                  f"rating={m.rating})")

    print(f"\n{'=' * 70}")
    print("If an expected tutor isn't at or near the top, adjust that tutor's")
    print("style scores in the TUTORS list above, then re-run seed.")
    return 0


_COMMANDS = {"schema": cmd_schema, "seed": cmd_seed, "check": cmd_check}


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd not in _COMMANDS:
        print(__doc__.split("SETUP")[0].rstrip())
        print(f"\nUnknown command: {cmd!r}" if cmd else "\nNo command given.")
        print(f"Try one of: {', '.join(_COMMANDS)}")
        return 1
    if not settings.has_db:
        print("TIGER_URL is not set. Put your TigerData connection string in .env")
        return 1
    return _COMMANDS[cmd]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
