import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import quote, urlparse
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pathlib import Path
from fastapi import UploadFile, File, Form
from Transcriber.Transcribe import run_pipeline
from Transcriber.notes_file import build_notes, write_notes_files
from tutormatch import (
    save_pipeline_result,
    set_session_status,
    get_session,
    get_user,
    public_name,
    get_transcript,
    get_session_notes,
    get_session_participants,
)
from ice_servers import get_ice_servers
from websitefiles.session_bridge import user_from_cookies

app = FastAPI(title="TutorMatch Backend")


# Login lives in one place: the Flask site, through Auth0. This app never logs
# anyone in - it reads the Auth0 user from the site's session cookie, so every
# user["sub"] below is a real Auth0 ID, matching what TigerData keys users by.
def get_current_user(request: Request) -> dict | None:
    return user_from_cookies(request.cookies)


def _site_url() -> str:
    """The Flask site, where the Auth0 login routes live."""
    return os.getenv("APP_BASE_URL", "http://localhost:5000").strip().rstrip("/")


def _wrong_host(request: Request) -> RedirectResponse | None:
    """Switch to the site's hostname if the browser used a different one.

    The login cookie is only sent back to the hostname that set it, so a visit
    to 127.0.0.1 can't see a login made on localhost. Keeps port and path.
    """
    site_host = urlparse(_site_url()).hostname
    if site_host and request.url.hostname != site_host:
        return RedirectResponse(str(request.url.replace(hostname=site_host)))
    return None


def _login_redirect(request: Request, route: str = "login", back: str | None = None) -> RedirectResponse:
    """Send the browser to Auth0 login on the site, returning here afterwards."""
    return RedirectResponse(f"{_site_url()}/{route}?next={quote(back or str(request.url), safe='')}")


# The call page is served by the website but uploads its recording here. Browsers
# block that cross-address upload unless this app allows the website's origin,
# and credentials lets the login cookie come along so the upload can be checked.
# Websockets aren't subject to this, so the live chat and call need no entry.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_site_url()],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# ============================================================
# CONNECTION MANAGER
# ============================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        self.active_connections.setdefault(room_id, []).append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, message: str, room_id: str, sender: WebSocket):
        for connection in self.active_connections.get(room_id, []):
            if connection != sender:
                await connection.send_text(message)
VIDEO_FOLDER = Path(__file__).parent / "Video_Folder"
VIDEO_FOLDER.mkdir(exist_ok=True)


def _participant(request: Request, session_id: str) -> tuple[dict, dict]:
    """The logged-in user and their session, or an HTTP error if they weren't in it."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    session = get_session(session_id)
    if not session or user["sub"] not in (session.get("student_id"), session.get("tutor_id")):
        raise HTTPException(status_code=403, detail="You weren't a participant in this session")
    return user, session


def _session_folder(session: dict) -> Path:
    """Video_Folder/<room>/ - one folder per pair, holding one folder per call."""
    name = re.sub(r"[^A-Za-z0-9_-]", "_", session.get("room_key") or str(session["id"]))
    return VIDEO_FOLDER / name


def _display_name(auth_sub: str | None) -> str:
    user = get_user(auth_sub) if auth_sub else None
    return public_name(user) if user else "Unknown"


def process_recording(session_id: str, recording: Path) -> None:
    """Transcribe a call, write notes.json + notes.md beside the recording, then
    store the same notes in TigerData. Runs after the upload response is sent,
    so the server keeps serving chats and calls while Gemini works."""
    (recording.parent / "error.txt").unlink(missing_ok=True)   # clear a failure from an earlier try
    try:
        result = run_pipeline(str(recording), output_path=None)
        session = get_session(session_id) or {}
        notes = build_notes(
            result,
            session_id=session_id,
            tutor=_display_name(session.get("tutor_id")),
            student=_display_name(session.get("student_id")),
        )
        paths = write_notes_files(notes, recording.parent)
        print(f"Notes written: {paths['md']}")
        save_pipeline_result(session_id, result, recording_url=str(recording))
    except Exception as err:
        traceback.print_exc()
        (recording.parent / "error.txt").write_text(f"{type(err).__name__}: {err}\n", encoding="utf-8")
        try:
            set_session_status(session_id, "failed")
        except Exception:
            traceback.print_exc()


@app.post("/upload_recording")
async def upload_recording(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    role: str = Form(...),
):
    user, session = _participant(request, session_id)
    if role not in ("tutor", "tutee"):
        raise HTTPException(status_code=400, detail="role must be tutor or tutee")
    if user["sub"] != session.get("tutor_id" if role == "tutor" else "student_id"):
        raise HTTPException(status_code=403, detail="That isn't your role in this session")

    # A new folder for every call, so a pair's earlier notes are never overwritten.
    call_folder = _session_folder(session) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    call_folder.mkdir(parents=True, exist_ok=True)
    save_path = call_folder / f"{role}.webm"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Only the tutor's recording is transcribed - it carries both voices.
    processing = role == "tutor"
    if processing:
        set_session_status(session_id, "processing", recording_url=str(save_path))
        background.add_task(process_recording, session_id, save_path)

    return {"status": "ok", "processing": processing}

manager = ConnectionManager()


# ============================================================
# SHARED CSS
# ============================================================
SHARED_CSS = """
:root {
  --green: #25A36F;
  --teal: #069494;
  --deep: #00637c;
  --deep-2: #004a5e;
  --deep-3: #00333f;
  --text: #e8f4f5;
  --text-dim: #a8c4c8;
  --glass: rgba(255, 255, 255, 0.08);
  --glass-hover: rgba(255, 255, 255, 0.14);
  --shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
  background: linear-gradient(160deg, var(--deep) 0%, var(--deep-2) 55%, var(--deep-3) 100%);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px 16px;
}
h1, h2 {
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 8px 0;
}
h1 { font-size: 28px; }
h2 { font-size: 22px; }
p.subtitle {
  color: var(--text-dim);
  margin: 0 0 24px 0;
  font-size: 14px;
}
.card {
  background: var(--glass);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 18px;
  padding: 24px;
  box-shadow: var(--shadow);
  width: 100%;
  max-width: 760px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.badge.tutee { background: var(--green); color: white; }
.badge.tutor { background: var(--teal); color: white; }
.badge.unknown { background: #666; color: white; }
.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 18px;
}
.meta-pill {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 10px;
  padding: 6px 12px;
  font-size: 13px;
  color: var(--text-dim);
}
input[type="text"], select {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  color: var(--text);
  padding: 10px 14px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s ease, background 0.15s ease;
  width: 100%;
}
input[type="text"]:focus, select:focus {
  border-color: var(--teal);
  background: rgba(255, 255, 255, 0.12);
}
input::placeholder { color: var(--text-dim); }
button {
  border: none;
  border-radius: 10px;
  padding: 10px 18px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.08s ease, filter 0.15s ease, background 0.15s ease;
  color: white;
  background: var(--teal);
}
button:hover { filter: brightness(1.12); }
button:active { transform: scale(0.97); }
button.primary { background: var(--green); }
button.ghost {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
}
button.ghost:hover { background: var(--glass-hover); }
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  filter: none;
  transform: none;
}
.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #888;
  margin-right: 6px;
  vertical-align: middle;
}
.status-dot.live { background: var(--green); box-shadow: 0 0 8px var(--green); }
"""


# ============================================================
# 1. PRE-CALL CHAT WEBSOCKET
# ============================================================
@app.websocket("/ws/precall/{match_id}/{client_id}")
async def precall_chat(websocket: WebSocket, match_id: str, client_id: str):
    room_id = f"precall-{match_id}"
    await manager.connect(websocket, room_id)

    await manager.broadcast(
        json.dumps({"type": "user_joined", "client_id": client_id}),
        room_id,
        websocket,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") == "chat":
                await manager.broadcast(
                    json.dumps({
                        "type": "chat",
                        "client_id": client_id,
                        "content": data.get("content", ""),
                    }),
                    room_id,
                    websocket,
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        await manager.broadcast(
            json.dumps({"type": "user_left", "client_id": client_id}),
            room_id,
            websocket,
        )


# ============================================================
# 2. IN-CALL WEBSOCKET
# ============================================================
@app.websocket("/ws/call/{session_id}/{client_id}")
async def call_channel(websocket: WebSocket, session_id: str, client_id: str):
    room_id = f"call-{session_id}"
    await manager.connect(websocket, room_id)

    await manager.broadcast(
        json.dumps({"type": "user_joined", "client_id": client_id}),
        room_id,
        websocket,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type in ("offer", "answer", "ice-candidate"):
                await manager.broadcast(raw, room_id, websocket)
            elif msg_type == "chat":
                await manager.broadcast(
                    json.dumps({
                        "type": "chat",
                        "client_id": client_id,
                        "content": data.get("content", ""),
                    }),
                    room_id,
                    websocket,
                )
            elif msg_type == "screen_share_toggle":
                await manager.broadcast(
                    json.dumps({
                        "type": "screen_share_toggle",
                        "client_id": client_id,
                        "sharing": data.get("sharing", False),
                    }),
                    room_id,
                    websocket,
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        await manager.broadcast(
            json.dumps({"type": "user_left", "client_id": client_id}),
            room_id,
            websocket,
        )


# ============================================================
# 3. ROUTES
# ============================================================
@app.get("/login")
async def login(request: Request):
    """Log in through Auth0 on the main site, then return to this app's home."""
    if (bounce := _wrong_host(request)) is not None:
        return bounce
    return _login_redirect(request, back=str(request.url.replace(path="/", query="")))


@app.get("/signup")
async def signup(request: Request):
    """Sign up through Auth0 on the main site, then return to this app's home."""
    if (bounce := _wrong_host(request)) is not None:
        return bounce
    return _login_redirect(request, route="signup", back=str(request.url.replace(path="/", query="")))


@app.get("/logout")
async def logout():
    """Log out of Auth0 through the main site."""
    return RedirectResponse(f"{_site_url()}/logout")

# The home, chat and call pages now live on the website, which links two real
# people into a shared room. Anyone arriving at the old addresses goes there.
@app.get("/")
@app.get("/precall")
@app.get("/call")
async def moved_to_site():
    return RedirectResponse(_site_url())


@app.get("/ice-servers")
def ice_servers():
    """STUN + TURN servers for the call page. Credentials come from Cloudflare."""
    return {"iceServers": get_ice_servers()}


@app.get("/transcript/{session_id}")
async def transcript_page(session_id: str, request: Request):
    if (bounce := _wrong_host(request)) is not None:
        return bounce
    user = get_current_user(request)
    if not user:
        return _login_redirect(request)
    participants = get_session_participants(session_id) or {}
    if user["sub"] not in (participants.get("student_id"), participants.get("tutor_id")):
        raise HTTPException(status_code=403, detail="You weren't a participant in this session")
    return HTMLResponse(TRANSCRIPT_HTML)


def _latest_notes_folder(session: dict) -> Path | None:
    """The most recent call folder that has finished notes. Folder names are
    UTC timestamps, so sorting them by name sorts them by time."""
    folder = _session_folder(session)
    calls = sorted((p for p in folder.glob("*/notes.json")), key=lambda p: p.parent.name) if folder.is_dir() else []
    return calls[-1].parent if calls else None


@app.get("/api/transcript/{session_id}")
async def api_transcript(session_id: str, request: Request):
    _, session = _participant(request, session_id)
    status = session.get("status")
    segments = get_transcript(session_id)
    notes = get_session_notes(session_id)
    return {
        "session_id": session_id,
        "status": status,
        # While a newer call is being transcribed, don't show the previous one as if it were done.
        "ready": bool(segments) and status not in ("processing", "failed"),
        "segments": segments,
        "notes": notes,
        "has_notes_file": _latest_notes_folder(session) is not None,
    }


@app.get("/notes/{session_id}/{fmt}")
async def download_notes(session_id: str, fmt: str, request: Request):
    """Download the latest call's notes file: /notes/<room>/md or /notes/<room>/json."""
    if fmt not in ("md", "json"):
        raise HTTPException(status_code=404)
    _, session = _participant(request, session_id)
    folder = _latest_notes_folder(session)
    if folder is None:
        raise HTTPException(status_code=404, detail="No notes have been written for this session yet")
    return FileResponse(
        folder / f"notes.{fmt}",
        media_type="text/markdown; charset=utf-8" if fmt == "md" else "application/json",
        filename=f"tutoring-notes-{folder.name}.{fmt}",
    )

TRANSCRIPT_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>TutorMatch - Transcript</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
""" + SHARED_CSS + """
.card.wide { max-width: 900px; }
.moment {
  border-left: 3px solid var(--green);
  padding: 10px 14px;
  margin-bottom: 10px;
  background: rgba(255,255,255,0.04);
  border-radius: 8px;
}
.moment .t { color: var(--green); font-weight: 600; margin-right: 8px; }
.segment { padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
.segment .t { color: #9fd; opacity: 0.8; margin-right: 8px; font-variant-numeric: tabular-nums; }
#status { opacity: 0.8; margin-bottom: 16px; }
a.btn {
  display: inline-block; margin: 0 8px 8px 0; padding: 10px 18px; border-radius: 10px;
  background: var(--green); color: white; font-size: 14px; font-weight: 600; text-decoration: none;
}
a.btn + a.btn { background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.15); }
a.btn:hover { filter: brightness(1.12); }
</style>
</head>
<body>
<div class="card wide">
  <h2>Session Transcript</h2>
  <div id="status">Loading...</div>
  <div id="downloads" style="display:none; margin-bottom: 16px;">
    <a class="btn" id="downloadMd">Download notes</a>
    <a class="btn" id="downloadJson">Download notes (JSON)</a>
  </div>
  <div id="notesSection" style="display:none;">
    <h3>Key parts</h3>
    <div id="moments"></div>
  </div>
  <div id="transcriptSection" style="display:none;">
    <h3>Full Transcript</h3>
    <div id="segments"></div>
  </div>
</div>

<script>
function fmt(ms) {
  const total = Math.floor((ms || 0) / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// Transcript text is whatever was said on the call - never insert it as HTML.
function esc(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

async function load() {
  const sessionId = window.location.pathname.split('/').pop();
  const statusEl = document.getElementById('status');
  try {
    const resp = await fetch(`/api/transcript/${sessionId}`);
    const data = await resp.json();

    if (data.status === 'failed') {
      statusEl.textContent = "Sorry - this call's notes couldn't be written. The recording is saved, so they can be re-run.";
      return;
    }
    if (!data.ready) {
      statusEl.textContent = 'Writing your notes - this usually takes about a minute...';
      setTimeout(load, 4000);
      return;
    }

    statusEl.style.display = 'none';

    if (data.has_notes_file) {
      document.getElementById('downloadMd').href = `/notes/${sessionId}/md`;
      document.getElementById('downloadJson').href = `/notes/${sessionId}/json`;
      document.getElementById('downloads').style.display = 'block';
    }

    const moments = (data.notes && data.notes.key_moments) || [];
    if (moments.length) {
      document.getElementById('notesSection').style.display = 'block';
      document.getElementById('moments').innerHTML = moments.map(m => `
        <div class="moment"><span class="t">${fmt(m.tMs)}</span><strong>${esc(m.title)}</strong><div>${esc(m.why)}</div></div>
      `).join('');
    }

    const segments = data.segments || [];
    if (segments.length) {
      document.getElementById('transcriptSection').style.display = 'block';
      document.getElementById('segments').innerHTML = segments.map(s => `
        <div class="segment"><span class="t">${fmt(s.start_ms)}</span>${esc(s.text)}</div>
      `).join('');
    }

    if (!moments.length && !segments.length) {
      statusEl.style.display = 'block';
      statusEl.textContent = 'No transcript found for this session.';
    }
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Could not load transcript.';
  }
}

load();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))