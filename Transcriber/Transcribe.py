import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List
 
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
 
load_dotenv()  # reads a .env file in the project root, if present
 
 
# ---------- data shapes ----------
 
class Segment(BaseModel):
    start_sec: float
    end_sec: float
    start_ts: str
    end_ts: str
    text: str
 
 
class Highlight(BaseModel):
    start_ts: str
    end_ts: str
    title: str
    note: str
 
 
class NotesResponse(BaseModel):
    highlights: List[Highlight]
 
 
# ---------- helpers ----------
 
def format_ts(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"
 
 
def ffmpeg_exe() -> str:
    """ffmpeg from PATH if installed, otherwise the copy bundled with imageio-ffmpeg,
    so a fresh laptop or server works after just `pip install -r requirements.txt`."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str, audio_path: str | None = None) -> str:
    """Pull the audio track out of the tutoring video with ffmpeg.

    The audio goes next to the video by default, so two calls being processed
    at once never write over each other's audio.
    """
    audio_path = audio_path or str(Path(video_path).with_suffix(".mp3"))
    result = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", audio_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-1500:])  # show the tail of ffmpeg's actual error
        raise RuntimeError(f"ffmpeg failed on {video_path}"
    )
    return audio_path
 
 
def upload_and_wait(client: genai.Client, path: str, max_wait_seconds: int = 180):
    """Upload via the Files API and wait until Gemini has finished processing it.

    Bails out after max_wait_seconds instead of polling forever, so a stuck
    upload fails loudly with a clear error rather than freezing the server.
    """
    uploaded = client.files.upload(file=path)
    waited = 0
    while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
        if waited >= max_wait_seconds:
            raise RuntimeError(
                f"Gemini file processing did not finish within {max_wait_seconds}s "
                f"(file: {uploaded.name}, last state: {uploaded.state.name})"
            )
        print(f"  ...still processing on Gemini's side ({waited}s elapsed)")
        time.sleep(2)
        waited += 2
        uploaded = client.files.get(name=uploaded.name)

    if getattr(uploaded, "state", None) and uploaded.state.name == "FAILED":
        raise RuntimeError(f"Gemini failed to process the uploaded file: {uploaded.name}")

    return uploaded

def _parse_offset(value) -> float:
    """Timestamps can come back as plain numbers or as duration strings
    like '1.500s' - normalize either into a float number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.endswith("s"):
        s = s[:-1]
    return float(s)
 
# ---------- step 1: word-level transcript with timestamps ----------
 
def transcribe_with_word_timestamps(client: genai.Client, audio_file) -> List[dict]:
    """
    Calls the dedicated transcription model and returns a flat list of
    {"text": ..., "start_sec": ..., "end_sec": ...} word annotations.
    """
    interaction = client.interactions.create(
        model="gemini-3.5-transcribe",
        input=[{
            "type": "audio",
            "uri": audio_file.uri,
            "mime_type": audio_file.mime_type,
        }],
        generation_config={
            "transcription_config": {
                "mode": {
                    "type": "verbatim",
                    "timestamp_granularities": ["word"],
                },
            }
        },
    )
 
    words = []
    # Word-level timing comes back as annotations of type "word_info" on the
    # interaction's content steps. Verify this path against the live docs if
    # the SDK version you install returns a different shape.
    for step in getattr(interaction, "steps", []) or []:
        for content in getattr(step, "content", []) or []:
            for ann in getattr(content, "annotations", []) or []:
                            words.append({
                "text": ann.text,
                "start_sec": _parse_offset(ann.start_offset),
                "end_sec": _parse_offset(ann.end_offset),
            })
    return words
 
 
# ---------- step 2: group words into readable segments ----------

def group_into_segments(words: List[dict], max_gap: float = 1.2, max_duration: float = 15.0) -> List[Segment]:
    """
    Turns a flat word list into sentence-ish chunks: start a new segment
    whenever there's a pause longer than max_gap seconds, or the current
    segment has already run past max_duration seconds.
    """
    if not words:
        return []
 
    segments: List[Segment] = []
    current_words = [words[0]]
 
    for prev, word in zip(words, words[1:]):
        gap = word["start_sec"] - prev["end_sec"]
        duration_so_far = word["end_sec"] - current_words[0]["start_sec"]
        if gap > max_gap or duration_so_far > max_duration:
            segments.append(_finalize_segment(current_words))
            current_words = [word]
        else:
            current_words.append(word)
 
    segments.append(_finalize_segment(current_words))
    return segments
 
 
def _finalize_segment(words: List[dict]) -> Segment:
    start = words[0]["start_sec"]
    end = words[-1]["end_sec"]
    text = " ".join(w["text"] for w in words)
    return Segment(
        start_sec=start,
        end_sec=end,
        start_ts=format_ts(start),
        end_ts=format_ts(end),
        text=text,
    )
 
 
# ---------- step 3: turn segments into timestamped study notes ----------
 
def generate_notes(client: genai.Client, segments: List[Segment]) -> List[Highlight]:
    transcript_block = "\n".join(f"[{s.start_ts}-{s.end_ts}] {s.text}" for s in segments)
 
    prompt = (
        "You are turning a tutoring session transcript into short study notes.\n"
        "Each line below is [start-end] followed by what was said.\n"
        "Produce a concise list of the key teaching moments as highlights. "
        "Each highlight must reuse the exact start_ts/end_ts (MM:SS) from the "
        "line(s) it summarizes, plus a short title and a 1-2 sentence note.\n\n"
        f"{transcript_block}"
    )
 
    # generateContent is the long-stable API (still fully supported) - used
    # here for reliable structured JSON output via response_schema.
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": NotesResponse,
        },
    )
 
    parsed = NotesResponse.model_validate_json(response.text)
    return parsed.highlights
 
 
# ---------- glue ----------

# ffmpeg's default MP3 is 128 kbit/s: 16 KB per second of audio.
AUDIO_BYTES_PER_MINUTE = 16_000 * 60


def _rate_limit_wait(err: Exception) -> float | None:
    """Seconds Gemini asks us to wait if this is a rate-limit (429) error, else None."""
    text = str(err)
    if getattr(err, "status_code", None) != 429 and "429" not in text and "quota" not in text.lower():
        return None
    match = re.search(r"retry in ([\d.]+)s", text)
    return min(float(match.group(1)) + 1, 90) if match else 30


def with_retries(step, name: str, attempts: int = 5):
    """Run one Gemini step, trying again after a timeout or a temporary error.

    Rate limits (the free tier allows only a few requests a minute) wait as long
    as Gemini asks; other errors retry after a short pause.
    """
    for attempt in range(1, attempts + 1):
        try:
            return step()
        except Exception as err:
            if attempt == attempts:
                raise
            wait = _rate_limit_wait(err)
            reason = "rate limited" if wait else f"{type(err).__name__}: {str(err)[:200]}"
            wait = wait or 2 * attempt
            print(f"  {name} failed ({reason}); retrying in {wait:.0f}s ({attempt + 1}/{attempts})...")
            time.sleep(wait)


def run_pipeline(video_path: str, output_path: str = "session_output.json") -> dict:
    """
    Runs the full pipeline on a video/audio file and returns the result dict
    ({"video_path", "segments", "highlights"}). Pass output_path=None to skip
    writing a JSON file (e.g. when called from a web endpoint).
    """
    print("Extracting audio...")
    audio_path = extract_audio(video_path)

    # A Gemini request can occasionally hang and never answer. Without a timeout
    # the call's notes would stay "processing" forever, so give every request a
    # deadline that grows with the recording's length, then retry it.
    audio_minutes = os.path.getsize(audio_path) / AUDIO_BYTES_PER_MINUTE
    timeout_sec = int(120 + 30 * audio_minutes)
    client = genai.Client(http_options={"timeout": timeout_sec * 1000})  # reads GEMINI_API_KEY from env

    try:
        print("Uploading audio to Gemini...")
        audio_file = with_retries(lambda: upload_and_wait(client, audio_path), "upload")
    finally:
        # Once uploaded, the local audio copy is no longer needed.
        if os.path.exists(audio_path):
            os.remove(audio_path)

    print(f"Transcribing with word-level timestamps (timeout {timeout_sec}s per try)...")
    words = with_retries(lambda: transcribe_with_word_timestamps(client, audio_file), "transcription")

    print(f"Got {len(words)} words. Grouping into segments...")
    segments = group_into_segments(words)

    print(f"Built {len(segments)} segments. Generating notes/highlights...")
    highlights = with_retries(lambda: generate_notes(client, segments), "notes")

    result = {
        "video_path": video_path,
        "segments": [s.model_dump() for s in segments],
        "highlights": [h.model_dump() for h in highlights],
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Done. Wrote {output_path}")

    print(f"  {len(segments)} transcript segments, {len(highlights)} highlights")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Transcribe.py path/to/video.mp4")
        sys.exit(1)
    run_pipeline(sys.argv[1])