"""
core.py — Shared booking logic (used by both Streamlit UI and phone server)
"""

import os, json, time, tempfile, re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Helpers ────────────────────────────────────────────────────────────────────
def gk(name: str) -> str:
    return os.getenv(name.upper(), "").strip()

# ── Token cache ────────────────────────────────────────────────────────────────
_tok = {"token": None, "expiry": 0}

def get_access_token() -> str:
    if _tok["token"] and time.time() < _tok["expiry"] - 60:
        return _tok["token"]
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id":     gk("google_client_id"),
        "client_secret": gk("google_client_secret"),
        "refresh_token": gk("google_refresh_token"),
        "grant_type":    "refresh_token",
    })
    if not r.ok:
        raise Exception(f"Token refresh failed: {r.text}")
    d = r.json()
    _tok["token"]  = d["access_token"]
    _tok["expiry"] = time.time() + d["expires_in"]
    return _tok["token"]

# ── Transcribe ─────────────────────────────────────────────────────────────────
def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {gk('groq_api_key')}"},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model": "whisper-large-v3", "response_format": "json"},
        )
    if not r.ok:
        raise Exception(f"Whisper error: {r.text}")
    return r.json()["text"].strip()

# ── Day name → next occurrence date ───────────────────────────────────────────
def _next_weekday(day_name: str, today: datetime) -> str:
    """Return YYYY-MM-DD of the next occurrence of the named weekday."""
    days = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    target = days.get(day_name.lower())
    if target is None:
        return None
    current_weekday = today.weekday()
    days_ahead = (target - current_weekday) % 7
    if days_ahead == 0:
        days_ahead = 7  # next week if today is the same day
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

# ── Extract ────────────────────────────────────────────────────────────────────
def extract_info(transcript: str, existing: dict = None, followup_q: str = "") -> dict:
    now      = datetime.now()
    today    = now.strftime("%Y-%m-%d")
    weekday  = now.strftime("%A")  # e.g. "Friday"
    ctx      = f"\nAlready known: {json.dumps(existing)}" if existing else ""
    fctx     = f'\nYou previously asked the user: "{followup_q}"\nThe user is directly answering that question.' if followup_q else ""

    prompt = f"""You are an AI assistant for a BOOKING APP (doctor appointments, meetings, salon visits, etc).
Today is {today} ({weekday}). Current time is {now.strftime("%H:%M")}.
The user said: "{transcript}"{ctx}{fctx}

CRITICAL DATE RULES:
- Today is {today} which is a {weekday}.
- "Sunday" means the NEXT Sunday from today. Count carefully from {weekday}.
- "Monday" means the NEXT Monday from today. Count carefully from {weekday}.
- NEVER confuse day names. If user says "Sunday", the date MUST be a Sunday.
- To verify: after computing the date, double-check that day name matches what user said.
- "tomorrow" = {(now + timedelta(days=1)).strftime("%Y-%m-%d")}
- "day after tomorrow" = {(now + timedelta(days=2)).strftime("%Y-%m-%d")}
- Next weekday dates from today ({weekday} {today}):
  * Monday = {_next_weekday("monday", now)}
  * Tuesday = {_next_weekday("tuesday", now)}
  * Wednesday = {_next_weekday("wednesday", now)}
  * Thursday = {_next_weekday("thursday", now)}
  * Friday = {_next_weekday("friday", now)}
  * Saturday = {_next_weekday("saturday", now)}
  * Sunday = {_next_weekday("sunday", now)}

CRITICAL TIME RULES:
- If user says ONLY a number like "3", "at 3", "3 o'clock" with NO am/pm context → set time=null, ask "Is that 3 AM or 3 PM?"
- NEVER guess or assume AM or PM when not stated.
- NEVER convert 3 to 10:00 or 11:00 or any other time. Use the EXACT number the user said.
- "3 PM" = 15:00, "3 AM" = 03:00, "morning" = AM, "afternoon/evening/night" = PM
- "noon" = 12:00, "midnight" = 00:00
- Only set time if user CLEARLY states AM/PM or context makes it obvious.

BOOKING RULES:
- Clean up titles: "show doctor apartment" → "Doctor Appointment", "hair cut" → "Haircut"
- Capitalize titles properly.
- If followup context exists, interpret transcript AS AN ANSWER to that question.

Extract calendar event details. Reply ONLY with valid JSON, no markdown:
{{
  "purpose": "clean booking title or null",
  "date": "YYYY-MM-DD or null",
  "time": "HH:MM 24h or null",
  "duration": "e.g. 1 hour or null",
  "missing": ["missing field names"],
  "followup_question": "short spoken question for missing info, or null if complete"
}}

If purpose+date+time all present: missing=[], followup_question=null."""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {gk('groq_api_key')}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # 0 for maximum accuracy
        },
    )
    if not r.ok:
        raise Exception(f"LLaMA error: {r.text}")
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    # ── Safety check: verify day name matches computed date ────────────────────
    if data.get("date"):
        try:
            computed_date = datetime.strptime(data["date"], "%Y-%m-%d")
            computed_day  = computed_date.strftime("%A").lower()
            transcript_lower = transcript.lower()
            # Check all day names
            day_names = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            for day in day_names:
                if day in transcript_lower and computed_day != day:
                    # Fix it using our reliable _next_weekday function
                    corrected = _next_weekday(day, now)
                    if corrected:
                        data["date"] = corrected
                    break
        except Exception:
            pass

    return data

# ── Build collective question ──────────────────────────────────────────────────
def build_collective_question(missing: list, existing: dict) -> str:
    parts = []
    if "purpose" in missing: parts.append("what type of appointment is this")
    if "date"    in missing: parts.append("which date")
    if "time"    in missing: parts.append("what time including AM or PM")
    if not parts:
        return "Could you repeat the details please?"
    if len(parts) == 1:
        q = parts[0].capitalize() + "?"
    elif len(parts) == 2:
        q = f"{parts[0].capitalize()} and {parts[1]}?"
    else:
        q = ", ".join(p for p in parts[:-1]).capitalize() + f", and {parts[-1]}?"
    known = existing.get("purpose")
    return f"For your {known}, could you tell me: {q}" if known else f"Could you tell me: {q}"

# ── Parse duration string → minutes ───────────────────────────────────────────
def parse_duration_mins(duration_str: str) -> int:
    if not duration_str:
        return 60
    dur  = duration_str.lower()
    hrs  = re.search(r"(\d+\.?\d*)\s*h", dur)
    mins = re.search(r"(\d+)\s*m", dur)
    total = 0
    if hrs:  total += int(float(hrs.group(1)) * 60)
    if mins: total += int(mins.group(1))
    return total if total else 60

# ── Check slot conflict ────────────────────────────────────────────────────────
def check_slot_conflict(date: str, time_str: str, duration_mins: int = 60) -> list:
    token    = get_access_token()
    start_dt = datetime.fromisoformat(f"{date}T{time_str}:00")
    end_dt   = start_dt + timedelta(minutes=duration_mins)
    cal_id   = gk("google_calendar_id") or "primary"
    r = requests.get(
        f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "timeMin":      start_dt.strftime("%Y-%m-%dT%H:%M:%S+05:00"),
            "timeMax":      end_dt.strftime("%Y-%m-%dT%H:%M:%S+05:00"),
            "singleEvents": "true",
            "orderBy":      "startTime",
        },
    )
    if not r.ok:
        raise Exception(f"Conflict check error: {r.text}")
    return [e.get("summary", "Unnamed event") for e in r.json().get("items", [])]

# ── Save to Google Calendar ────────────────────────────────────────────────────
def save_to_calendar(ev: dict) -> dict:
    token    = get_access_token()
    start_dt = datetime.fromisoformat(f"{ev['date']}T{ev['time'] or '09:00'}:00")
    dur_mins = parse_duration_mins(ev.get("duration"))
    end_dt   = start_dt + timedelta(minutes=dur_mins)
    cal_id   = gk("google_calendar_id") or "primary"
    r = requests.post(
        f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "summary": ev["purpose"],
            "start": {"dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S+05:00"), "timeZone": "Asia/Karachi"},
            "end":   {"dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S+05:00"),   "timeZone": "Asia/Karachi"},
        },
    )
    if not r.ok:
        raise Exception(f"Calendar error: {r.text}")
    return r.json()

# ── TTS → µ-law 8kHz bytes for phone ──────────────────────────────────────────
def tts_to_ulaw(text: str) -> bytes:
    el_key   = gk("elevenlabs_api_key")
    el_voice = gk("elevenlabs_voice_id") or "JBFqnCBsd6RMkjVDRZzb"
    if not el_key:
        return b""
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=el_key)
    gen = client.text_to_speech.convert(
        voice_id=el_voice,
        model_id="eleven_multilingual_v2",
        text=text,
        voice_settings={"stability": 0.5, "similarity_boost": 0.75},
        output_format="ulaw_8000",
    )
    return b"".join(gen)