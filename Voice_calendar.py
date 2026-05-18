"""
Voice Calendar Booking — Streamlit App
=======================================
Install:
    pip install streamlit requests sounddevice soundfile numpy python-dotenv

Run:
    streamlit run Voice_calendar.py
"""

import os, json, time, tempfile, re, threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Calendar",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0c0e16; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

.app-header { text-align:center; padding:2rem 0 1.5rem; }
.app-header h1 { font-size:2.2rem; font-weight:600; color:#fff; letter-spacing:-0.5px; margin-bottom:0.3rem; }
.app-header p  { color:#6b7280; font-size:1rem; }

.steps-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1.2rem; justify-content:center; }
.step-pill { font-size:0.72rem; padding:4px 14px; border-radius:20px; border:1px solid #252838; background:#0c0e16; color:#4b5563; font-family:'DM Mono',monospace; }
.step-pill.active { background:#1a2150; border-color:#4f6ef7; color:#7b9fff; }
.step-pill.done   { background:#0d2b1e; border-color:#16a34a; color:#4ade80; }

.chips-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:1rem 0; }
.chip { background:#0c0e16; border:1px solid #252838; border-radius:10px; padding:12px 16px; }
.chip-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; color:#4b5563; margin-bottom:5px; }
.chip-value { font-size:0.95rem; font-weight:500; color:#e2e8f0; font-family:'DM Mono',monospace; }

.followup-box { background:#1f1500; border:1px solid #b45309; border-radius:12px; padding:18px 20px; color:#fbbf24; font-size:1rem; line-height:1.7; margin:1rem 0; }
.followup-title { font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; color:#d97706; font-weight:600; }

.success-box { background:#0a2318; border:1px solid #16a34a; border-radius:12px; padding:22px 24px; color:#4ade80; font-size:0.95rem; line-height:2; text-align:center; margin:1rem 0; }
.success-box a { color:#86efac; }

.rec-indicator { text-align:center; color:#ef4444; font-weight:600; padding:14px; border:1px solid #7f1d1d; border-radius:10px; background:#1c0a0a; margin:0.5rem 0 1rem; font-size:1rem; }

.log-box { background:#0c0e16; border:1px solid #252838; border-radius:10px; padding:14px 16px; font-family:'DM Mono',monospace; font-size:0.78rem; color:#6b7280; line-height:1.8; max-height:160px; overflow-y:auto; white-space:pre-wrap; margin-top:1rem; }

.keys-card { background:#161824; border:1px solid #252838; border-radius:16px; padding:1.5rem 1.75rem; margin-bottom:1.5rem; }

.stButton > button { width:100%; border-radius:10px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; font-size:0.95rem !important; padding:0.65rem 1rem !important; }
.stTextInput > div > div > input { background:#0c0e16 !important; border:1px solid #252838 !important; border-radius:8px !important; color:#e2e8f0 !important; font-family:'DM Mono',monospace !important; font-size:0.82rem !important; }
label { color:#9ca3af !important; font-size:0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO — singleton via @st.cache_resource + threading.Lock
#
#  WHY cache_resource:
#    Streamlit re-runs the whole script on every click. Plain module-level
#    assignments like `frames = []` reset on every rerun, wiping audio mid-
#    recording. cache_resource returns the SAME dict object every time.
#
#  WHY Lock:
#    audio_callback runs in sounddevice's background thread. The main thread
#    reads frames in stop_and_save(). The lock prevents torn reads.
#
#  WHY no time.sleep/st.rerun loop:
#    A polling loop (sleep → rerun → sleep → rerun) causes button clicks to be
#    swallowed on Windows — the rerun fires before Streamlit registers the press.
#    We show a static "recording" indicator instead; the user clicks Stop when ready.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _rec() -> dict:
    return {"frames": [], "active": False, "lock": threading.Lock()}


SAMPLE_RATE = 16_000


def audio_callback(indata, frames, time_info, status):
    """Sounddevice background thread — must never touch st.session_state."""
    r = _rec()
    if r["active"]:
        with r["lock"]:
            r["frames"].append(indata.copy())


def start_recording():
    r = _rec()
    with r["lock"]:
        r["frames"] = []
    r["active"] = True
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", callback=audio_callback,
    )
    stream.start()
    st.session_state["_stream"] = stream


def stop_and_save() -> str | None:
    r = _rec()
    r["active"] = False                        # stop appending first

    stream = st.session_state.pop("_stream", None)
    if stream:
        stream.stop()
        stream.close()

    with r["lock"]:
        frames = list(r["frames"])             # copy under lock
        r["frames"] = []                       # clear for next recording

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0)
    tmp   = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
#
#  We use a version key ("_v") to detect stale session state left over from a
#  previous version of the app. When the version changes, we wipe everything
#  and re-initialise cleanly — this is what fixes the "keys screen reappears"
#  bug caused by old step=0 surviving in session state.
# ══════════════════════════════════════════════════════════════════════════════

_APP_V = "3"

if st.session_state.get("_v") != _APP_V:
    # Wipe all keys from any previous version
    for _k in list(st.session_state.keys()):
        del st.session_state[_k]
    st.session_state["_v"] = _APP_V

_REQUIRED_ENV = ["GROQ_API_KEY", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
                 "GOOGLE_REFRESH_TOKEN", "GOOGLE_CALENDAR_ID"]

_DEFAULTS = {
    # Start at step 1 if .env is complete, else step 0 (keys entry)
    "step":               1 if all(os.getenv(k) for k in _REQUIRED_ENV) else 0,
    "extracted":          {},
    "is_followup":        False,
    "log":                [],
    "followup_q":         "",
    "saved_link":         "",
    "followup_recording": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ────────────────────────────────────────────────────────────────────

def gk(name: str) -> str:
    """Read key from session state (manual entry) OR .env. Always stripped."""
    val = st.session_state.get("key_" + name) or os.getenv(name.upper(), "")
    return val.strip() if val else ""


def log(msg: str):
    st.session_state.log.append(msg)


# ── Token cache ────────────────────────────────────────────────────────────────
_tok: dict = {"token": None, "expiry": 0}


def get_access_token() -> str:
    if _tok["token"] and time.time() < _tok["expiry"] - 60:
        return _tok["token"]
    log("🔑 Refreshing Google token...")
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
    log("✅ Token refreshed")
    return _tok["token"]


# ── Transcribe ─────────────────────────────────────────────────────────────────
def transcribe(path: str) -> str:
    log("🎙️ Transcribing with Groq Whisper...")
    with open(path, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {gk('groq_api_key')}"},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model": "whisper-large-v3", "response_format": "json"},
        )
    if not r.ok:
        raise Exception(f"Whisper error: {r.text}")
    text = r.json()["text"].strip()
    log(f'📝 You said: "{text}"')
    return text


# ── Extract ────────────────────────────────────────────────────────────────────
def extract_info(transcript: str, existing: dict | None = None) -> dict:
    log("🤖 Extracting with LLaMA 3.3...")
    today  = datetime.now().strftime("%Y-%m-%d")
    ctx    = f"\nAlready known: {json.dumps(existing)}" if existing else ""
    prompt = f"""Today is {today}. The user said: "{transcript}"{ctx}

Extract calendar event details. Reply ONLY with valid JSON, no markdown:
{{
  "purpose": "event title or null",
  "date": "YYYY-MM-DD or null",
  "time": "HH:MM 24h or null",
  "duration": "e.g. 1 hour or null",
  "missing": ["missing field names"],
  "followup_question": "short spoken question for missing info, or null if complete"
}}
Resolve relative dates from today {today}.
If purpose+date+time all present: missing=[], followup_question=null."""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {gk('groq_api_key')}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile",
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1},
    )
    if not r.ok:
        raise Exception(f"LLaMA error: {r.text}")
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── TTS ────────────────────────────────────────────────────────────────────────
def speak(text: str):
    el_key   = gk("elevenlabs_api_key")
    el_voice = gk("elevenlabs_voice_id") or "JBFqnCBsd6RMkjVDRZzb"
    if not el_key:
        log("ℹ️ TTS skipped — no ElevenLabs key")
        return
    log(f"🔊 Speaking... (key: ...{el_key[-6:]})")
    try:
        client = ElevenLabs(api_key=el_key)
        audio_generator = client.text_to_speech.convert(
            voice_id=el_voice,
            model_id="eleven_multilingual_v2",
            text=text,
            voice_settings={"stability": 0.5, "similarity_boost": 0.75},
        )
        audio_bytes = b"".join(audio_generator)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        data, sr = sf.read(tmp.name)
        sd.play(data, sr); sd.wait()
        os.unlink(tmp.name)
    except Exception as e:
        log(f"⚠️ TTS error: {e}")


# ── Save to Calendar ───────────────────────────────────────────────────────────
def save_to_calendar(ev: dict) -> dict:
    log("📅 Saving to Google Calendar...")
    token    = get_access_token()
    start_dt = datetime.fromisoformat(f"{ev['date']}T{ev['time'] or '09:00'}:00")
    dur_mins = 60
    if ev.get("duration"):
        dur = ev["duration"].lower()
        hrs  = re.search(r"(\d+\.?\d*)\s*h", dur)
        mins = re.search(r"(\d+)\s*m", dur)
        dur_mins = 0
        if hrs:  dur_mins += int(float(hrs.group(1)) * 60)
        if mins: dur_mins += int(mins.group(1))
        if not dur_mins: dur_mins = 60
    end_dt = start_dt + timedelta(minutes=dur_mins)
    body = {
        "summary": ev["purpose"],
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Karachi"},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": "Asia/Karachi"},
    }
    r = requests.post(
        f"https://www.googleapis.com/calendar/v3/calendars/{gk('google_calendar_id')}/events",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
    )
    if not r.ok:
        raise Exception(f"Calendar error: {r.text}")
    return r.json()


# ── Process audio (transcribe → extract → route) ───────────────────────────────
def process_audio(path: str):
    try:
        transcript = transcribe(path)
        try: os.unlink(path)
        except: pass

        existing = st.session_state.extracted if st.session_state.is_followup else None
        data     = extract_info(transcript, existing)

        if st.session_state.is_followup:
            for f in ["purpose", "date", "time", "duration"]:
                if data.get(f):
                    st.session_state.extracted[f] = data[f]
        else:
            st.session_state.extracted = {k: data.get(k) for k in ["purpose", "date", "time", "duration"]}

        missing = [f for f in ["purpose", "date", "time"] if not st.session_state.extracted.get(f)]

        if missing:
            q = data.get("followup_question") or \
                f"I'm missing the {' and '.join(missing)}. Could you please say that?"
            st.session_state.followup_q         = q
            st.session_state.is_followup        = True
            st.session_state.followup_recording = False
            st.session_state.step               = 4
            log(f"❓ {q}")
            speak(q)
        else:
            st.session_state.is_followup = False
            st.session_state.step        = 5
            log("✅ All details captured!")
            speak(f"Got it! Ready to save your {st.session_state.extracted.get('purpose')}.")

    except Exception as e:
        log(f"❌ {e}")
        st.session_state.step = 1


# ── Step pills ─────────────────────────────────────────────────────────────────
def pills(s: int) -> str:
    items = [(1,"Record"),(2,"Transcribe"),(3,"Extract"),(4,"Clarify?"),(5,"Save")]
    return '<div class="steps-row">' + "".join(
        f'<div class="step-pill {"done" if s>n else "active" if s==n else ""}">'
        f'{"✓ " if s>n else ""}{n}·{lbl}</div>'
        for n, lbl in items
    ) + '</div>'


def show_log():
    if st.session_state.log:
        st.markdown(
            '<div class="log-box">' + "\n".join(st.session_state.log) + '</div>',
            unsafe_allow_html=True,
        )


def show_chips():
    e = st.session_state.extracted
    st.markdown(f"""
    <div class="chips-grid">
      <div class="chip"><div class="chip-label">📌 Purpose</div><div class="chip-value">{e.get('purpose') or '—'}</div></div>
      <div class="chip"><div class="chip-label">📅 Date</div><div class="chip-value">{e.get('date') or '—'}</div></div>
      <div class="chip"><div class="chip-label">🕐 Time</div><div class="chip-value">{e.get('time') or '—'}</div></div>
      <div class="chip"><div class="chip-label">⏳ Duration</div><div class="chip-value">{e.get('duration') or '—'}</div></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🎙️ Voice Calendar</h1>
  <p>Speak your event · AI extracts details · Saves to Google Calendar</p>
</div>""", unsafe_allow_html=True)

step = st.session_state.step

# ══════════════════════
# STEP 0 — Keys entry
# Only shown when .env is missing required keys.
# Fields are ALWAYS blank — env values are never rendered on screen.
# ══════════════════════
if step == 0:
    st.markdown('<div class="keys-card">', unsafe_allow_html=True)
    st.markdown("#### 🔑 Enter your API Keys")
    st.caption("Stored only in this session. Never displayed again after saving.")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state["key_groq_api_key"]        = st.text_input("Groq API Key *",       type="password", value="", placeholder="gsk_...")
        st.session_state["key_elevenlabs_api_key"]  = st.text_input("ElevenLabs API Key",   type="password", value="", placeholder="optional")
        st.session_state["key_elevenlabs_voice_id"] = st.text_input("ElevenLabs Voice ID",               value="", placeholder="optional")
    with c2:
        st.session_state["key_google_client_id"]     = st.text_input("Google Client ID *",     type="password", value="", placeholder="xxx.apps.googleusercontent.com")
        st.session_state["key_google_client_secret"] = st.text_input("Google Client Secret *", type="password", value="", placeholder="GOCSPX-...")
        st.session_state["key_google_refresh_token"] = st.text_input("Google Refresh Token *", type="password", value="", placeholder="1//0g...")
        st.session_state["key_google_calendar_id"]   = st.text_input("Google Calendar ID *",               value="", placeholder="you@gmail.com or primary")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("✅ Save Keys & Start", type="primary", use_container_width=True):
        missing_keys = [
            lbl for lbl, k in [
                ("Groq API Key",         "key_groq_api_key"),
                ("Google Client ID",     "key_google_client_id"),
                ("Google Client Secret", "key_google_client_secret"),
                ("Google Refresh Token", "key_google_refresh_token"),
                ("Google Calendar ID",   "key_google_calendar_id"),
            ] if not st.session_state.get(k)
        ]
        if missing_keys:
            st.error(f"Please fill in: {', '.join(missing_keys)}")
        else:
            st.session_state.step = 1
            st.rerun()

# ══════════════════════
# STEP 1 — Idle / Ready
# ══════════════════════
elif step == 1:
    st.markdown(pills(1), unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0 1.5rem;">
      <div style="font-size:4rem;">🎙️</div>
      <p style="color:#9ca3af; margin-top:0.8rem; font-size:1rem;">Press record and speak your event naturally</p>
      <p style="color:#4b5563; font-size:0.85rem; margin-top:0.3rem;">
        e.g. "Schedule a dentist appointment this Friday at 3pm for 1 hour"
      </p>
    </div>""", unsafe_allow_html=True)

    if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
        st.session_state.log  = []
        st.session_state.step = 2
        start_recording()
        st.rerun()

# ══════════════════════
# STEP 2 — Recording
#
# NO auto-rerun loop here. A polling loop (sleep → rerun → sleep …) causes
# button clicks to be dropped on Windows because the rerun fires before
# Streamlit registers the press. Static indicator + manual Stop is reliable.
# ══════════════════════
elif step == 2:
    st.markdown(pills(1), unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem;">
      <div style="font-size:3rem;">⏺</div>
      <p style="color:#ef4444; font-weight:600; margin-top:0.5rem;">Recording in progress...</p>
      <p style="color:#6b7280; font-size:0.85rem;">Speak clearly, then press Stop when done</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="rec-indicator">🔴 &nbsp; Listening to your voice...</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⏹️ Stop & Process", type="secondary", use_container_width=True):
        log("⏹️ Stopped.")
        audio_path = stop_and_save()
        if audio_path:
            st.session_state["_pending"] = audio_path
            st.session_state.step = 3
        else:
            log("⚠️ No audio captured — check microphone and try again.")
            st.session_state.step = 1
        st.rerun()

# ══════════════════════
# STEP 3 — Processing
# ══════════════════════
elif step == 3:
    st.markdown(pills(2), unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:2rem 0;">
      <div style="font-size:3rem;">⚙️</div>
      <p style="color:#9ca3af; margin-top:0.8rem;">Transcribing & extracting event details...</p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Working..."):
        path = st.session_state.pop("_pending", None)
        if path:
            process_audio(path)
        else:
            st.session_state.step = 1
    st.rerun()

# ══════════════════════
# STEP 4 — Clarification
# No sidebar. No redo. Only answer the AI's question.
# ══════════════════════
elif step == 4:
    st.markdown(pills(4), unsafe_allow_html=True)

    if any(v for v in st.session_state.extracted.values()):
        show_chips()

    st.markdown(f"""
    <div class="followup-box">
      <div class="followup-title">🤖 AI needs one more detail</div>
      {st.session_state.followup_q}
    </div>""", unsafe_allow_html=True)

    if not st.session_state.followup_recording:
        if st.button("🎙️ Record Your Answer", type="primary", use_container_width=True):
            st.session_state.log                = []
            st.session_state.followup_recording = True
            start_recording()
            st.rerun()
    else:
        st.markdown(
            '<div class="rec-indicator">🔴 &nbsp; Recording... press Stop when done</div>',
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⏹️ Stop & Submit Answer", type="secondary", use_container_width=True):
            log("⏹️ Stopped.")
            audio_path = stop_and_save()
            st.session_state.followup_recording = False
            if audio_path:
                st.session_state["_pending"] = audio_path
                st.session_state.step = 3
            else:
                log("⚠️ No audio captured — please try again.")
            st.rerun()

    show_log()

# ══════════════════════
# STEP 5 — Confirm & Save
# ══════════════════════
elif step == 5:
    st.markdown(pills(5), unsafe_allow_html=True)
    show_chips()

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("📅 Save to Google Calendar", type="primary", use_container_width=True):
            try:
                saved = save_to_calendar(st.session_state.extracted)
                st.session_state.saved_link = saved.get("htmlLink", "")
                st.session_state.step = 6
                log("✅ Saved!")
                speak(f"Done! Your {st.session_state.extracted.get('purpose')} is saved to Google Calendar.")
                st.rerun()
            except Exception as ex:
                log(f"❌ {ex}")
                st.rerun()
    with col2:
        if st.button("↩ Redo", use_container_width=True):
            st.session_state.extracted   = {}
            st.session_state.is_followup = False
            st.session_state.step        = 1
            st.rerun()

    show_log()

# ══════════════════════
# STEP 6 — Success
# ══════════════════════
elif step == 6:
    e    = st.session_state.extracted
    link = st.session_state.saved_link
    st.markdown(f"""
    <div class="success-box">
      ✅ <strong>Event saved to Google Calendar!</strong><br><br>
      📌 {e.get('purpose')}<br>
      📅 {e.get('date')} &nbsp;·&nbsp; 🕐 {e.get('time')} &nbsp;·&nbsp; ⏳ {e.get('duration','1 hour')}<br><br>
      <a href="{link}" target="_blank">→ View in Google Calendar</a>
    </div>""", unsafe_allow_html=True)

    if st.button("🎙️ Book Another Event", type="primary", use_container_width=True):
        # Keep keys in session state, reset everything else
        keys_to_keep = {k: v for k, v in st.session_state.items() if k.startswith("key_")}
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.update(keys_to_keep)
        st.session_state["_v"]   = _APP_V
        st.session_state["step"] = 1
        for k, v in _DEFAULTS.items():
            if k not in st.session_state:
                st.session_state[k] = v
        st.rerun()