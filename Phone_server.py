"""
phone_server.py — Twilio phone call handler with recording + transcription
Run: uvicorn Phone_server:app --host 0.0.0.0 --port 8001

Requirements:
    pip install fastapi uvicorn python-dotenv requests numpy soundfile twilio elevenlabs
"""

import os, json, tempfile, base64, asyncio
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

from core import (
    transcribe,
    extract_info,
    build_collective_question,
    check_slot_conflict,
    save_to_calendar,
    parse_duration_mins,
    tts_to_ulaw,
)

app = FastAPI()

# ── Call state (one entry per active call) ─────────────────────────────────────
call_states: dict = {}

def _new_call_state():
    return {"extracted": {}, "is_followup": False, "followup_q": ""}


# ── Helper: Build TwiML response with <Say> ────────────────────────────────────
def twiml_say(message: str, gather: bool = True, action: str = "/process-speech", record: bool = False) -> str:
    record_tag = '<Record action="/recording-complete" transcribe="true" transcribeCallback="/transcription-complete" playBeep="false"/>' if record else ""

    if gather:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    {record_tag}
    <Gather input="speech" action="{action}" method="POST" speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">{message}</Say>
    </Gather>
    <Say voice="Polly.Joanna">I didn't hear anything. Please call back and try again. Goodbye!</Say>
</Response>"""
    else:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{message}</Say>
    <Hangup/>
</Response>"""


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 1 — Incoming call webhook
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form     = await request.form()
    call_sid = form.get("CallSid", "unknown")
    call_states[call_sid] = _new_call_state()

    greeting = (
        "Welcome to the voice booking system! "
        "Please tell me what you'd like to book, the date, and the time including AM or PM."
    )
    return Response(
        content=twiml_say(
            greeting,
            gather=True,
            action=f"/process-speech?call_sid={call_sid}",
            record=True
        ),
        media_type="application/xml"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 2 — Process speech input
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/process-speech")
async def process_speech(request: Request):
    form        = await request.form()
    call_sid    = request.query_params.get("call_sid") or form.get("CallSid", "unknown")
    transcript  = form.get("SpeechResult", "").strip()

    # Get or create call state
    state = call_states.get(call_sid, _new_call_state())
    call_states[call_sid] = state

    # No speech detected
    if not transcript:
        q = state.get("followup_q") or (
            "I didn't catch that. Please tell me what you'd like to book, the date, and the time."
        )
        return Response(
            content=twiml_say(q, gather=True, action=f"/process-speech?call_sid={call_sid}"),
            media_type="application/xml"
        )

    # ── Extract booking details ────────────────────────────────────────────────
    try:
        existing   = state["extracted"] if state["is_followup"] else None
        followup_q = state["followup_q"] if state["is_followup"] else ""
        data       = extract_info(transcript, existing, followup_q)
    except Exception as e:
        return Response(
            content=twiml_say(
                "Sorry, something went wrong processing your request. Please try again.",
                gather=True,
                action=f"/process-speech?call_sid={call_sid}"
            ),
            media_type="application/xml"
        )

    # ── Merge extracted fields into state ─────────────────────────────────────
    if state["is_followup"]:
        for f in ["purpose", "date", "time", "duration"]:
            if data.get(f):
                state["extracted"][f] = data[f]
    else:
        state["extracted"] = {
            k: data.get(k) for k in ["purpose", "date", "time", "duration"]
        }

    missing = [
        f for f in ["purpose", "date", "time"]
        if not state["extracted"].get(f)
    ]

    # ── Still missing info → ask collectively ─────────────────────────────────
    if missing:
        q = build_collective_question(missing, state["extracted"])
        state["followup_q"]  = q
        state["is_followup"] = True
        return Response(
            content=twiml_say(q, gather=True, action=f"/process-speech?call_sid={call_sid}"),
            media_type="application/xml"
        )

    # ── All info collected → check conflict → save ────────────────────────────
    ev       = state["extracted"]
    dur_mins = parse_duration_mins(ev.get("duration"))

    try:
        conflicts = check_slot_conflict(ev["date"], ev["time"], dur_mins)
    except Exception:
        return Response(
            content=twiml_say(
                "I couldn't check your calendar. Please try again later.",
                gather=False
            ),
            media_type="application/xml"
        )

    if conflicts:
        names = ", ".join(conflicts)
        q = f"That slot is already taken by {names}. What other time works for you?"
        state["followup_q"]        = q
        state["is_followup"]       = True
        state["extracted"]["time"] = None
        return Response(
            content=twiml_say(q, gather=True, action=f"/process-speech?call_sid={call_sid}"),
            media_type="application/xml"
        )

    # ── Save to calendar ──────────────────────────────────────────────────────
    try:
        save_to_calendar(ev)
        done = (
            f"Done! Your {ev['purpose']} on {ev['date']} "
            f"at {ev['time']} has been booked successfully. Goodbye!"
        )
        call_states.pop(call_sid, None)
        return Response(
            content=twiml_say(done, gather=False),
            media_type="application/xml"
        )
    except Exception:
        return Response(
            content=twiml_say(
                "Sorry, I couldn't save to your calendar. Please try again.",
                gather=True,
                action=f"/process-speech?call_sid={call_sid}"
            ),
            media_type="application/xml"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 3 — Recording complete webhook
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/recording-complete")
async def recording_complete(request: Request):
    form          = await request.form()
    call_sid      = form.get("CallSid", "unknown")
    recording_sid = form.get("RecordingSid", "")
    recording_url = form.get("RecordingUrl", "")
    duration      = form.get("RecordingDuration", "0")

    print(f"\n📼 Recording complete!")
    print(f"   Call SID:      {call_sid}")
    print(f"   Recording SID: {recording_sid}")
    print(f"   Duration:      {duration}s")
    print(f"   Download URL:  {recording_url}.mp3")

    return Response(content="<?xml version='1.0'?><Response/>", media_type="application/xml")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 4 — Transcription complete webhook
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/transcription-complete")
async def transcription_complete(request: Request):
    form             = await request.form()
    call_sid         = form.get("CallSid", "unknown")
    transcription    = form.get("TranscriptionText", "")
    transcription_sid= form.get("TranscriptionSid", "")
    status           = form.get("TranscriptionStatus", "")
    recording_url    = form.get("RecordingUrl", "")

    print(f"\n📝 Transcription complete!")
    print(f"   Call SID:   {call_sid}")
    print(f"   Status:     {status}")
    print(f"   Text:       {transcription}")
    print(f"   Recording:  {recording_url}.mp3")

    # Optional: save transcription to a file
    try:
        with open("call_transcriptions.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Call SID:  {call_sid}\n")
            f.write(f"Status:    {status}\n")
            f.write(f"Recording: {recording_url}.mp3\n")
            f.write(f"Text:\n{transcription}\n")
        print(f"   Saved to call_transcriptions.txt ✅")
    except Exception as e:
        print(f"   Could not save: {e}")

    return Response(content="<?xml version='1.0'?><Response/>", media_type="application/xml")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 5 — Health check
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def health():
    return {"status": "ok", "active_calls": len(call_states)}