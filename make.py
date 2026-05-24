from twilio.rest import Client
import os

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
call = client.calls.create(
    to="+923320272311",
    from_="+19129554256",
    url="https://demonstrates-tuner-necessary-writer.trycloudflare.com/incoming-call"
)
print("Calling you now!", call.sid)