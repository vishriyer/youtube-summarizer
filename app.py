"""
YouTube Video Summarizer — Streamlit web app
----------------------------------------------
A simple web UI: paste a YouTube URL, click a button, get a formatted
summary (Overview / Key Points / Takeaways). Optionally email it via Resend.

Deploy for free on Streamlit Community Cloud (share.streamlit.io):
    1. Push this file + requirements.txt to a GitHub repo.
    2. Go to share.streamlit.io, connect the repo, point it at app.py.
    3. Deploy. You get a public URL like https://yourapp.streamlit.app
    (No secrets needed — AICredits, Resend, and Supadata keys are entered
    by the user directly in the app's sidebar.)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import streamlit as st
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="📺", layout="centered")


# --------------------------------------------------------------------------
# Core logic (same as the CLI script: youtube_summarizer.py)
# --------------------------------------------------------------------------
def extract_video_id(url: str) -> str:
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        vid = parsed.path.lstrip("/")
        if vid:
            return vid
    if "youtube.com" in host:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        m = re.match(r"/(embed|shorts|live)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(2)
    raise ValueError(f"Could not extract a video ID from URL: {url}")


def fetch_video_title(video_url: str) -> str:
    """Fetch the video's title via YouTube's public oEmbed endpoint — a
    lightweight, unauthenticated metadata lookup (separate from the
    transcript-scraping endpoint that gets IP-blocked), so no API key
    or proxy is needed for this."""
    query = urllib.parse.urlencode({"url": video_url, "format": "json"})
    req = urllib.request.Request(
        f"https://www.youtube.com/oembed?{query}",
        headers={"User-Agent": "youtube-summarizer/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("title", "").strip()
    except Exception:
        return ""  # Fall back gracefully — title is a nice-to-have, not critical


SUPADATA_API_URL = "https://api.supadata.ai/v1/transcript"


def fetch_transcript(video_url: str, supadata_api_key: str, lang: str = "en", poll_timeout: int = 120) -> str:
    """Fetch a transcript via the Supadata API instead of scraping YouTube
    directly. Supadata handles IP-block avoidance and AI fallback on its
    own infrastructure, so this works reliably from cloud-hosted apps."""

    def _request(url: str):
        req = urllib.request.Request(
            url,
            headers={
                "x-api-key": supadata_api_key.strip(),
                "User-Agent": "youtube-summarizer/1.0",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Supadata API error ({e.code}): {body}")

    query = urllib.parse.urlencode({"url": video_url, "lang": lang, "text": "true"})
    status, data = _request(f"{SUPADATA_API_URL}?{query}")

    if status == 200:
        content = data.get("content", "")
        if not content:
            raise RuntimeError(
                "Supadata returned an empty transcript (no speech detected, or the video "
                "has no captions and AI transcription found nothing)."
            )
        return re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", content)

    if status == 202:
        job_id = data.get("jobId")
        if not job_id:
            raise RuntimeError(f"Supadata returned 202 but no jobId: {data}")
        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            time.sleep(1)
            _, job_data = _request(f"{SUPADATA_API_URL}/{job_id}")
            job_status = job_data.get("status")
            if job_status == "completed":
                result = job_data.get("result", {})
                content = result.get("content", "") if isinstance(result, dict) else result
                if not content:
                    raise RuntimeError("Supadata job completed but returned no transcript content.")
                return re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", content)
            if job_status == "failed":
                raise RuntimeError(f"Supadata transcription job failed: {job_data.get('error')}")
            # else: queued/active — keep polling
        raise RuntimeError("Timed out waiting for Supadata to finish transcribing this video.")

    raise RuntimeError(f"Unexpected response from Supadata (status {status}): {data}")


def chunk_text(text: str, max_chars: int = 60000):
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks, current, length = [], [], 0
    for w in words:
        current.append(w)
        length += len(w) + 1
        if length >= max_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


SUMMARY_PROMPT = """You are given the transcript of a YouTube video (auto-generated captions, so punctuation/formatting may be rough).

Produce a clear, well-structured summary in Markdown with these sections:

## Overview
A 2-4 sentence summary of what the video is about.

## Key Points
5-10 bullet points covering the main ideas, arguments, or content, in the order they appear.

## Takeaways
3-5 bullet points on the practical conclusions, lessons, or "so what" of the video — what a viewer should remember or act on.

Keep it concise and skip filler. Do not invent information that isn't in the transcript.

TRANSCRIPT:
{transcript}
"""

MAP_PROMPT = """This is one part ({part_num} of {total_parts}) of a longer video transcript (auto-generated captions).
Summarize the key content and points covered in THIS PART ONLY, in a few concise bullet points. Do not add an intro/outro, just the bullets.

PART:
{transcript}
"""

REDUCE_PROMPT = """You are given partial summaries of consecutive segments of a single YouTube video's transcript.
Combine them into one cohesive summary in Markdown with these sections:

## Overview
A 2-4 sentence summary of what the whole video is about.

## Key Points
5-10 bullet points covering the main ideas across the whole video, in order.

## Takeaways
3-5 bullet points on the practical conclusions or lessons from the video.

PARTIAL SUMMARIES:
{partials}
"""

AICREDITS_BASE_URL = "https://api.aicredits.in/v1"


def clean_header_value(value: str) -> str:
    if value is None:
        return value
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", value)
    return value.strip()


def call_llm(api_key: str, prompt: str, model: str, max_tokens: int = 2000) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=AICREDITS_BASE_URL, api_key=clean_header_value(api_key))
    resp = client.chat.completions.create(
        model=clean_header_value(model),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def summarize_transcript(transcript: str, api_key: str, model: str, progress_cb=None) -> str:
    chunks = chunk_text(transcript)

    if len(chunks) == 1:
        return call_llm(api_key, SUMMARY_PROMPT.format(transcript=chunks[0]), model, max_tokens=1500)

    partials = []
    for i, chunk in enumerate(chunks, 1):
        if progress_cb:
            progress_cb(f"Summarizing part {i}/{len(chunks)}...")
        partial = call_llm(
            api_key,
            MAP_PROMPT.format(part_num=i, total_parts=len(chunks), transcript=chunk),
            model, max_tokens=800,
        )
        partials.append(f"--- Part {i} ---\n{partial}")

    if progress_cb:
        progress_cb("Combining into final summary...")
    return call_llm(api_key, REDUCE_PROMPT.format(partials="\n\n".join(partials)), model, max_tokens=1800)


def parse_summary_sections(summary_md: str) -> dict:
    sections = {}
    current_title = "Summary"
    current_lines = []
    for line in summary_md.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line.strip())
        if m:
            if current_lines:
                sections[current_title] = current_lines
            current_title = m.group(1).strip()
            current_lines = []
        else:
            if line.strip():
                current_lines.append(line.strip())
    if current_lines:
        sections[current_title] = current_lines
    return sections


SECTION_ICONS = {"overview": "🎬", "key points": "🔑", "takeaways": "✅"}


def render_html_email(video_url: str, video_id: str, model: str, summary_md: str, video_title: str = "") -> str:
    sections = parse_summary_sections(summary_md)
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    def inline_md(text):
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)

    body_parts = []
    for title, lines in sections.items():
        icon = SECTION_ICONS.get(title.strip().lower(), "📌")
        is_bullets = all(l.startswith(("-", "*", "•")) for l in lines) and len(lines) > 1
        if is_bullets:
            items = ""
            for l in lines:
                clean_line = re.sub(r"^[-*•]\s*", "", l)
                items += f'<li style="margin:0 0 10px 0;line-height:1.55;color:#2d2d2d;">{inline_md(clean_line)}</li>'
            content_html = f'<ul style="margin:0;padding-left:20px;">{items}</ul>'
        else:
            content_html = "".join(
                f'<p style="margin:0 0 10px 0;line-height:1.6;color:#2d2d2d;">{inline_md(l)}</p>' for l in lines
            )
        body_parts.append(f'''
        <tr><td style="padding:0 0 28px 0;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-left:4px solid #d97757;background:#fff9f6;border-radius:8px;">
            <tr><td style="padding:18px 22px;">
              <div style="font-size:15px;font-weight:700;color:#1a1a1a;letter-spacing:.02em;text-transform:uppercase;margin-bottom:12px;">{icon}&nbsp;&nbsp;{title}</div>
              {content_html}
            </td></tr>
          </table>
        </td></tr>''')

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f2ede8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f2ede8;padding:32px 16px;">
<tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
<tr><td style="background:#1a1a1a;padding:0;"><img src="{thumbnail_url}" width="600" style="display:block;width:100%;max-width:600px;height:auto;" alt="Video thumbnail"></td></tr>
<tr><td style="padding:28px 28px 8px 28px;">
<div style="font-size:12px;font-weight:600;color:#d97757;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">Video Summary</div>
{f'<div style="font-size:18px;font-weight:700;color:#1a1a1a;margin-bottom:6px;line-height:1.3;">{video_title}</div>' if video_title else ''}
<a href="{video_url}" style="font-size:14px;color:#999;text-decoration:none;word-break:break-all;">{video_url}</a>
</td></tr>
<tr><td style="padding:20px 28px 6px 28px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0">{"".join(body_parts)}</table></td></tr>
<tr><td style="padding:4px 28px 28px 28px;border-top:1px solid #eee;">
<p style="margin:16px 0 0 0;font-size:12px;color:#999;line-height:1.5;">Generated automatically from the video's captions using {model} via AICredits. Some nuance may be lost — watch the full video for complete context.</p>
</td></tr>
</table></td></tr></table></body></html>'''


RESEND_API_URL = "https://api.resend.com/emails"


def send_summary_email_resend(resend_api_key, from_addr, to_addrs, subject, html_body, text_body):
    import urllib.request
    import urllib.error
    if isinstance(to_addrs, str):
        to_addrs = [to_addrs]
    payload = {"from": from_addr, "to": to_addrs, "subject": subject, "html": html_body, "text": text_body}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RESEND_API_URL, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {clean_header_value(resend_api_key)}",
            "Content-Type": "application/json",
            "User-Agent": "youtube-summarizer/1.0 (+https://resend.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Resend API error ({e.code}): {e.read().decode('utf-8', errors='replace')}")


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------


def get_secret(name: str, default=""):
    """Read from Streamlit secrets first (configured once by you in the
    Cloud dashboard), falling back to an env var, then a blank default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


st.title("📺 YouTube Video Summarizer")
st.caption("Paste a YouTube link → get an Overview, Key Points, and Takeaways.")

with st.sidebar:
    st.subheader("Settings")
    st.caption("Pre-filled from your configured secrets. Override here only if needed for this session.")
    aicredits_key_input = st.text_input(
        "AICredits API key *", type="password",
        value=get_secret("AICREDITS_API_KEY"),
        help="Get one at aicredits.in",
    )
    supadata_key_input = st.text_input(
        "Supadata API key *", type="password",
        value=get_secret("SUPADATA_API_KEY"),
        help="Get one free at supadata.ai (100 free requests/month).",
    )
    resend_key_input = st.text_input(
        "Resend API key *", type="password",
        value=get_secret("RESEND_API_KEY"),
        help="Get one free at resend.com",
    )

    st.markdown("---")
    st.subheader("Model")
    model = st.text_input("Model", value="claude-sonnet-4.5", help="Model name as listed in your AICredits catalog")

    st.markdown("---")
    st.subheader("Email this summary")
    default_sender = get_secret("RESEND_FROM", "onboarding@resend.dev")
    default_recipient = get_secret("EMAIL_TO", "")
    send_email = st.checkbox("Email me the result", value=bool(default_recipient))
    to_email = st.text_input(
        "Send to", value=default_recipient, disabled=not send_email,
        help="Without a verified domain in Resend, this can only be the email address "
             "you signed up to Resend with.",
    )
    from_addr_input = st.text_input(
        "Sender address", value=default_sender, disabled=not send_email,
        help="Defaults to Resend's shared test address.",
    )

url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
go = st.button("Summarize", type="primary", use_container_width=True)

if go:
    missing = []
    if not aicredits_key_input:
        missing.append("AICredits API key")
    if not supadata_key_input:
        missing.append("Supadata API key")
    if not resend_key_input:
        missing.append("Resend API key")
    if missing:
        st.error(f"Missing required key(s): {', '.join(missing)}. Add them in the sidebar or in app secrets.")
        st.stop()
    if not url:
        st.error("Please paste a YouTube URL.")
        st.stop()

    recipient_list = []
    if send_email:
        if not to_email:
            st.error("Please enter a recipient email address, or uncheck 'Email me the result'.")
            st.stop()
        raw_addrs = [a.strip() for a in to_email.split(",") if a.strip()]
        invalid = [a for a in raw_addrs if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", a)]
        if invalid:
            st.error(f"These don't look like valid email addresses: {', '.join(invalid)}")
            st.stop()
        recipient_list = raw_addrs

    aicredits_key = aicredits_key_input
    resend_key = resend_key_input

    try:
        with st.spinner("Extracting video ID..."):
            video_id = extract_video_id(url)
            video_url = f"https://youtu.be/{video_id}"

        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")

        video_title = fetch_video_title(video_url)
        if video_title:
            st.markdown(f"**{video_title}**")

        with st.spinner("Fetching transcript via Supadata..."):
            transcript = fetch_transcript(video_url, supadata_key_input)

        status = st.empty()
        summary = summarize_transcript(
            transcript, aicredits_key, model,
            progress_cb=lambda msg: status.info(msg),
        )
        status.empty()

        st.success("Done!")
        sections = parse_summary_sections(summary)
        for section_title, lines in sections.items():
            icon = SECTION_ICONS.get(section_title.strip().lower(), "📌")
            st.subheader(f"{icon} {section_title}")
            for line in lines:
                if line.startswith(("-", "*", "•")):
                    st.markdown(f"- {re.sub(r'^[-*•]\\s*', '', line)}")
                else:
                    st.write(line)

        st.download_button(
            "⬇️ Download summary (.md)",
            data=f"# Summary: {video_title or video_url}\n{video_url}\n\n{summary}\n",
            file_name="video_summary.md",
            mime="text/markdown",
        )

        if send_email:
            email_subject = f"Video Summary: {video_title}" if video_title else f"Video Summary: {video_url}"
            recipients_display = ", ".join(recipient_list)
            with st.spinner(f"Sending email to {recipients_display}..."):
                try:
                    html_body = render_html_email(video_url, video_id, model, summary, video_title=video_title)
                    text_body = f"{video_title or 'Video Summary'}\n{video_url}\n\n{summary}"
                    send_summary_email_resend(
                        resend_key, from_addr_input, recipient_list,
                        email_subject, html_body, text_body,
                    )
                    st.success(f"Email sent to {recipients_display}")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

    except Exception as e:
        st.error(str(e))
