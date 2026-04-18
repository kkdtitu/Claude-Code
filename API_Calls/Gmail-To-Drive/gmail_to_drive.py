#!/usr/bin/env python3
"""
Gmail Email Summarizer — Claude API + Google Drive
----------------------------------------------------
Reads Gmail from the past 24 hours, uses Claude to categorize and summarize
each email, saves the results to a CSV, and uploads it to Google Drive.

Usage:
    python gmail_to_drive.py

Requirements:
    pip install anthropic python-dotenv google-auth-oauthlib google-api-python-client

Setup:
    1. Create a Google Cloud project:
       https://console.cloud.google.com
    2. Enable the Gmail API and Google Drive API for the project.
    3. Create an OAuth 2.0 Client ID (Application type: Desktop app).
    4. Download the JSON file → save it as 'credentials.json' in this folder.
    5. Set ANTHROPIC_API_KEY in .env.
    6. Run the script — a browser window opens for Google sign-in on first run.
"""

import anthropic
import base64
import csv
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr

from dotenv import load_dotenv

# Google APIs
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

# ── Load environment variables from .env ────────────────────────────────────
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY")
CREDENTIALS_FILE     = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE           = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")
DRIVE_FOLDER_ID      = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "").strip()
MODEL                = "claude-opus-4-6"
BATCH_SIZE           = 20   # emails analyzed per Claude API call

# Gmail (read-only) + Drive (files created by this app only)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

CATEGORIES = [
    "Work", "Finance", "Shopping", "Social", "News",
    "Newsletter", "Promotions", "Travel", "Healthcare",
    "Education", "Personal", "Spam", "Other",
]

SYSTEM_PROMPT = f"""You are an email analyst. For each email provided, assign a category and write a one-sentence summary.

Categories (choose exactly one): {", ".join(CATEGORIES)}

Respond with a JSON array only — no markdown fences, no explanation, no extra text.
Each object must have exactly these keys:
  "id"       — the email id string copied from input (do not change it)
  "category" — one category from the list above
  "summary"  — a concise one-sentence summary of the email

Example output format:
[
  {{"id": "abc123", "category": "Work", "summary": "Team standup scheduled for Monday at 10am."}},
  {{"id": "def456", "category": "Finance", "summary": "Credit card statement for March is ready to view."}}
]"""


# ── API key guard ─────────────────────────────────────────────────────────────
def check_api_key() -> None:
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found.")
        print("Add it to your .env file:")
        print("  ANTHROPIC_API_KEY=your-api-key-here")
        print("Get a key at: https://console.anthropic.com")
        sys.exit(1)


# ── Google OAuth ──────────────────────────────────────────────────────────────
def get_google_credentials() -> Credentials:
    """Authenticate with Google OAuth2. Opens browser on first run."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"\nError: '{CREDENTIALS_FILE}' not found.")
                print("Download it from the Google Cloud Console:")
                print("  1. Visit https://console.cloud.google.com")
                print("  2. APIs & Services → Credentials")
                print("  3. Create OAuth 2.0 Client ID (Desktop app)")
                print("  4. Download JSON → rename to 'credentials.json'")
                print("  5. Place it in the same folder as this script.\n")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


# ── Gmail helpers ──────────────────────────────────────────────────────────────
def decode_body(data: str) -> str:
    """Decode a base64url-encoded email body part."""
    try:
        # Correct padding before decoding
        missing = len(data) % 4
        if missing:
            data += "=" * (4 - missing)
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


def strip_html(html: str) -> str:
    """Remove HTML tags and decode common HTML entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip()


def extract_body(payload: dict, max_chars: int = 1500) -> str:
    """Recursively extract the text body from a Gmail message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return decode_body(data)[:max_chars] if data else ""

    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        return strip_html(decode_body(data))[:max_chars] if data else ""

    parts = payload.get("parts", [])
    # Prefer plain text; fall back to HTML
    for preferred_mime in ("text/plain", "text/html"):
        part = next((p for p in parts if p.get("mimeType") == preferred_mime), None)
        if part:
            return extract_body(part, max_chars)

    # Recurse into nested multipart
    for part in parts:
        body = extract_body(part, max_chars)
        if body:
            return body

    return "(no text content)"


def fetch_emails(gmail_service, hours: int = 24) -> list[dict]:
    """Return all emails received in the past `hours` hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    after_ts = int(cutoff.timestamp())
    query = f"after:{after_ts}"

    print(f"Fetching emails from the past {hours} hours...")
    emails = []
    page_token = None

    while True:
        kwargs: dict = {"userId": "me", "q": query, "maxResults": 200}
        if page_token:
            kwargs["pageToken"] = page_token

        result = gmail_service.users().messages().list(**kwargs).execute()
        messages = result.get("messages", [])

        for msg_ref in messages:
            try:
                msg = gmail_service.users().messages().get(
                    userId="me", id=msg_ref["id"], format="full"
                ).execute()

                headers = {
                    h["name"].lower(): h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }

                from_raw    = headers.get("from", "(unknown)")
                subject     = headers.get("subject", "(no subject)")
                date_str    = headers.get("date", "")
                body        = extract_body(msg.get("payload", {}))

                emails.append({
                    "id":      msg_ref["id"],
                    "date":    date_str,
                    "from":    from_raw,
                    "subject": subject,
                    "body":    body,
                })

            except HttpError as e:
                print(f"  Warning: Could not fetch message {msg_ref['id']}: {e}")

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    print(f"  Found {len(emails)} email(s).")
    return emails


# ── Claude analysis ────────────────────────────────────────────────────────────
def analyze_batch(client: anthropic.Anthropic, batch: list[dict]) -> list[dict]:
    """Send one batch of emails to Claude; return list of {id, category, summary}."""
    email_text = "\n\n---\n\n".join(
        f"ID: {e['id']}\nFrom: {e['from']}\nSubject: {e['subject']}\nBody:\n{e['body'][:600]}"
        for e in batch
    )
    user_message = f"Analyze these {len(batch)} email(s):\n\n{email_text}"

    # Stream the response; cache the system prompt across all batches
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # reused across batches
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()

    raw = next((b.text for b in response.content if b.type == "text"), "")

    # Strip markdown code fences Claude might add
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError as exc:
        print(f"  Warning: Could not parse Claude response: {exc}")
        print(f"  Response preview: {raw[:300]}")

    return []


def analyze_emails(emails: list[dict]) -> list[dict]:
    """Analyze all emails in batches; return CSV-ready row dicts."""
    client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    email_map = {e["id"]: e for e in emails}
    rows: list[dict] = []

    total_batches = (len(emails) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(emails), BATCH_SIZE):
        batch     = emails[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} — analyzing {len(batch)} email(s)...")

        try:
            results = analyze_batch(client, batch)
        except anthropic.AuthenticationError:
            print("Auth Error: Invalid ANTHROPIC_API_KEY. Check .env.")
            sys.exit(1)
        except anthropic.APIConnectionError:
            print("Connection Error: Cannot reach Anthropic API.")
            sys.exit(1)
        except anthropic.APIStatusError as e:
            print(f"API Error {e.status_code}: {e.message}")
            sys.exit(1)

        # Map Claude results back to original email data
        analyzed_ids = set()
        for item in results:
            eid = item.get("id", "")
            if eid in email_map:
                analyzed_ids.add(eid)
                orig = email_map[eid]
                rows.append({
                    "Date":     orig["date"],
                    "From":     orig["from"],
                    "Subject":  orig["subject"],
                    "Category": item.get("category", "Other"),
                    "Summary":  item.get("summary", ""),
                })

        # Fallback row for any email Claude missed
        for email in batch:
            if email["id"] not in analyzed_ids:
                rows.append({
                    "Date":     email["date"],
                    "From":     email["from"],
                    "Subject":  email["subject"],
                    "Category": "Other",
                    "Summary":  "(analysis unavailable)",
                })

    return rows


# ── CSV builder ────────────────────────────────────────────────────────────────
def build_csv(rows: list[dict]) -> str:
    """Serialize a list of row dicts to a CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["Date", "From", "Subject", "Category", "Summary"],
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ── Google Drive upload ────────────────────────────────────────────────────────
def upload_to_drive(drive_service, csv_content: str, filename: str) -> str:
    """Upload `csv_content` to Google Drive; return the file's web link."""
    file_metadata: dict = {"name": filename, "mimeType": "text/csv"}
    if DRIVE_FOLDER_ID:
        file_metadata["parents"] = [DRIVE_FOLDER_ID]

    media = MediaIoBaseUpload(
        io.BytesIO(csv_content.encode("utf-8")),
        mimetype="text/csv",
        resumable=False,
    )

    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, webViewLink",
    ).execute()

    return uploaded.get(
        "webViewLink",
        f"https://drive.google.com/file/d/{uploaded['id']}/view",
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Gmail Email Summarizer ===\n")

    # 1 — Verify Anthropic API key
    check_api_key()

    # 2 — Authenticate with Google
    print("Authenticating with Google...")
    creds         = get_google_credentials()
    gmail_service = build("gmail", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    print("  Authenticated.\n")

    # 3 — Fetch emails from past 24 hours
    emails = fetch_emails(gmail_service, hours=24)
    if not emails:
        print("No emails found in the past 24 hours. Nothing to do.")
        return
    print()

    # 4 — Categorize and summarize with Claude
    print("Analyzing emails with Claude...\n")
    rows = analyze_emails(emails)
    print()

    # 5 — Build CSV
    csv_content = build_csv(rows)
    filename    = f"email_summary_{datetime.now().strftime('%Y-%m-%d')}.csv"

    # 6 — Upload to Google Drive
    location = f"folder {DRIVE_FOLDER_ID}" if DRIVE_FOLDER_ID else "My Drive root"
    print(f"Uploading '{filename}' to Google Drive ({location})...")
    link = upload_to_drive(drive_service, csv_content, filename)

    print(f"\nDone!")
    print(f"  Emails processed : {len(rows)}")
    print(f"  File             : {filename}")
    print(f"  Google Drive link: {link}")


if __name__ == "__main__":
    main()
