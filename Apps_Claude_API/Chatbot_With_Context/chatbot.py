import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found.\n"
        "Create a .env file:\n  ANTHROPIC_API_KEY=sk-ant-your-key-here"
    )

MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 1024
MAX_HISTORY = 20
OUTPUT_FILE = "session_output.json"

client = anthropic.Anthropic(api_key=API_KEY)


# ── Build personalized system prompt ─────────────────────────────────────────
def build_system_prompt(user_name: str, user_profession: str) -> str:
    return f"""You are a warm, friendly AI assistant named Aria.

USER CONTEXT:
- Name      : {user_name}
- Profession: {user_profession}

Use this context naturally — tailor examples and follow-up questions to their
profession. Address the user by name occasionally to keep things personal.

PERSONALITY RULES:
1. Always respond in valid JSON only — no markdown, no code fences, no extra text.
2. Be conversational and human — avoid formal or robotic language.
3. Keep replies concise. If a long answer is needed, ask a clarifying question first.
4. If the user says hi / hello / hey, open with a warm greeting using their name.
5. If the user seems stuck, guide them gently — never guess their intent.
6. Never produce unsafe, harmful, or offensive content.
7. End with a natural follow-up question when it fits.

REQUIRED JSON FORMAT:
{{
  "reply": "your response here",
  "mood": "friendly | curious | helpful | encouraging | cautious",
  "follow_up": "follow-up question or null",
  "needs_clarification": true or false
}}

FIELD RULES:
- reply              : non-empty string, warm tone, concise
- mood               : exactly one of the five options above
- follow_up          : string or null
- needs_clarification: true when intent is unclear, false otherwise
- Never wrap in markdown fences
- Never add extra keys"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_json(raw: str) -> dict:
    """Strip accidental markdown fences and parse JSON safely."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts   = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "reply": raw,
            "mood": "cautious",
            "follow_up": None,
            "needs_clarification": False,
            "parse_error": str(e),
        }


def trim_history(history: list) -> list:
    if len(history) > MAX_HISTORY:
        print(f"  [history trimmed to last {MAX_HISTORY} messages]")
        return history[-MAX_HISTORY:]
    return history


# ── Core chat function ────────────────────────────────────────────────────────
def chat(history: list, system: str, user_message: str) -> tuple[dict, list]:
    history.append({"role": "user", "content": user_message})
    history = trim_history(history)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=history,
        )
        raw = response.content[0].text

    except anthropic.RateLimitError:
        raw = json.dumps({
            "reply": "I'm getting a lot of requests right now — give me a moment and try again!",
            "mood": "friendly",
            "follow_up": "Ready when you are — what were you asking?",
            "needs_clarification": False,
        })

    except anthropic.APIError as e:
        raw = json.dumps({
            "reply": f"Something went sideways on my end. Let's try again! (Error: {str(e)[:80]})",
            "mood": "cautious",
            "follow_up": "Want to rephrase your question?",
            "needs_clarification": False,
        })

    parsed = parse_json(raw)
    history.append({"role": "assistant", "content": raw})
    return parsed, history


# ── Display ───────────────────────────────────────────────────────────────────
def print_response(parsed: dict) -> None:
    mood      = parsed.get("mood", "friendly")
    reply     = parsed.get("reply", "")
    follow_up = parsed.get("follow_up")

    print(f"\nAria [{mood}]: {reply}")
    if follow_up:
        print(f"          >> {follow_up}")
    if "parse_error" in parsed:
        print(f"  [warning: JSON parse error — {parsed['parse_error']}]")


# ── Save session ──────────────────────────────────────────────────────────────
def save_session(user_name: str, user_profession: str, all_turns: list) -> None:
    output = {
        "session_date": datetime.now().isoformat(),
        "model":        MODEL,
        "user": {
            "name":       user_name,
            "profession": user_profession,
        },
        "total_turns":  len(all_turns),
        "conversation": all_turns,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Session saved -> {OUTPUT_FILE}  ({len(all_turns)} turns)")


# ── Startup: collect user profile ─────────────────────────────────────────────
def collect_user_profile() -> tuple[str, str]:
    print("\n  Before we start, I'd love to know a bit about you.")
    try:
        name       = input("  What's your name? ").strip()
        profession = input("  What's your profession or field? ").strip()
    except (EOFError, KeyboardInterrupt):
        name, profession = "Friend", "General"
    return name or "Friend", profession or "General"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 54)
    print("  Aria — Personalized AI Chatbot")
    print("  Type 'quit' to save and exit")
    print("=" * 54)

    user_name, user_profession = collect_user_profile()
    system = build_system_prompt(user_name, user_profession)

    print(f"\n  Great! Hi {user_name} — let's chat.")
    print("=" * 54)

    history   = []
    all_turns = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            save_session(user_name, user_profession, all_turns)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print(f"\nAria: It was lovely chatting with you, {user_name} — take care!")
            save_session(user_name, user_profession, all_turns)
            break

        parsed, history = chat(history, system, user_input)
        print_response(parsed)

        all_turns.append({
            "turn": len(all_turns) + 1,
            "user": user_input,
            "aria": parsed,
        })


if __name__ == "__main__":
    main()
