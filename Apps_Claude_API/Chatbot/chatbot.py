import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# ── Load API key ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found.\n"
        "Create a .env file with:\n  ANTHROPIC_API_KEY=sk-ant-your-key-here"
    )

MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 1024
MAX_HISTORY = 20
OUTPUT_FILE = "conversation_output.json"

client = anthropic.Anthropic(api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = """You are a warm, friendly AI assistant named Aria.

PERSONALITY RULES:
1. Always respond in valid JSON only — no markdown, no code fences, no extra text.
2. Be conversational and human — avoid formal or robotic language.
3. Keep replies concise and focused. If a long answer is needed, ask a follow-up question first.
4. If the user says hi / hello / hey, respond with a warm, personalized greeting.
5. If the user seems stuck or confused, gently guide and motivate them — never guess their intent.
6. Never produce unsafe, harmful, or offensive content.
7. End with a friendly follow-up question whenever it feels natural.

REQUIRED JSON FORMAT — every response must follow this exactly:
{
  "reply": "your conversational response here",
  "mood": "friendly | curious | helpful | encouraging | cautious",
  "follow_up": "a short follow-up question, or null",
  "needs_clarification": true or false
}

FIELD RULES:
- reply              : non-empty string, conversational tone, not too long
- mood               : exactly one of the five options listed above
- follow_up          : string when a follow-up feels natural, otherwise null
- needs_clarification: true when user intent is unclear, false otherwise
- Never wrap output in markdown code fences
- Never add keys outside the four defined above"""


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
    """Keep only the most recent MAX_HISTORY messages."""
    if len(history) > MAX_HISTORY:
        print(f"  [history trimmed to last {MAX_HISTORY} messages]")
        return history[-MAX_HISTORY:]
    return history


# ── Core chat function ────────────────────────────────────────────────────────
def chat(history: list, user_message: str) -> tuple[dict, list]:
    """Append user message, call API, return (parsed_response, updated_history)."""
    history.append({"role": "user", "content": user_message})
    history = trim_history(history)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=history,
        )
        raw = response.content[0].text

    except anthropic.RateLimitError:
        raw = json.dumps({
            "reply": "I'm getting a lot of requests right now — give me just a moment and try again!",
            "mood": "friendly",
            "follow_up": "Ready when you are — what were we talking about?",
            "needs_clarification": False,
        })

    except anthropic.APIError as e:
        raw = json.dumps({
            "reply": f"Something went sideways on my end. Let's try again! (Error: {str(e)[:80]})",
            "mood": "cautious",
            "follow_up": "Want to rephrase or try a different question?",
            "needs_clarification": False,
        })

    parsed = parse_json(raw)
    history.append({"role": "assistant", "content": raw})
    return parsed, history


# ── Print response ────────────────────────────────────────────────────────────
def print_response(parsed: dict) -> None:
    reply     = parsed.get("reply", "")
    follow_up = parsed.get("follow_up")
    mood      = parsed.get("mood", "friendly")

    print(f"\nAria [{mood}]: {reply}")
    if follow_up:
        print(f"          >> {follow_up}")
    if "parse_error" in parsed:
        print(f"  [warning: JSON parse error — {parsed['parse_error']}]")


# ── Save session ──────────────────────────────────────────────────────────────
def save_session(all_turns: list) -> None:
    output = {
        "session_date": datetime.now().isoformat(),
        "model":        MODEL,
        "total_turns":  len(all_turns),
        "conversation": all_turns,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Session saved to {OUTPUT_FILE}  ({len(all_turns)} turns)")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Aria — Friendly AI Chatbot")
    print("  Commands: 'quit' or 'exit' to save and leave")
    print("=" * 52)

    history   = []
    all_turns = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            save_session(all_turns)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nAria: It was so nice chatting with you — take care!")
            save_session(all_turns)
            break

        parsed, history = chat(history, user_input)
        print_response(parsed)

        all_turns.append({
            "turn": len(all_turns) + 1,
            "user": user_input,
            "aria": parsed,
        })


if __name__ == "__main__":
    main()
