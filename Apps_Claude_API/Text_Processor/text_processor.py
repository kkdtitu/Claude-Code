import anthropic
import json
import sys
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
MAX_TOKENS  = 2048
MAX_HISTORY = 20
OUTPUT_FILE = "processing_output.json"

client = anthropic.Anthropic(api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = """You are a friendly, precise text processing assistant named Orion.

YOUR JOB:
- Receive a piece of text and an instruction
- Process the text according to the instruction
- Return a focused result in valid JSON only

PERSONALITY RULES:
1. Always respond in valid JSON only — no markdown, no code fences, no extra text.
2. Be warm and encouraging — this is a collaborative editing session.
3. If the instruction is vague or ambiguous, set needs_clarification to true and ask
   ONE focused clarifying question before processing. Do not guess.
4. If the user seems stuck, guide them with a concrete example of what they could try.
5. Keep result concise and focused — quality over length.
6. Never produce unsafe, harmful, or offensive content.

REQUIRED JSON FORMAT — follow exactly every time:
{
  "result": "the processed text, or empty string if needs_clarification is true",
  "summary": "one sentence describing what you did or what you need",
  "mood": "friendly | helpful | curious | encouraging | cautious",
  "follow_up": "a practical next-step suggestion, or null",
  "needs_clarification": true or false,
  "clarifying_question": "your single question if needs_clarification is true, else null"
}

FIELD RULES:
- result              : processed output; must be "" when needs_clarification is true
- summary             : one sentence max
- mood                : exactly one of the five options above
- follow_up           : useful suggestion or null
- needs_clarification : true only when you cannot safely infer intent
- clarifying_question : non-null only when needs_clarification is true
- Never add keys beyond the six above
- Never wrap output in markdown code fences"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_multiline(prompt: str) -> str:
    """Read multiline text until Ctrl+D (EOF). Reopens /dev/tty after."""
    print(f"\n{prompt}")
    print("  (Press Ctrl+D on Mac/Linux when done)\n")
    try:
        text = sys.stdin.read().strip()
    except EOFError:
        text = ""
    try:
        sys.stdin = open("/dev/tty")
    except OSError:
        pass
    return text


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
            "result": raw,
            "summary": "Raw response (JSON parse failed)",
            "mood": "cautious",
            "follow_up": None,
            "needs_clarification": False,
            "clarifying_question": None,
            "parse_error": str(e),
        }


def trim_history(history: list) -> list:
    if len(history) > MAX_HISTORY:
        print(f"  [history trimmed to last {MAX_HISTORY} messages]")
        return history[-MAX_HISTORY:]
    return history


# ── API call ──────────────────────────────────────────────────────────────────
def call_api(history: list, user_message: str) -> tuple[dict, list]:
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
            "result": "",
            "summary": "Rate limit reached — please wait a moment and try again.",
            "mood": "friendly",
            "follow_up": "Give it a few seconds, then re-enter your instruction.",
            "needs_clarification": False,
            "clarifying_question": None,
        })

    except anthropic.APIError as e:
        raw = json.dumps({
            "result": "",
            "summary": f"API error: {str(e)[:100]}",
            "mood": "cautious",
            "follow_up": "Try rephrasing your instruction.",
            "needs_clarification": False,
            "clarifying_question": None,
        })

    parsed = parse_json(raw)
    history.append({"role": "assistant", "content": raw})
    return parsed, history


# ── Clarification loop ────────────────────────────────────────────────────────
def handle_clarification(parsed: dict, history: list) -> tuple[dict, list]:
    """Loop until Claude no longer needs clarification."""
    while parsed.get("needs_clarification"):
        question = parsed.get("clarifying_question", "Could you clarify your instruction?")
        print(f"\nOrion [{parsed.get('mood', 'curious')}]: {question}")
        try:
            answer = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if answer.lower() == "quit":
            break
        parsed, history = call_api(history, answer)
    return parsed, history


# ── Display ───────────────────────────────────────────────────────────────────
def display_output(parsed: dict) -> None:
    mood      = parsed.get("mood", "helpful")
    summary   = parsed.get("summary", "")
    result    = parsed.get("result", "")
    follow_up = parsed.get("follow_up")

    print("\n" + "─" * 54)
    print(f"Orion [{mood}]: {summary}")
    print("─" * 54)
    if result:
        print(result)
    if follow_up:
        print(f"\n  Tip: {follow_up}")
    if "parse_error" in parsed:
        print(f"  [warning: parse error — {parsed['parse_error']}]")
    print("─" * 54)


# ── Save session ──────────────────────────────────────────────────────────────
def save_session(all_rounds: list) -> None:
    output = {
        "session_date": datetime.now().isoformat(),
        "model":        MODEL,
        "total_rounds": len(all_rounds),
        "rounds":       all_rounds,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Session saved -> {OUTPUT_FILE}  ({len(all_rounds)} rounds)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 54)
    print("  Orion — Text Processor")
    print("  Type 'quit' at any prompt to save and exit")
    print("=" * 54)

    history    = []
    all_rounds = []
    round_num  = 0

    while True:
        round_num += 1
        print(f"\n{'=' * 54}")
        print(f"  Round {round_num}")
        print(f"{'=' * 54}")

        # ── Step 1: collect text ──────────────────────────
        input_text = read_multiline("Step 1  Paste your text:")
        if not input_text or input_text.lower() == "quit":
            save_session(all_rounds)
            break

        # ── Step 2: collect instruction ───────────────────
        instruction = read_multiline(
            "Step 2  What should I do with this text?\n"
            "  e.g. summarize, shorten, make formal, extract key points, fix grammar"
        )
        if not instruction or instruction.lower() == "quit":
            save_session(all_rounds)
            break

        # ── Step 3: process ───────────────────────────────
        prompt = f"TEXT:\n{input_text}\n\nINSTRUCTION:\n{instruction}"
        parsed, history = call_api(history, prompt)
        parsed, history = handle_clarification(parsed, history)
        display_output(parsed)

        round_data = {
            "round":      round_num,
            "input_text": input_text,
            "iterations": [{"instruction": instruction, "output": parsed}],
        }

        # ── Step 4: review loop ───────────────────────────
        while True:
            try:
                choice = input("\nIs this OK? (yes / more / quit): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                all_rounds.append(round_data)
                save_session(all_rounds)
                return

            if choice == "quit":
                all_rounds.append(round_data)
                save_session(all_rounds)
                return

            elif choice == "yes":
                all_rounds.append(round_data)
                break  # next round — new text

            elif choice == "more":
                new_instruction = read_multiline(
                    "What else should I do with this text?\n"
                    "  e.g. now make it shorter, add a title, translate to bullet points"
                )
                if not new_instruction or new_instruction.lower() == "quit":
                    all_rounds.append(round_data)
                    save_session(all_rounds)
                    return

                parsed, history = call_api(history, f"INSTRUCTION:\n{new_instruction}")
                parsed, history = handle_clarification(parsed, history)
                display_output(parsed)
                round_data["iterations"].append({
                    "instruction": new_instruction,
                    "output": parsed,
                })

            else:
                print("  Please type yes, more, or quit")


if __name__ == "__main__":
    main()
