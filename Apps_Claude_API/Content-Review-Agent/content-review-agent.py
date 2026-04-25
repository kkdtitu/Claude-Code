import json
import os
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-opus-4-5"

# ── Helpers ────────────────────────────────────────────────────────────────────

def call_claude(system: str, messages: list[dict]) -> str:
    """Send a conversation to Claude and return the text of the first content block."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text.strip()


def parse_json_response(raw: str) -> dict:
    """
    Safely parse a JSON object from Claude's reply.
    Claude is instructed to return raw JSON, but strip fences just in case.
    """
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"answer": cleaned, "confident": False, "clarifying_questions": []}


def read_multiline_input(prompt: str) -> str:
    """Read all input until EOF (Ctrl+D on Mac/Linux, Ctrl+Z on Windows)."""
    print(prompt)
    print("(Paste your text, then press Ctrl+D on a new line when done)")
    try:
        return sys.stdin.read().strip()
    except KeyboardInterrupt:
        return ""


# ── PII Detection ──────────────────────────────────────────────────────────────

PII_SYSTEM = """
You are a strict PII detection engine.
Analyze the text the user provides and respond ONLY with a JSON object — no prose.

Format:
{
  "pii_detected": true | false,
  "pii_types": ["<type>", ...]   // empty list when false
}

PII categories to flag (non-exhaustive):
  full names, email addresses, phone numbers, physical addresses,
  Social Security Numbers (SSN), tax IDs, passport numbers,
  credit / debit card numbers, bank account numbers,
  dates of birth, medical record numbers, health / diagnosis information,
  biometric identifiers, IP addresses tied to individuals,
  usernames combined with passwords or security answers.
""".strip()


def check_for_pii(text: str) -> dict:
    raw = call_claude(PII_SYSTEM, [{"role": "user", "content": text}])
    return parse_json_response(raw)


# ── Q&A Loop ───────────────────────────────────────────────────────────────────

QA_SYSTEM = """
You are a friendly, concise content analyst.
The user will share a block of text (the "document") and then ask questions about it.

ALWAYS respond with a raw JSON object and nothing else — no markdown fences, no preamble.

Schema:
{
  "answer": "<short, friendly answer or explanation>",
  "confident": true | false,
  "clarifying_questions": ["<q1>", "<q2>"]   // 1-3 questions only when confident=false; else []
}

Rules:
- Keep "answer" to 2-3 sentences max.
- Set "confident": false when the question is ambiguous, the document lacks enough
  information, or you are genuinely unsure.
- When confident=false, populate "clarifying_questions" with 1-3 targeted questions
  that would allow you to give a better answer.
- When the question cannot be answered at all (e.g. it asks for information completely
  outside the document), set "answer" to a brief explanation of why, "confident": true,
  and "clarifying_questions": [].
""".strip()


def ask_about_content(document: str, conversation: list[dict]) -> dict:
    raw = call_claude(QA_SYSTEM, conversation)
    return parse_json_response(raw)


def build_initial_user_message(document: str, question: str) -> str:
    return f"Document:\n\"\"\"\n{document}\n\"\"\"\n\nQuestion: {question}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Content Review Agent")
    print("=" * 60)

    # 1. Collect text
    document = read_multiline_input("\n📄  Paste the text you want to analyse:")
    if not document.strip():
        print("No text provided. Exiting.")
        sys.exit(0)

    # 2. PII check
    print("\n🔍  Checking for PII …")
    pii_result = check_for_pii(document)

    if pii_result.get("pii_detected"):
        types = ", ".join(pii_result.get("pii_types", ["unknown"]))
        print(f"\n🚫  PII detected ({types}). Cannot proceed. Exiting.")
        sys.exit(1)

    print("✅  No PII detected. Content is safe to analyse.\n")

    # 3. Interactive Q&A
    satisfied_rounds = 0
    MAX_CLARIFY = 3
    MAX_SATISFIED = 3

    while satisfied_rounds < MAX_SATISFIED:
        # Get initial question
        question = input("💬  What would you like to know about this text?\n> ").strip()
        if not question:
            print("No question entered. Exiting.")
            break

        conversation = [
            {"role": "user", "content": build_initial_user_message(document, question)}
        ]

        clarify_rounds = 0

        while True:
            result = ask_about_content(document, conversation)

            answer = result.get("answer", "")
            confident = result.get("confident", True)
            clarifying_questions = result.get("clarifying_questions", [])

            # Pretty-print the JSON response
            print("\n📋  Response:")
            print(json.dumps(result, indent=2))

            if confident or clarify_rounds >= MAX_CLARIFY:
                break

            # Low confidence — ask clarifying questions (up to MAX_CLARIFY rounds)
            clarify_rounds += 1
            print(f"\n🤔  Clarifying questions (round {clarify_rounds}/{MAX_CLARIFY}):")
            for i, q in enumerate(clarifying_questions, 1):
                print(f"   {i}. {q}")

            clarification = input("\nYour clarification (or press Enter to skip): ").strip()

            conversation.append({"role": "assistant", "content": json.dumps(result)})
            if clarification:
                conversation.append({"role": "user", "content": clarification})
            else:
                conversation.append(
                    {"role": "user", "content": "Please give your best answer with what you have."}
                )

        # Satisfaction check
        satisfied_rounds += 1
        satisfied = input("\n✅  Are you satisfied with this answer? (yes/no): ").strip().lower()
        if satisfied in ("yes", "y"):
            print("\n👍  Great! Goodbye.")
            break

        if satisfied_rounds >= MAX_SATISFIED:
            print("\n⚠️   Maximum follow-up rounds reached. Goodbye.")
            break

        print(f"\n🔄  Follow-up round {satisfied_rounds}/{MAX_SATISFIED}. Ask another question.\n")


if __name__ == "__main__":
    main()
