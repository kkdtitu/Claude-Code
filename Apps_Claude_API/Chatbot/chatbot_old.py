import json
import os
from datetime import datetime
import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a warm, friendly AI assistant. Keep all responses concise and conversational.

Rules you must follow in every reply:
1. Return ONLY a valid JSON object — no text outside the JSON.
   Format: {"reply": "<your response>", "follow_up": "<optional short follow-up question or empty string>"}
2. Use natural, casual language — never stiff or overly formal.
3. If the user's message looks like "hi", "hello", "hey", or similar greetings, respond with a warm greeting.
4. Before giving a long or complex answer, ask a short clarifying question first.
5. If the user seems confused or stuck, gently guide them with a motivating hint — do not guess what they mean.
6. Never produce harmful, offensive, or unsafe content. Politely decline if asked.
"""

MAX_HISTORY = 20
OUTPUT_FILE = "conversation_output.json"


def build_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
    return anthropic.Anthropic(api_key=api_key)


def chat(client: anthropic.Anthropic, full_history: list[dict], user_input: str) -> dict:
    full_history.append({"role": "user", "content": user_input})

    # Send only the most recent MAX_HISTORY messages to Claude
    window = full_history[-MAX_HISTORY:]

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=window,
        )
        raw = response.content[0].text.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"reply": raw, "follow_up": ""}

        full_history.append({"role": "assistant", "content": raw})
        return parsed

    except anthropic.RateLimitError:
        full_history.pop()
        return {
            "reply": "I'm a bit overwhelmed right now — give me a moment and try again!",
            "follow_up": "",
        }

    except anthropic.APIError as e:
        full_history.pop()
        return {
            "reply": f"Something went wrong on my end (error {e.status_code}). Please try again.",
            "follow_up": "",
        }


def save_output(full_history: list[dict]) -> None:
    output = {
        "session_end": datetime.now().isoformat(),
        "total_turns": len(full_history),
        "conversation": full_history,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nConversation saved to {OUTPUT_FILE}")


def main():
    client = build_client()
    full_history: list[dict] = []

    print("Chatbot ready! Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Take care! Talk soon.")
            break

        result = chat(client, full_history, user_input)

        print(f"Bot: {result['reply']}")
        if result.get("follow_up"):
            print(f"     → {result['follow_up']}")
        print()

    save_output(full_history)


if __name__ == "__main__":
    main()
