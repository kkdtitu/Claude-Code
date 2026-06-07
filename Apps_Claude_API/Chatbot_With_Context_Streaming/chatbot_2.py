"""
AI Chatbot — powered by Anthropic Claude
Streaming · Multi-turn history · User profiling · JSON output
"""

import anthropic
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MAX_HISTORY = 20        # keep last 20 messages (10 user + 10 assistant)
MODEL       = "claude-opus-4-5"
MAX_TOKENS  = 1024

SYSTEM_PROMPT = """You are a friendly, helpful AI assistant. You are talking to {name}, who works as a {profession}.

RESPONSE RULES — follow all of these on every single turn:

1. OUTPUT FORMAT: Respond ONLY with a valid JSON object. No markdown, no code fences, no extra text outside the JSON.
   Exact schema:
   {{"message": "<your reply>", "follow_up": "<optional short question or empty string>", "tip": "<optional encouragement or empty string>"}}

2. TONE: Be warm, conversational, and human. Avoid jargon and overly formal language. Write as if texting a knowledgeable friend.

3. GREETING: If the user's first message is a greeting (hi, hello, hey, etc.), respond with a warm, personalised greeting that mentions their name and role.

4. FOLLOW-UP QUESTIONS: Before giving a long or complex answer, ask ONE focused clarifying question in "follow_up" to make sure you understand what they need.

5. GUIDE STUCK USERS: If the user seems lost, confused, or unsure what to ask, gently guide and motivate them. Do NOT guess or assume. Use "follow_up" to nudge them.

6. SAFETY: Never produce harmful, offensive, misleading, or dangerous content. Politely decline and redirect if needed.

7. BREVITY: Keep "message" focused and concise. No walls of text.

8. PERSONALISATION: Reference {name}'s profession ({profession}) naturally when it adds value.
"""


def trim_history(history: list[dict]) -> list[dict]:
    return history[-MAX_HISTORY:]


def stream_response(system_prompt: str, history: list[dict]) -> dict | None:
    full_text = ""
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=trim_history(history),
        ) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                full_text += chunk
        print()
        return json.loads(full_text.strip())

    except anthropic.RateLimitError:
        print("\n[Rate limit — waiting 30 s and retrying...]")
        time.sleep(30)
        try:
            full_text = ""
            with client.messages.stream(
                model=MODEL, max_tokens=MAX_TOKENS,
                system=system_prompt, messages=trim_history(history),
            ) as stream:
                for chunk in stream.text_stream:
                    print(chunk, end="", flush=True)
                    full_text += chunk
            print()
            return json.loads(full_text.strip())
        except Exception as e:
            print(f"\n[Retry failed: {e}]")
            return None

    except anthropic.APIError as e:
        print(f"\n[API error: {e}. Please try again.]")
        return None

    except json.JSONDecodeError:
        print("\n[Response was not valid JSON — showing raw output.]")
        return {"message": full_text.strip(), "follow_up": "", "tip": ""}

    except Exception as e:
        print(f"\n[Unexpected error: {e}]")
        return None


def collect_user_profile() -> tuple[str, str]:
    print("\n👋  Hi there! Before we start, I'd love to get to know you a little.\n")
    name = input("   What's your name? → ").strip() or "friend"
    profession = input(f"\n   Nice to meet you, {name}! What do you do for a living? → ").strip() or "professional"
    return name, profession


def run_chatbot() -> None:
    name, profession = collect_user_profile()
    system_prompt = SYSTEM_PROMPT.format(name=name, profession=profession)
    history: list[dict] = []

    print(f"\n✅  All set, {name}! Type 'quit', 'exit', or 'bye' to leave.\n")
    print("─" * 60)

    while True:
        try:
            user_input = input(f"\n{name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAssistant: Goodbye! 👋\n")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nAssistant: It was great chatting! Take care. 👋\n")
            break

        history.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end="", flush=True)

        response_data = stream_response(system_prompt, history)

        if response_data is None:
            history.pop()   # keep history consistent on failure
            continue

        message   = response_data.get("message", "").strip()
        follow_up = response_data.get("follow_up", "").strip()
        tip       = response_data.get("tip", "").strip()

        if follow_up:
            print(f"\n   ❓ {follow_up}")
        if tip:
            print(f"   💡 {tip}")

        full_reply = message
        if follow_up:
            full_reply += f"\n\n{follow_up}"
        if tip:
            full_reply += f"\n\n💡 {tip}"

        history.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    run_chatbot()
