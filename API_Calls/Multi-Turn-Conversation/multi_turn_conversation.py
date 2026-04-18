import json
import os
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise and accurate in your responses."""

# MODEL = "claude-opus-4-7"
MODEL = "claude-sonnet-4-6"


def create_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=api_key)


def send_message(
    client: anthropic.Anthropic,
    messages: list[dict],
    use_cache: bool = True,
) -> dict:
    """Send messages to Claude and return a JSON-formatted response dict."""

    # 1. Build the system prompt
    # If caching is enabled, wrap system prompt as a list with cache_control so the
    # API caches it server-side — subsequent turns won't re-process those tokens.
    system: list[dict] | str
    if use_cache:
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = SYSTEM_PROMPT

    # 2. Call the API
    # Sends the full conversation history plus the system prompt to Claude.
    # thinking: adaptive lets the model decide when to reason internally.
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=messages,
        thinking={"type": "adaptive"},
    )

    # 3. Shape the response into JSON
    # The API returns a content list with typed blocks (text, thinking, tool_use…).
    # Extract only the text block; fall back to empty string if none present.
    text_content = next(
        (block.text for block in response.content if block.type == "text"), ""
    )

    # Return a structured dict with the assistant reply + metadata for logging.
    return {
        "role": "assistant",
        "content": text_content,
        "metadata": {
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                # cache_creation_input_tokens: tokens written to cache (~1.25x cost)
                "cache_creation_input_tokens": getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ),
                # cache_read_input_tokens: tokens served from cache (~0.1x cost)
                "cache_read_input_tokens": getattr(
                    response.usage, "cache_read_input_tokens", 0
                ),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }


def run_conversation() -> None:
    client = create_client()
    messages: list[dict] = []
    conversation_log: list[dict] = []

    print("Multi-turn Claude Conversation (type 'quit' to exit, 'history' to view log)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "history":
            print(json.dumps(conversation_log, indent=2))
            continue

        user_turn = {
            "role": "user",
            "content": user_input,
            "metadata": {"timestamp": datetime.utcnow().isoformat() + "Z"},
        }
        conversation_log.append(user_turn)

        # Build the messages list for the API (role + content only)
        messages.append({"role": "user", "content": user_input})

        response_data = send_message(client, messages)

        # Append assistant reply to API message history
        messages.append({"role": "assistant", "content": response_data["content"]})
        conversation_log.append(response_data)

        usage = response_data["metadata"]["usage"]
        cache_read = usage["cache_read_input_tokens"]
        cache_created = usage["cache_creation_input_tokens"]

        print(f"\nAssistant: {response_data['content']}")
        print(
            f"[tokens — in: {usage['input_tokens']} | out: {usage['output_tokens']}"
            f" | cache_read: {cache_read} | cache_created: {cache_created}]\n"
        )

    # Save full conversation as JSON on exit
    output = {
        "model": MODEL,
        "total_turns": len([t for t in conversation_log if t["role"] == "user"]),
        "conversation": conversation_log,
    }
    output_path = "conversation_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nConversation saved to {output_path}")


if __name__ == "__main__":
    run_conversation()
