import anthropic
import os
from dotenv import load_dotenv

def write_paragraph_about_nyc() -> str:
    """Ask Claude to write one paragraph about NYC."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY='your-key-here'"
        )
    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": "Write one paragraph about New York City.",
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        response = stream.get_final_message()

    print()  # newline after streaming output
    return next(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    load_dotenv()   # reads .env and sets os.environ
    write_paragraph_about_nyc()
