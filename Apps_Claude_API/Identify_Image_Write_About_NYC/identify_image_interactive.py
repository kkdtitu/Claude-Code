import anthropic
import base64
import os
from pathlib import Path
from dotenv import load_dotenv

SUPPORTED_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def identify_image(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    media_type = SUPPORTED_TYPES.get(path.suffix.lower())
    if not media_type:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {', '.join(SUPPORTED_TYPES)}")

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in your .env file.")

    client = anthropic.Anthropic(api_key=api_key)

    print("\nAnalyzing image...\n")

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "Identify and describe everything you see in this image in detail."},
                ],
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        response = stream.get_final_message()

    print()
    return next(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    load_dotenv()

    image_path = input("Enter the full path to the image file: ").strip()
    identify_image(image_path)
