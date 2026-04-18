#!/usr/bin/env python3
"""
Claude API Image Identifier
----------------------------
Loads the Anthropic API key from a .env file, prompts the user for an image
file path, sends the image to the Claude API, and prints a detailed visual
analysis of the image.

Usage:
    python identify_image.py

Requirements:
    pip install anthropic python-dotenv
"""

import anthropic
import base64
import os
import sys

from dotenv import load_dotenv

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL   = "claude-opus-4-6"     # claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5-20251001

# ── Supported image formats ───────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def check_api_key() -> None:
    """Exit with a helpful message if the API key is missing."""
    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY not found.")
        print("Please add it to your .env file:")
        print("  ANTHROPIC_API_KEY=your-api-key-here")
        print("Get a key at: https://console.anthropic.com")
        sys.exit(1)


def get_image_path() -> str:
    """Prompt the user to enter the full path of the image file."""
    print("=== Claude API Image Identifier ===\n")
    image_path = input("Enter the full path to the image file: ").strip()
    return image_path


def validate_image(image_path: str) -> str:
    """
    Validate that the file exists and has a supported format.

    Args:
        image_path: Full path to the image file.

    Returns:
        The MIME media type string (e.g. 'image/jpeg').

    Raises:
        SystemExit: If the file does not exist or format is unsupported.
    """
    if not os.path.exists(image_path):
        print(f"\nError: File not found — '{image_path}'")
        print("Please check the path and try again.")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"\nError: '{image_path}' is a directory, not a file.")
        sys.exit(1)

    ext = os.path.splitext(image_path)[1].lower()
    media_type = SUPPORTED_FORMATS.get(ext)

    if not media_type:
        supported = ", ".join(SUPPORTED_FORMATS.keys())
        print(f"\nError: Unsupported format '{ext}'.")
        print(f"Supported formats: {supported}")
        sys.exit(1)

    return media_type


def load_image_as_base64(image_path: str) -> str:
    """
    Read an image file from disk and return it as a base64-encoded string.

    Args:
        image_path: Full path to the image file.

    Returns:
        Base64-encoded string of the image bytes.
    """
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def identify_image(image_path: str, media_type: str) -> str:
    """
    Send the image to Claude and return its textual analysis.

    Args:
        image_path:  Full path to the image file.
        media_type:  MIME type string (e.g. 'image/jpeg').

    Returns:
        Claude's description of the image as a plain string.
    """
    client = anthropic.Anthropic(api_key=API_KEY)
    image_data = load_image_as_base64(image_path)

    message = client.messages.create(
        model=MODEL,
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
                    {
                        "type": "text",
                        "text": (
                            "Please analyze this image in detail. "
                            "Describe what you see, including objects, colors, "
                            "scene, mood, and any other relevant details."
                        ),
                    },
                ],
            }
        ],
    )

    return message.content[0].text


def main():
    # Step 1 — Verify API key is loaded from .env
    check_api_key()

    # Step 2 — Get image path from user
    image_path = get_image_path()

    # Step 3 — Validate file and format
    media_type = validate_image(image_path)

    # Step 4 — Call Claude API
    print(f"\nAnalyzing: {image_path}")
    print("Please wait...\n")

    try:
        result = identify_image(image_path, media_type)

    except anthropic.AuthenticationError:
        print("Auth Error: Invalid API key.")
        print("Check the ANTHROPIC_API_KEY value in your .env file.")
        print("Get a key at: https://console.anthropic.com")
        sys.exit(1)

    except anthropic.APIConnectionError:
        print("Connection Error: Could not reach the Anthropic API.")
        print("Check your internet connection and try again.")
        sys.exit(1)

    except anthropic.APIStatusError as e:
        print(f"API Error {e.status_code}: {e.message}")
        sys.exit(1)

    # Step 5 — Print result
    print("── Claude's Analysis ─────────────────────────────────────────────")
    print(result)
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
