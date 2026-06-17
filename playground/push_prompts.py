"""
Push local prompts from app/prompts/*.txt to LangFuse.
Run once after initial setup, then again whenever you want to reset to the local version.
Each push creates a new version in LangFuse; the 'production' label is moved to the new version.

Usage:
    uv run python playground/push_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client

PROMPTS_DIR = Path(__file__).parent.parent / "app" / "prompts"


def main() -> None:
    client = get_client()
    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        name = path.stem
        content = path.read_text().strip()
        client.create_prompt(name=name, type="text", prompt=content, labels=["production"])
        print(f"pushed: {name}")


if __name__ == "__main__":
    main()
