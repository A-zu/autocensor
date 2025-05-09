import json
from pathlib import Path
from typing import List
import ollama


def generate_keywords(
    user_input: str, model: str = "qwen3:4b", prompt_file: str = "prompt.txt"
) -> List[int]:
    prefix = Path(prompt_file).read_text(encoding="utf-8")
    full_prompt = f"{prefix.strip()}\n\nUser input:\n{user_input.strip()}\n\nResponse:"

    response = ollama.generate(
        model=model, prompt=full_prompt, stream=False, keep_alive="5m"
    )

    output = response["response"].strip()

    think, output = output.split("</think>")
    # Clean model output (handle code block formatting)
    if output.startswith("\n\n```"):
        output = output.strip("`").strip()
        if output.lower().startswith("json"):
            output = output[4:].strip()

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("Model returned invalid JSON.")

    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError("Expected a list of integers.")

    return parsed
