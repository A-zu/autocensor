import json
from pathlib import Path
from typing import List
import ollama


def generate_keywords(
    user_input: str, model: str = "llama3.2", prompt_file: str = "prompt.txt"
) -> List[int]:
    prefix = Path(prompt_file).read_text(encoding="utf-8")
    full_prompt = f"{prefix.strip()}\n\nUser input:\n{user_input.strip()}\n\nResponse:"

    response = ollama.generate(model=model, prompt=full_prompt, stream=False)

    output = response["response"].strip()

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("Model returned invalid JSON.")

    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError("Expected a list of integers.")

    return parsed
