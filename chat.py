import json
import logging
import time
from pathlib import Path

import fitz
import ollama

logger = logging.getLogger(__name__)


def clean_output(response):
    output = response["response"].strip()
    think, output = output.split("</think>")
    # Clean model output (handle code block formatting)
    if output.startswith("\n\n```"):
        output = output.strip("`").strip()
        if output.lower().startswith("json"):
            output = output[4:].strip()

    return output


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)  # crude approximation: 1 token ≈ 4 characters


def chunk_text(text: str, max_tokens: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        temp_chunk = f"{current_chunk}\n\n{para}".strip()
        if estimate_token_count(temp_chunk) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

        else:
            current_chunk = temp_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_with_throttle(model: str, prompt: str):
    full_response = ""
    buffer = ""
    last_log_time = time.time()

    stream = ollama.generate(
        model=model,
        prompt=prompt,
        stream=True,
    )

    for chunk in stream:
        text = chunk.get("response")
        buffer += text
        full_response += text

        now = time.time()
        if now - last_log_time >= 5:
            logger.debug(
                f"Partial response chunk:\n{buffer}",
            )
            buffer = ""
            last_log_time = now

    return {"response": full_response}


def get_redactions(
    user_prompt: str,
    pdf_path: str,
    model: str,
    prompt_file: str = "redact_prompt.txt",
) -> set[str]:
    # 1. Extract text from PDF
    def extract_text(pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())

        doc.close()
        return "\n".join(full_text)

    pdf_text = extract_text(pdf_path)

    # 2. Load system prompt
    system_prompt = Path(prompt_file).read_text(encoding="utf-8").strip()

    # 3. Prepare chunking
    max_context_tokens = 40000
    buffer_tokens = 1500  # system + user prompt overhead
    max_doc_tokens = max_context_tokens - buffer_tokens

    chunks = chunk_text(pdf_text, max_doc_tokens)
    redaction_results = []

    # 4. Generate per chunk
    for i, chunk in enumerate(chunks):
        full_prompt = (
            f"{system_prompt.strip()}\n"
            "Document Content:\n"
            f"{chunk}\n"
            "User Instruction:\n"
            f"{user_prompt.strip()}\n"
        )

        logger.info(f"User instruction: {user_prompt.strip()}")
        response = generate_with_throttle(model, full_prompt)

        output = clean_output(response)
        logger.debug(f"Model output: {output.strip()}")

        result = json.loads(output)
        if not isinstance(result, list):
            raise ValueError("Response was not a list")
        redaction_results.extend(result)

    return set(redaction_results)
