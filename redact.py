import fitz  # PyMuPDF
import re

def redact_pdf(input_path: str, output_path: str, redactions: set[str]):
    doc = fitz.open(input_path)

    for page in doc:
        for phrase in redactions:
            # Use regex to be case-insensitive and match variations
            text_instances = page.search_for(phrase, quads=False)
            for inst in text_instances:
                page.add_redact_annot(inst, fill=(0, 0, 0))  # Black box

        page.apply_redactions()

    doc.save(output_path)
    doc.close()
