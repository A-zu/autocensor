import os
import pytest
from pathlib import Path

from blur_masked_images import process_file

INPUT_DIR = Path(__file__).parent / "bench_inputs"
OUTPUT_DIR = Path(__file__).parent / "bench_outputs"
YOLOE_MODEL = os.getenv("YOLOE_MODEL", "yoloe-v8l-seg.pt")
BLUR_INTENSITY = 0.5
SELECTED_ITEMS = ["document", "monitor", "person"]


@pytest.mark.parametrize("file_path", list(INPUT_DIR.iterdir()))
def test_process_file_speed(benchmark, file_path):
    def run():
        process_file(file_path, OUTPUT_DIR, SELECTED_ITEMS, BLUR_INTENSITY, YOLOE_MODEL)

    benchmark.name = file_path.name
    benchmark(run)
