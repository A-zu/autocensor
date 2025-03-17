import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO


def blur(bbox, image, blur_coefficient):
    top_left = [bbox[0], bbox[1]]
    bottom_right = [bbox[2], bbox[3]]
    roi = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    blurred = cv2.GaussianBlur(roi, (blur_coefficient, blur_coefficient), 0)
    image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = blurred


def blur_items(result, items_to_blur, image, blur_coefficient):
    for box in result.boxes:
        if box.cls in items_to_blur:
            bbox = [int(x) for x in box.xyxy[0]]
            blur(bbox, image, blur_coefficient)


def blur_text(results, image, blur_coefficient):
    for bbox, text, confidence in results:
        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1]) for p in bbox]

        # Get the top-left and bottom-right
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        blur(bbox, image, blur_coefficient)


def process_images(dir_path: Path):
    items_to_blur = [62, 63]  # tv, laptop
    model = YOLO("yolo12x.pt")
    reader = easyocr.Reader(["en", "no"], gpu=True)
    blur_coefficient = 100 * 2 + 1

    for file in dir_path.iterdir():
        if not file.name.endswith(".png"):
            file.unlink()
            continue

        image = cv2.imread(file)

        results = model(image)
        blur_items(results[0], items_to_blur, image, blur_coefficient)

        results_text = reader.readtext(image)
        blur_text(results_text, image, blur_coefficient)
        # Overwrite image
        cv2.imwrite(dir_path / file.name, image)


def process_zip_file(uploaded_file_path: Path, processed_id, processed_dir):
    original_filename = uploaded_file_path.name.split("_", 1)[
        1
    ]  # Remove the UUID prefix
    processed_filename = f"processed_{processed_id}_{original_filename}"
    processed_file_path = processed_dir / processed_filename

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Extract the original zip
        with zipfile.ZipFile(uploaded_file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir_path)

        process_images(temp_dir_path)

        with zipfile.ZipFile(processed_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir_path):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, arcname=file_path.relative_to(temp_dir_path))

    return processed_file_path
