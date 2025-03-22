import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pillow_heif
from PIL import Image
from ultralytics import YOLO


def blur(bbox, image, blur_function):
    top_left = [bbox[0], bbox[1]]
    bottom_right = [bbox[2], bbox[3]]
    roi = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    blurred = blur_function(roi)
    image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = blurred


def blur_items(result, items_to_blur, image, blur_function):
    for box in result.boxes:
        if box.cls in items_to_blur:
            bbox = [int(x) for x in box.xyxy[0]]
            blur(bbox, image, blur_function)


def blur_text(results, image, blur_function):
    for bbox, text, confidence in results:
        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1]) for p in bbox]

        # Get the top-left and bottom-right
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        blur(bbox, image, blur_function)


def process_images(dir_path: Path):
    items_to_blur = [62, 63]  # tv, laptop
    model = YOLO("yolo12x.pt")
    reader = easyocr.Reader(["en", "no"], gpu=True)

    def blur_function(roi):
        blur_coefficient = 0.05 * 3
        h, w = roi.shape[:2]
        ksize = (int(h * blur_coefficient) | 1, int(w * blur_coefficient) | 1)
        return cv2.GaussianBlur(roi, ksize, 0)

    for root, _, files in os.walk(dir_path):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() == ".heic":
                heif_file = pillow_heif.read_heif(file)
                image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(file)

            if image is None:
                file.unlink()
                continue

            results = model(image)
            blur_items(results[0], items_to_blur, image, blur_function)

            results_text = reader.readtext(image)
            blur_text(results_text, image, blur_function)
            # Overwrite image
            cv2.imwrite(
                (dir_path / file.name).with_suffix(".jpg"),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            if file.suffix != ".jpg":
                file.unlink()


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
