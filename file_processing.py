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
    # Extract coordinates from bounding box
    x_min, y_min = bbox[0], bbox[1]
    x_max, y_max = bbox[2], bbox[3]

    # Define the region of interest (ROI)
    roi = image[y_min:y_max, x_min:x_max]

    # Apply blurring function to the ROI
    blurred_roi = blur_function(roi)

    # Replace the original ROI with the blurred version in the image
    image[y_min:y_max, x_min:x_max] = blurred_roi


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


def blur_function(roi):
    # Calculate the kernel size based on the ROI dimensions
    h, w = roi.shape[:2]
    blur_coefficient = 0.05 * 6
    ksize_height = int(h * blur_coefficient) | 1
    ksize_width = int(w * blur_coefficient) | 1

    # Apply Gaussian Blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi, (ksize_height, ksize_width), 0)
    return blurred_roi


def load_image(file: Path):
    if file.suffix.lower() == ".heic":
        heif_file = pillow_heif.read_heif(file)
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to NumPy array and BGR color space
        image_np = np.array(image)
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(str(file))


def process_image(model: YOLO, reader: easyocr.Reader, file: Path, items_to_blur):
    image = load_image(file)
    if image is None:
        file.unlink()
        return

    results = model(image)
    blur_items(results[0], items_to_blur, image, blur_function)

    # 80 -> id of text
    if 80 in items_to_blur:
        results_text = reader.readtext(image)
        blur_text(results_text, image, blur_function)

    cv2.imwrite(
        file.with_suffix(".jpg"),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if file.suffix != ".jpg":
        file.unlink()


def process_images(dir_path: Path, items_to_blur):
    model = YOLO("yolo12x.pt")
    reader = easyocr.Reader(["en", "no"], gpu=True)
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            file_path = Path(root) / file_name
            process_image(model, reader, file_path, items_to_blur)


def extract_zip(uploaded_file_path: Path, temp_dir_path: Path):
    with zipfile.ZipFile(uploaded_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir_path)


def process_images_in_temp_dir(temp_dir_path: Path, selected_items):
    process_images(temp_dir_path, selected_items)


def create_processed_zip(processed_file_path: Path, temp_dir_path: Path):
    with zipfile.ZipFile(processed_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir_path):
            for file_name in files:
                file_path = Path(root) / file_name
                zipf.write(file_path, arcname=file_path.relative_to(temp_dir_path))


def process_zip_file(
    uploaded_file_path: Path,
    output_path: Path,
    selected_items,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        extract_zip(uploaded_file_path, temp_dir_path)
        process_images_in_temp_dir(temp_dir_path, selected_items)
        create_processed_zip(output_path, temp_dir_path)
