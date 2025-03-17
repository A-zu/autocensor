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


def main():
    items_to_blur = [62, 63]  # tv, laptop
    input_folder = Path().parent / "images" / "input"
    output_folder = Path().parent / "images" / "both"
    blur_coefficient = 100 * 2 + 1

    model = YOLO("yolo12x.pt")
    reader = easyocr.Reader(["en", "no"], gpu=True)
    output_folder.mkdir(exist_ok=True)

    files = [x for x in input_folder.iterdir()]
    for file in files:
        image = cv2.imread(file)

        results = model(image)
        blur_items(results[0], items_to_blur, image, blur_coefficient)

        results_text = reader.readtext(image)
        blur_text(results_text, image, blur_coefficient)
        cv2.imwrite(output_folder / file.name, image)


if __name__ == "__main__":
    main()
