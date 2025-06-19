import os
import time
import logging
import zipfile
import tempfile
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
from ultralytics import YOLOE
from ultralytics.data.utils import VID_FORMATS
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

YOLO_BATCH_SIZE = os.getenv("YOLO_BATCH_SIZE", "8")


class VideoWriterContext:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.cap = None

    def __enter__(self):
        self.writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, self.size)
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.release()


class VideoCaptureContext:
    def __init__(self, path):
        self.path = path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.path))
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()


class FrameTimeLogger:
    def __init__(self, label: str, count: int = None):
        self.label = label
        self.count = count

    def set_count(self, count: int):
        self.count = count

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        if self.count:
            avg = duration / self.count
            fps = self.count / duration
            logger.info(
                f"{self.label:<20} — {self.count:>4} frames in {duration:>5.2f}s "
                f"(avg: {avg:.4f}s, {fps:.2f} FPS)"
            )
        else:
            logger.info(f"{self.label} took {duration:.2f}s")


def get_kernel(
    image: np.ndarray, *, coefficient: float, max_fraction: float = 0.05
) -> tuple[int, int]:
    """
    Compute an odd-sized Gaussian kernel based on image size.

    Args:
        image: HxWxC array.
        coefficient: [0.0-1.0] fraction of `max_fraction` of min(H, W).
        max_fraction: maximum fraction of the smaller image dimension.

    Returns:
        A tuple (k, k) where k is odd and ~ coefficient*max_fraction*min(H,W).
    """
    # clamp coefficient
    coefficient = max(0.0, min(1.0, coefficient))

    h, w = image.shape[:2]
    base = min(h, w)
    # force odd with bitwise OR 1
    k = int(base * max_fraction * coefficient) | 1
    return (k, k)


def blur_image_copy(
    image: np.ndarray, *, coefficient: float = 0.5, max_fraction: float = 0.05
) -> np.ndarray:
    """
    Return a fully blurred copy of `image`, with blur scaled
    by `coefficient` at a working height of 1080 px.

    Args:
        image: BGR numpy array.
        coefficient: [0.0-1.0] controls blur strength.
        max_fraction: maximum blur kernel relative to image size.

    Returns:
        A new numpy array, same shape as `image`, blurred.
    """
    # remember original
    h, w = image.shape[:2]

    # scale to height=1080 (preserve aspect)
    target_h = 640
    new_w = int(w / h * target_h)
    resized = cv2.resize(image, (new_w, target_h))

    # blur & back-resize
    kernel = get_kernel(resized, coefficient=coefficient, max_fraction=max_fraction)
    blurred = cv2.GaussianBlur(resized, kernel, 0)
    return cv2.resize(blurred, (w, h))


def blur_mask(mask: torch.Tensor, image: np.ndarray, blurred_image: np.ndarray) -> None:
    """
    In-place blend `blurred_image` into `image` where `mask` is True.

    Args:
        mask: 2D tensor (H, W) of booleans or floats.
        image: original image to modify.
        blurred_image: fully blurred version (same shape as image).

    Side-Effects:
        Writes into `image` where mask==255.
    """
    # binary 0/255
    mask_np = (mask.cpu().numpy().astype(np.uint8)) * 255

    # expand to 3 channels
    mask_3ch = cv2.merge([mask_np] * 3)

    # in-place blend
    image[mask_3ch == 255] = blurred_image[mask_3ch == 255]


def process_results(
    results: List[Results], *, coefficient: float = 0.5, max_fraction: float = 0.05
) -> Tuple[Dict[Path, np.ndarray], Dict[Path, List[np.ndarray]]]:
    """
    Process YOLO model prediction results to blur masked regions in each image.

    Args:
        results: List of YOLO `Results` objects.
        coefficient: Strength of blur to apply [0.0–1.0].
        max_fraction: Maximum blur kernel size as a fraction of image size.

    Returns:
        A tuple of:
            - Dictionary mapping image paths to processed image arrays.
            - Dictionary mapping video paths to lists of processed frame arrays.
    """
    images = {}
    videos = defaultdict(list)

    for result in results:
        image = result.orig_img.copy()

        if result.masks:
            try:
                blurred_image = blur_image_copy(
                    image, coefficient=coefficient, max_fraction=max_fraction
                )
                combined_mask = torch.any(result.masks.data > 0.5, dim=0)
                blur_mask(combined_mask, image, blurred_image)

            except Exception:
                logger.exception("Error during mask processing and blurring")

        path = Path(result.path)

        if path.suffix[1:].lower() in VID_FORMATS:
            videos[path].append(image)
        else:
            images[path] = image

    return images, videos


def save_images(
    images: Dict[Path, np.ndarray], root_dir: Path, output_dir: Path
) -> None:
    """
    Save processed images to disk, preserving relative paths.

    Args:
        images: Dictionary mapping input paths to image arrays.
        root_dir: Root of original input directory structure.
        output_dir: Destination directory to save processed images.
    """
    for image_path, image in images.items():
        if image is None:
            logger.warning(f"No frame to write for image: {image_path=}")
            continue

        rel = image_path.relative_to(root_dir)
        dest = output_dir / rel.with_suffix(".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)

        # save as JPEG at quality=90
        cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.debug(f"Saved image: {dest=}")


def save_videos(
    videos: Dict[Path, List[np.ndarray]], root_dir: Path, output_dir: Path
) -> None:
    """
    Save processed video frames to disk as MP4 files.

    Args:
        videos: Dictionary mapping video paths to lists of frame arrays.
        root_dir: Root of original input directory structure.
        output_dir: Destination directory to save processed videos.
    """
    for video_path, frames in videos.items():
        if frames is None:
            logger.warning(f"No frames to write for video: {video_path=}")
            continue

        if any(frame.shape != frames[0].shape for frame in frames):
            logger.error(f"Inconsistent frame shapes in video: {video_path=}")
            continue

        rel = video_path.relative_to(root_dir)
        dest = output_dir / rel.with_suffix(".mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)

        with VideoCaptureContext(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps < 1:
            fps = 30.0
            logger.warning(f"Invalid FPS from {video_path=}, defaulting to 30.0")

        h, w = frames[0].shape[:2]
        logger.debug(
            f"Preparing to write video: {dest} ({len(frames)} frames @ {fps:.2f} FPS)"
        )

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            with VideoWriterContext(dest, fourcc, fps, (w, h)) as writer:
                start = time.perf_counter()

                for frame in frames:
                    writer.write(frame)

                elapsed = time.perf_counter() - start
                logger.debug(
                    f"Saved video: {dest} ({len(frames)} frames @ {fps:.2f} FPS) in {elapsed:.2f}s"
                )

        except Exception:
            logger.exception(f"Failed to write video {dest}")


def process_images(
    model: YOLOE,
    input_dir: Path,
    output_dir: Path,
    blur_intensity: float,
    verbose: bool = True,
) -> int:
    """
    End-to-end image and video processing for a given directory.

    This function:
      - Runs detection on all frames.
      - Applies blur to masked regions.
      - Saves processed images and video outputs.

    Args:
        model: YOLOE model used for prediction.
        input_dir: Folder with images/videos to process.
        output_dir: Folder to write processed files.
        blur_intensity: Blur strength passed to blurring function.
        verbose: Whether to print model.predict logs.

    Returns:
        Total number of frames processed.
    """
    with FrameTimeLogger("Detection") as timer:
        try:
            results = model.predict(
                input_dir,
                verbose=verbose,
                batch=int(YOLO_BATCH_SIZE),
                retina_masks=True,
                show_labels=False,
                show_conf=False,
                show_boxes=False,
            )
            total_frames = len(results)
            timer.set_count(total_frames)

        except FileNotFoundError as e:
            logger.debug(str(e))
            return

    with FrameTimeLogger("Blurring", total_frames):
        images, videos = process_results(results, coefficient=blur_intensity)

    if images:
        with FrameTimeLogger("Saving images", len(images)):
            save_images(images, input_dir, output_dir)

    if videos:
        with FrameTimeLogger("Saving videos", total_frames - len(images)):
            save_videos(videos, input_dir, output_dir)

    return total_frames


def process_directory(
    input_dir: Path,
    output_dir: Path,
    selected_items: List[str],
    blur_intensity: float,
    model_name: str,
) -> None:
    """
    Top-level image processing: sets up the YOLOE model and
    calls `process_images` for each subfolder.

    Args:
        input_dir: folder with extracted images.
        output_dir: folder to receive processed subfolders.
        selected_items: items for class selection.
    """
    model = YOLOE(model_name)
    classes = selected_items
    model.set_classes(classes, model.get_text_pe(classes))

    with FrameTimeLogger(f"Processing directory — '{input_dir.name}'") as timer:
        total_frames = process_images(model, input_dir, output_dir, blur_intensity)
        timer.set_count(total_frames)

    for subdir in input_dir.rglob("*"):
        if not subdir.is_dir():
            continue

        rel = subdir.relative_to(input_dir)
        out = output_dir / rel
        out.mkdir(parents=True, exist_ok=True)

        with FrameTimeLogger(f"Processing directory — '{subdir.name}'") as timer:
            total_frames = process_images(model, subdir, out, blur_intensity)
            timer.set_count(total_frames)


def extract_zip(uploaded_file_path: Path, temp_dir_path: Path) -> None:
    """
    Unzip `uploaded_file_path` into `temp_dir_path`.
    """
    with zipfile.ZipFile(uploaded_file_path, "r") as z:
        z.extractall(temp_dir_path)


def create_processed_zip(processed_file_path: Path, temp_dir_path: Path) -> None:
    """
    Zip all files under `temp_dir_path` into `processed_file_path`,
    preserving folder structure.
    """
    with zipfile.ZipFile(processed_file_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file in temp_dir_path.rglob("*"):
            if file.is_file():
                z.write(file, arcname=file.relative_to(temp_dir_path))


def process_file(
    uploaded_file_path: Path,
    output_dir: Path,
    selected_items: List[str],
    blur_intensity: float,
    model_name: str,
) -> Path:
    """
    Handles processing of an uploaded file (image, video, or ZIP archive).

    Behavior:
    - If the uploaded file is a ZIP archive:
        1. Extracts contents into a temporary "original" folder.
        2. Processes each file, applying blur to selected items.
        3. Saves processed outputs into a temporary "processed" folder.
        4. Re-archives the processed files into a ZIP saved at `output_dir`.

    - If the uploaded file is a single image or video:
        1. Moves the file to a temporary "original" folder.
        2. Processes the file, applying blur to selected items.
        3. Saves the result as a .jpg or .mp4 file in `output_dir`.

    Returns:
        Path to the final processed file (ZIP, JPG, or MP4).
    """
    output_path = output_dir / uploaded_file_path.name

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orig = tmp_path / "original"
            proc = tmp_path / "processed"
            orig.mkdir()
            proc.mkdir()

            if uploaded_file_path.suffix != ".zip":
                suffix = uploaded_file_path.suffix[1:].lower()
                if suffix in VID_FORMATS:
                    output_path = output_path.with_suffix(".mp4")
                else:
                    output_path = output_path.with_suffix(".jpg")

                uploaded_file_path.rename(orig / uploaded_file_path.name)
                process_directory(
                    orig, output_dir, selected_items, blur_intensity, model_name
                )
            else:
                extract_zip(uploaded_file_path, orig)
                uploaded_file_path.unlink()
                process_directory(
                    orig, proc, selected_items, blur_intensity, model_name
                )
                create_processed_zip(output_path, proc)

    finally:
        torch.cuda.empty_cache()

    return output_path
