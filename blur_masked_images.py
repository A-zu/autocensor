import os
import time
import shutil
import logging
import zipfile
import tempfile
from typing import Iterator, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import ffmpeg
import numpy as np
from ultralytics import YOLOE
from ultralytics.data.utils import VID_FORMATS
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

YOLO_BATCH_SIZE = os.getenv("YOLO_BATCH_SIZE", "16")


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
    def __init__(self, label: str, count: int | None = None, *, level: str = "INFO"):
        self.label = label
        self.level = getattr(logging, level.upper(), logging.INFO)
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
            logger.log(
                self.level,
                f"{self.label:<20} — {self.count:>4} frames in {duration:>5.2f}s "
                f"(avg: {avg:.4f}s, {fps:.2f} FPS)",
            )
        else:
            logger.log(self.level, f"{self.label} took {duration:.2f}s")


def get_kernel(
    image: np.ndarray, *, coefficient: float, max_fraction: float = 0.05
) -> tuple[int, int]:
    """
    Compute an odd-sized Gaussian kernel based on image dimensions.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C) or (H, W).
        coefficient (float): Fraction [0.0, 1.0] of max kernel size.
        max_fraction (float): Maximum fraction of the smaller image dimension.

    Returns:
        Tuple[int, int]: Kernel size (k, k), where k is odd and
                         proportional to coefficient * max_fraction * min(H, W).
    """
    # clamp coefficient
    coefficient = max(0.0, min(1.0, coefficient))

    h, w = image.shape[:2]
    base = min(h, w)
    # force odd with bitwise OR 1
    k = int(base * max_fraction * coefficient) | 1
    return (k, k)


def blur_frame_copy(
    frame: np.ndarray, *, coefficient: float = 0.5, max_fraction: float = 0.05
) -> np.ndarray:
    """
    Return a blurred copy of the input frame.

    Resizes frame to a working height of 640 px, applies Gaussian blur,
    then rescales back to original size.

    Args:
        frame (np.ndarray): BGR image array of shape (H, W, 3).
        coefficient (float): Blur strength [0.0, 1.0].
        max_fraction (float): Max kernel size fraction relative to frame size.

    Returns:
        np.ndarray: New blurred image array, same shape as frame.
    """
    h, w = frame.shape[:2]

    target_h = 640
    new_w = int(w / h * target_h)
    resized = cv2.resize(frame, (new_w, target_h))

    kernel = get_kernel(resized, coefficient=coefficient, max_fraction=max_fraction)
    blurred = cv2.GaussianBlur(resized, kernel, 0)

    return cv2.resize(blurred, (w, h))


def blur_mask(mask: torch.Tensor, frame: np.ndarray, blurred_frame: np.ndarray) -> None:
    """
    In-place replace masked regions of frame with blurred version.

    Args:
        mask (torch.Tensor): 2D tensor (H, W) of bools or floats indicating mask.
        frame (np.ndarray): Original BGR frame array to modify.
        blurred_frame (np.ndarray): Fully blurred BGR frame, same shape as frame.

    Side effects:
        Modifies `frame` so that wherever mask is True, pixels come from blurred_frame.
    """
    # binary 0/255
    mask_np = (mask.cpu().numpy().astype(np.uint8)) * 255

    # expand to 3 channels
    mask_3ch = cv2.merge([mask_np] * 3)

    # in-place blend
    frame[mask_3ch == 255] = blurred_frame[mask_3ch == 255]


def blur_result(
    result: Results, coefficient: float = 0.5, max_fraction: float = 0.05
) -> Tuple[Path, np.ndarray, bool]:
    """
    Apply blur to all detected mask regions in a YOLO result.

    Args:
        result (Results): YOLOE Results object (has orig_img, masks, path).
        coefficient (float): Blur strength [0.0, 1.0].
        max_fraction (float): Max kernel size fraction relative to image size.

    Returns:
        Tuple[Path, np.ndarray, bool]:
            path: Path to original media.
            frame: Processed BGR image array with masks blurred.
            is_video: True if path suffix indicates a video format.
    """
    frame = result.orig_img.copy()

    if result.masks:
        try:
            combined_mask = torch.any(result.masks.data > 0.5, dim=0)
            blurred_frame = blur_frame_copy(
                frame, coefficient=coefficient, max_fraction=max_fraction
            )
            blur_mask(combined_mask, frame, blurred_frame)

        except Exception:
            logger.exception("Error during mask processing and blurring")

    path = Path(result.path)
    is_video = path.suffix[1:].lower() in VID_FORMATS
    return path, frame, is_video


def detect_result(result: Results) -> Tuple[Path, np.ndarray, bool]:
    """
    Render detection results (boxes, labels) onto the image.

    Args:
        result (Results): YOLOE Results object (has plot, path).

    Returns:
        Tuple[Path, np.ndarray, bool]:
            path: Path to original media.
            frame: BGR image array with detection overlays.
            is_video: True if path suffix indicates a video format.
    """
    frame = result.plot()
    path = Path(result.path)
    is_video = path.suffix[1:].lower() in VID_FORMATS
    return path, frame, is_video


def censor_mask(mask: torch.Tensor, image: np.ndarray) -> None:
    """
    In-place black out masked regions of image.

    Args:
        mask (torch.Tensor): 2D tensor (H, W) of bools or floats indicating mask.
        image (np.ndarray): Original BGR image array to modify.

    Side effects:
        Modifies `image` so that wherever mask is True, pixels are set to zero (black).
    """
    # binary 0/255
    mask_np = (mask.cpu().numpy().astype(np.uint8)) * 255

    # expand to 3 channels
    mask_3ch = cv2.merge([mask_np] * 3)

    # in-place blend
    image[mask_3ch == 255] = 0


def censor_result(result: Results) -> Tuple[Path, np.ndarray, bool]:
    """
    Apply hard censor (black fill) to all detected mask regions.

    Args:
        result (Results): YOLOE Results object (has orig_img, masks, path).

    Returns:
        Tuple[Path, np.ndarray, bool]:
            path: Path to original media.
            frame: Processed BGR image array with masks blacked out.
            is_video: True if path suffix indicates a video format.
    """
    frame = result.orig_img.copy()

    if result.masks:
        try:
            combined_mask = torch.any(result.masks.data > 0.5, dim=0)
            censor_mask(combined_mask, frame)

        except Exception:
            logger.exception("Error during mask processing and blurring")

    path = Path(result.path)
    is_video = path.suffix[1:].lower() in VID_FORMATS
    return path, frame, is_video


def process_results_threaded(
    mode: str,
    model: YOLOE,
    confidence_threshold: float,
    input_dir: Path,
    executor: ThreadPoolExecutor,
    blur_intensity: float,
) -> Iterator[Tuple[Path, np.ndarray, bool]]:
    """
    Run YOLO predictions on all media in input_dir, process each result in parallel,
    and yield processed frames immediately.

    Args:
        mode (str): One of "blur", "censor", or "detect".
        model (YOLOE): Initialized YOLOE model for inference.
        confidence_threshold (float): Minimum confidence for detections.
        input_dir (Path): Directory containing images or videos.
        executor (ThreadPoolExecutor): Executor for parallel frame processing.
        blur_intensity (float): Strength of blur to apply (only for mode "blur").

    Returns:
        Iterator[Tuple[Path, np.ndarray, bool]]:
            Yields (original_path, processed_frame, is_video_flag) per frame.
    """
    results_iter = iter([])

    try:
        batch = int(YOLO_BATCH_SIZE)

        if mode == "blur":
            results = model.predict(
                input_dir,
                conf=confidence_threshold,
                batch=batch,
                stream=True,
                verbose=True,
                show_conf=False,
                show_boxes=False,
                show_labels=False,
                retina_masks=True,
            )
            results_iter = executor.map(
                lambda result: blur_result(result, blur_intensity, max_fraction=0.25), results
            )
        elif mode == "censor":
            results = model.predict(
                input_dir,
                conf=confidence_threshold,
                batch=batch,
                stream=True,
                verbose=True,
                show_conf=False,
                show_boxes=False,
                show_labels=False,
                retina_masks=True,
            )
            results_iter = executor.map(lambda result: censor_result(result), results)
        elif mode == "detect":
            results = model.predict(
                input_dir,
                conf=confidence_threshold,
                batch=batch,
                stream=True,
                verbose=True,
                show_conf=True,
                show_boxes=True,
                show_labels=True,
                retina_masks=True,
            )
            results_iter = executor.map(lambda result: detect_result(result), results)
        else:
            raise ValueError("Unrecognized mode was detected")

    except FileNotFoundError as e:
        logger.debug(str(e))

    return results_iter


def save_image(image: np.ndarray, image_path: Path, output_dir: Path) -> None:
    """
    Save a single BGR image as JPEG in the output directory.

    Args:
        image (np.ndarray): BGR image array to save.
        image_path (Path): Original file path (used for naming).
        output_dir (Path): Directory under which .jpg will be written.

    Side effects:
        Creates output_dir if it does not exist.
    """
    if image is None:
        logger.warning(f"No frame to write for image: {image_path=}")
        return

    dest = output_dir / image_path.name
    dest = dest.with_suffix(".jpg")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # save as JPEG at quality=90
    cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.debug(f"Saved image: {dest=}")


def save_video(frames: List[np.ndarray], video_path: Path, output_dir: Path) -> None:
    """
    Encode a sequence of BGR frames into an MP4 file, copying original audio.

    Args:
        frames (List[np.ndarray]): List of BGR frames to encode.
        video_path (Path): Path to original video (for audio track).
        output_dir (Path): Directory under which .mp4 will be written.

    Side effects:
        Creates output_dir if it does not exist.
    """
    if not frames:
        logger.warning(f"No frames to write for video: {video_path=}")
        return

    if any(frame.shape != frames[0].shape for frame in frames):
        logger.error(f"Inconsistent frame shapes in video: {video_path=}")
        return

    dest = output_dir / video_path.name
    dest = dest.with_suffix(".mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with VideoCaptureContext(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps < 1:
        fps = 30.0
        logger.warning(f"Invalid FPS from {video_path=}, defaulting to 30.0")

    h, w = frames[0].shape[:2]
    logger.debug(f"Writing {dest.name} ({len(frames)} frames @ {fps:.2f} FPS)")

    try:
        video_input = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{w}x{h}",
            framerate=fps,
        )

        audio_input = ffmpeg.input(str(video_path), acodec="aac", vn=None).audio

        process = video_input.output(
            audio_input,
            map="1:a:0",
            acodec="copy",
            vcodec="h264_nvenc",
            preset="p4",
            pix_fmt="yuv420p",
            threads=0,
            filename=str(dest),
            loglevel="quiet",
            shortest=None,
        ).run_async(pipe_stdin=True, overwrite_output=True)

        with FrameTimeLogger(
            f"Saved video {dest.name}", level="DEBUG", count=len(frames)
        ):
            for frame in frames:
                process.stdin.write(frame.tobytes())

            process.stdin.close()
            process.wait()

    except Exception:
        logger.exception(f"Failed to write video {dest}")


def consume_and_save(
    results_iter: Iterator[Tuple[Path, np.ndarray, bool]],
    output_dir: Path,
) -> int:
    """
    Save each processed frame or video segment from the results iterator.

    Images are saved immediately; video frames are buffered per file and then saved.

    Args:
        results_iter (Iterator): Yields (path, frame, is_video) for each frame.
        output_dir (Path): Base directory for saving outputs.

    Returns:
        int: Total number of frames saved (images + video frames).
    """
    total_frames = 0
    last_path: Path | None = None
    video_buffer: List[np.ndarray] = []

    for path, frame, is_video in results_iter:
        if is_video:
            # Switched to a new video?
            if last_path is not None and path != last_path and video_buffer:
                # Flush the previous video
                total_frames += len(video_buffer)
                save_video(video_buffer, last_path, output_dir)
                video_buffer = [frame]
            else:
                video_buffer.append(frame)
            last_path = path

        else:
            # Standalone image
            total_frames += 1
            save_image(frame, path, output_dir)

    # Flush any remaining video frames
    if video_buffer and last_path is not None:
        total_frames += len(video_buffer)
        save_video(video_buffer, last_path, output_dir)

    return total_frames


def process_images(
    mode: str,
    model: YOLOE,
    confidence_threshold: float,
    input_dir: Path,
    output_dir: Path,
    blur_intensity: float = 0.5,
) -> int:
    """
    Run detection on every file in input_dir, process each frame,
    and save results under output_dir.

    This covers both images and videos, applying blur or censoring as specified.

    Args:
        mode (str): One of "blur", "censor", or "detect".
        model (YOLOE): YOLOE model instance.
        confidence_threshold (float): Minimum confidence threshold.
        input_dir (Path): Directory of source media.
        output_dir (Path): Directory to write processed media.
        blur_intensity (float): Blur strength (for mode "blur").

    Returns:
        int: Total number of frames saved.
    """
    with ThreadPoolExecutor() as executor:
        with FrameTimeLogger("Processing") as timer:
            results_iter = process_results_threaded(
                mode,
                model,
                confidence_threshold,
                input_dir,
                executor,
                blur_intensity,
            )

        with FrameTimeLogger("Saving frames") as timer:
            total_frames = consume_and_save(results_iter, output_dir)
            timer.set_count(total_frames)

    return total_frames


def process_directory(
    mode: str,
    model_name: str,
    classes: List[str],
    confidence_threshold: float,
    input_dir: Path,
    output_dir: Path,
    blur_intensity: float,
) -> None:
    """
    Initialize the YOLOE model, process all files in input_dir (recursively),
    and write outputs to output_dir preserving subfolder structure.

    Args:
        mode (str): One of "blur", "censor", or "detect".
        model_name (str): Name or path of YOLOE model.
        classes (List[str]): Class names to detect.
        confidence_threshold (float): Minimum confidence threshold.
        input_dir (Path): Root directory of input media.
        output_dir (Path): Root directory for processed outputs.
        blur_intensity (float): Blur strength (for mode "blur").
    """
    model = YOLOE(model_name)
    model.set_classes(classes, model.get_text_pe(classes))

    with FrameTimeLogger(f"Processing directory — '{input_dir.name}'") as timer:
        total_frames = process_images(
            mode,
            model,
            confidence_threshold,
            input_dir,
            output_dir,
            blur_intensity,
        )
        timer.set_count(total_frames)

    for subdir in input_dir.rglob("*"):
        if not subdir.is_dir():
            continue

        rel = subdir.relative_to(input_dir)
        out = output_dir / rel
        out.mkdir(parents=True, exist_ok=True)

        with FrameTimeLogger(f"Processing directory — '{subdir.name}'") as timer:
            total_frames = process_images(
                mode,
                model,
                confidence_threshold,
                subdir,
                out,
                blur_intensity,
            )
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
    mode: str,
    model_name: str,
    uploaded_file_path: Path,
    output_dir: Path,
    confidence_threshold: float,
    *,
    selected_items: List[str] | None = None,
    blur_intensity: float = 0.5,
) -> Path:
    """
    Handle a single uploaded file (image, video, or ZIP), process it, and return the output path.

    If input is a ZIP:
      1. Extract to a temp 'original' folder.
      2. Process each media file.
      3. Zip processed results.

    If input is an image/video:
      1. Copy to temp 'original'.
      2. Process and save to output_dir.

    Args:
        mode (str): One of "blur", "censor", or "detect".
        model_name (str): YOLOE model identifier.
        uploaded_file_path (Path): Path to uploaded media or ZIP.
        output_dir (Path): Directory to save final output.
        confidence_threshold (float): Minimum detection confidence.
        selected_items (List[str], optional): Subset of classes to process.
        blur_intensity (float): Blur strength (for mode "blur").

    Returns:
        Path: Path to the final processed file (ZIP, JPG, or MP4).
    """
    logger.info(
        f"START process_file: {uploaded_file_path.name}, {mode=}, {model_name=}, {selected_items=}, {confidence_threshold=}, {blur_intensity=}"
    )
    output_path = output_dir / uploaded_file_path.name
    if selected_items is None:
        selected_items = []

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orig = tmp_path / "original"
            proc = tmp_path / "processed"
            orig.mkdir()
            proc.mkdir()

            if uploaded_file_path.suffix != ".zip":
                logger.debug("Detected single media file, moving to original")
                suffix = uploaded_file_path.suffix[1:].lower()
                if suffix in VID_FORMATS:
                    output_path = output_path.with_suffix(".mp4")

                else:
                    output_path = output_path.with_suffix(".jpg")

                shutil.copy2(uploaded_file_path, orig / uploaded_file_path.name)
                process_directory(
                    mode,
                    model_name,
                    selected_items,
                    confidence_threshold,
                    orig,
                    output_dir,
                    blur_intensity,
                )
                logger.info(f"Finished processing chunk, output at {output_path}")

            else:
                logger.debug("Detected zip archive, extracting and processing folder")
                extract_zip(uploaded_file_path, orig)
                process_directory(
                    mode,
                    model_name,
                    selected_items,
                    confidence_threshold,
                    orig,
                    proc,
                    blur_intensity,
                )
                create_processed_zip(output_path, proc)

    finally:
        logger.debug("Clearing CUDA cache")
        torch.cuda.empty_cache()

    return output_path
