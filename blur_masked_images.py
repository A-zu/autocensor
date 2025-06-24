import os
import time
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
    def __init__(self, label: str, count: int = None, *, level: str = "INFO"):
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
    by `coefficient` at a working height of 640 px.

    Args:
        image: BGR numpy array.
        coefficient: [0.0-1.0] controls blur strength.
        max_fraction: maximum blur kernel relative to image size.

    Returns:
        A new numpy array, same shape as `image`, blurred.
    """
    h, w = image.shape[:2]

    target_h = 640
    new_w = int(w / h * target_h)
    resized = cv2.resize(image, (new_w, target_h))

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


def process_result(
    result: Results, coefficient: float = 0.5, max_fraction: float = 0.05
) -> Tuple[Path, np.ndarray, bool]:
    """
    Process a single YOLO prediction result by applying blur to detected mask regions.

    Args:
        result (Results): A YOLO `Results` object containing image, path, and masks.
        coefficient (float): Blur strength between 0.0 and 1.0.
        max_fraction (float): Maximum kernel size as a fraction of image dimensions.

    Returns:
        Tuple[Path, np.ndarray, bool]:
            - Image or video path as a `Path` object.
            - Processed image as a NumPy array (blurred where masked).
            - Boolean indicating whether the input was a video frame.
    """
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
    is_video = path.suffix[1:].lower() in VID_FORMATS
    return path, image, is_video


def process_results_threaded(
    model: YOLOE,
    input_dir: Path,
    executor: ThreadPoolExecutor,
    blur_intensity: float,
    verbose: bool = True,
) -> Iterator[Tuple[Path, np.ndarray, bool]]:
    """
    Run YOLO predictions on all media in `input_dir`, blur masks, and yield
    each processed frame as soon as it's ready.

    Args:
        model (YOLOE): Initialized YOLO model for detection.
        input_dir (Path): Directory containing images/videos to process.
        executor (ThreadPoolExecutor): Executor on which to dispatch frame blurring.
        blur_intensity (float): Strength of blur to apply to masked regions.
        verbose (bool): If True, print model.predict progress.

    Returns:
        Iterator[Tuple[Path, np.ndarray, bool]]:
            Yields a tuple for each frame:
            - Path: original media path (image or video file).
            - np.ndarray: the blurred (or unmodified) frame array (BGR).
            - bool: True if this frame came from a video, False for standalone images.
    """
    results_iter = iter([])

    try:
        batch = int(YOLO_BATCH_SIZE)
        results = model.predict(
            input_dir,
            verbose=verbose,
            batch=batch,
            retina_masks=True,
            stream=True,
            show_labels=False,
            show_conf=False,
            show_boxes=False,
        )
        results_iter = executor.map(
            lambda result: process_result(result, blur_intensity), results
        )

    except FileNotFoundError as e:
        logger.debug(str(e))

    return results_iter


def save_image(image: np.ndarray, image_path: Path, output_dir: Path) -> None:
    """
    Save a single processed image frame to JPEG, preserving its filename.

    Args:
        image (np.ndarray): BGR image array to save.
        image_path (Path): Original input image path.
        output_dir (Path): Root directory in which to write the JPEG.
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
    Encode and save a list of processed frames as an MP4, copying original audio.

    Args:
        frames (List[np.ndarray]): In-memory BGR frames for one video.
        video_path (Path): Original video file path (for audio & metadata).
        output_dir (Path): Directory in which to write the .mp4.
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

        audio_input = ffmpeg.input(str(video_path), vn=None).audio

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
    Consume a stream of processed frames, saving images immediately
    and collecting/saving video frames in sequence.

    Args:
        results_iter (Iterator[Tuple[Path, np.ndarray, bool]]):
            An iterator yielding (path, frame, is_video) for each processed frame.
        output_dir (Path):
            Directory under which to save images (.jpg) and videos (.mp4).

    Returns:
        int: Total number of frames (images + video frames) saved.
    """
    total_frames = 0
    last_path: Path = None
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
    model: YOLOE,
    input_dir: Path,
    output_dir: Path,
    blur_intensity: float,
    verbose: bool = True,
) -> int:
    """
    Full pipeline: detect, blur, and save both images and videos under `input_dir`.

    1. Runs YOLO on every file in `input_dir` (images & videos).
    2. Blurs detected mask regions frame-by-frame.
    3. Saves images as JPEGs and videos as MP4 (with copied audio).

    Args:
        model (YOLOE): YOLO model instance for prediction.
        input_dir (Path): Directory containing source images/videos.
        output_dir (Path): Directory to write processed outputs.
        blur_intensity (float): Blur strength for masked regions.
        verbose (bool): If True, show YOLO predict progress.

    Returns:
        int: Total number of frames (images + video frames) that were saved.
    """
    with ThreadPoolExecutor() as executor:
        with FrameTimeLogger("Processing") as timer:
            results_iter = process_results_threaded(
                model, input_dir, executor, blur_intensity, verbose
            )

        with FrameTimeLogger("Saving frames") as timer:
            total_frames = consume_and_save(results_iter, output_dir)
            timer.set_count(total_frames)

    return total_frames


def process_directory(
    input_dir: Path,
    output_dir: Path,
    classes: List[str],
    blur_intensity: float,
    model_name: str,
) -> None:
    """
    Top-level image processing: sets up the YOLOE model and
    calls `process_images` for each subfolder.

    Args:
        input_dir: folder with extracted images.
        output_dir: folder to receive processed subfolders.
        classes: items for class selection.
    """
    model = YOLOE(model_name)
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
    logger.info(
        f"START process_file: {uploaded_file_path.name}, items={selected_items}, blur={blur_intensity}, model={model_name}"
    )
    output_path = output_dir / uploaded_file_path.name

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

                uploaded_file_path.rename(orig / uploaded_file_path.name)
                process_directory(
                    orig, output_dir, selected_items, blur_intensity, model_name
                )
                logger.info(f"Finished processing chunk, output at {output_path}")

            else:
                logger.debug("Detected zip archive, extracting and processing folder")
                extract_zip(uploaded_file_path, orig)
                uploaded_file_path.unlink()
                process_directory(
                    orig, proc, selected_items, blur_intensity, model_name
                )
                create_processed_zip(output_path, proc)

    finally:
        logger.debug("Clearing CUDA cache")
        torch.cuda.empty_cache()

    return output_path
