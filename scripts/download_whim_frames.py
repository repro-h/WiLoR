#!/usr/bin/env python3
"""Download WHIM source videos as H.264 and extract annotated frames."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from pytubefix import YouTube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Parent containing WHIM/ (for example /data2/hyp/data).",
    )
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument(
        "--video-ids-json",
        type=Path,
        default=None,
        help="Defaults to WiLoR/whim/<mode>_video_ids.json.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Downloaded video cache; defaults to <root>/WHIM/Videos.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Extracted images; defaults to <root>/WHIM/<mode>/images.",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=None,
        help="Progress and failure report path.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Only process this video ID; may be passed repeatedly.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite images that already exist and are readable.",
    )
    parser.add_argument(
        "--delete-videos",
        action="store_true",
        help="Delete each cached video after successful extraction.",
    )
    return parser.parse_args()


def stream_codec(stream: object) -> str:
    fields = (
        getattr(stream, "video_codec", None),
        getattr(stream, "codecs", None),
        getattr(stream, "codec", None),
        getattr(stream, "mime_type", None),
    )
    return " ".join(str(value) for value in fields if value).lower()


def stream_height(stream: object) -> int:
    resolution = getattr(stream, "resolution", None)
    if not resolution:
        return 0
    try:
        return int(str(resolution).lower().rstrip("p"))
    except ValueError:
        return 0


def select_h264_stream(yt: YouTube, target_height: int) -> object:
    streams = list(yt.streams.filter(file_extension="mp4"))
    h264 = [stream for stream in streams if "avc1" in stream_codec(stream)]
    if not h264:
        available = sorted({stream_codec(stream) for stream in streams})
        raise RuntimeError(f"No H.264/avc1 MP4 stream; available codecs={available}")

    def rank(stream: object) -> Tuple[int, int, int, int]:
        height = stream_height(stream)
        fps = int(getattr(stream, "fps", 0) or 0)
        progressive = bool(getattr(stream, "is_progressive", False))
        return (
            -abs(height - target_height),
            int(height <= target_height),
            fps,
            int(progressive),
        )

    return max(h264, key=rank)


def first_readable_frame(video_path: Path) -> Tuple[bool, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None and frame.size), fps


def cached_video_is_usable(video_path: Path) -> Tuple[bool, str]:
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return False, "missing"
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        codec = result.stdout.strip().lower()
        if result.returncode != 0 or codec not in {"h264", "avc1"}:
            return False, codec or "ffprobe-failed"
    readable, _ = first_readable_frame(video_path)
    return readable, "h264" if readable else "opencv-unreadable"


def download_h264_video(
    video_id: str,
    video_path: Path,
    target_height: int,
    retries: int,
) -> Dict[str, object]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        temporary = video_path.with_name(f".{video_id}.download.mp4")
        try:
            if temporary.exists():
                temporary.unlink()
            yt = YouTube(f"https://youtu.be/{video_id}")
            stream = select_h264_stream(yt, target_height)
            downloaded = Path(
                stream.download(
                    output_path=str(video_path.parent),
                    filename=temporary.name,
                )
            )
            os.replace(downloaded, video_path)
            readable, fps = first_readable_frame(video_path)
            if not readable:
                raise RuntimeError("Downloaded H.264 stream is not OpenCV-readable")
            return {
                "codec": stream_codec(stream),
                "resolution": getattr(stream, "resolution", None),
                "fps": fps,
                "attempt": attempt,
            }
        except Exception as error:  # Network and YouTube failures vary by release.
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(2**attempt)

    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def annotation_frames(annotation_dir: Path) -> List[Tuple[int, Path]]:
    result = []
    for path in annotation_dir.glob("*.npy"):
        try:
            result.append((int(path.stem), path))
        except ValueError:
            continue
    return sorted(result)


def image_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def extract_frames(
    video_path: Path,
    frame_rows: Iterable[Tuple[int, Path]],
    output_dir: Path,
    source_fps: float,
    jpeg_quality: int,
    overwrite: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(frame_rows)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")

    decoded_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if decoded_fps <= 0 or source_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid FPS: decoded={decoded_fps}, annotation={source_fps}"
        )

    written = 0
    cached = 0
    failures = []
    for annotation_index, annotation_path in rows:
        output_path = output_dir / f"{annotation_path.stem}.jpg"
        if not overwrite and image_is_valid(output_path):
            cached += 1
            continue

        frame_index = int(round(annotation_index * decoded_fps / source_fps))
        if frame_index < 0 or (frame_count > 0 and frame_index >= frame_count):
            failures.append(
                {"annotation": annotation_index, "decoded_frame": frame_index}
            )
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = cap.read()
        if not ok or image is None or not image.size:
            failures.append(
                {"annotation": annotation_index, "decoded_frame": frame_index}
            )
            continue
        if not cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        ):
            failures.append(
                {"annotation": annotation_index, "decoded_frame": frame_index}
            )
            continue
        written += 1

    cap.release()
    return {
        "annotations": len(rows),
        "written": written,
        "cached": cached,
        "failed_frames": failures,
        "decoded_fps": decoded_fps,
        "decoded_frame_count": frame_count,
    }


def write_status(path: Path, status: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    whim_root = args.root / "WHIM"
    metadata_path = args.video_ids_json or repo_root / "whim" / (
        f"{args.mode}_video_ids.json"
    )
    annotation_root = whim_root / args.mode / "anno"
    video_root = args.video_root or whim_root / "Videos"
    image_root = args.image_root or whim_root / args.mode / "images"
    status_path = args.status_json or whim_root / f"download_{args.mode}_status.json"

    metadata = json.loads(metadata_path.read_text())
    video_ids = list(args.video_id or metadata.keys())
    if args.limit > 0:
        video_ids = video_ids[: args.limit]

    status: Dict[str, object] = {
        "mode": args.mode,
        "metadata": str(metadata_path),
        "annotation_root": str(annotation_root),
        "video_root": str(video_root),
        "image_root": str(image_root),
        "requested": len(video_ids),
        "completed": {},
        "failed": {},
    }

    for number, video_id in enumerate(video_ids, 1):
        print(f"[{number}/{len(video_ids)}] {video_id}", flush=True)
        if video_id not in metadata:
            status["failed"][video_id] = "video ID missing from metadata"
            write_status(status_path, status)
            continue

        annotation_dir = annotation_root / video_id
        frames = annotation_frames(annotation_dir)
        if not frames:
            status["failed"][video_id] = f"no annotations in {annotation_dir}"
            write_status(status_path, status)
            continue

        output_dir = image_root / video_id
        expected_paths = [output_dir / f"{path.stem}.jpg" for _, path in frames]
        if not args.overwrite_images and all(image_is_valid(path) for path in expected_paths):
            status["completed"][video_id] = {
                "annotations": len(frames),
                "written": 0,
                "cached": len(frames),
                "failed_frames": [],
            }
            print(f"  cached: {len(frames)} images", flush=True)
            write_status(status_path, status)
            continue

        video_path = video_root / f"{video_id}.mp4"
        try:
            readable, cached_codec = cached_video_is_usable(video_path)
            download_info: Dict[str, object] = {
                "cached_video": readable,
                "cached_codec": cached_codec,
            }
            if not readable:
                if video_path.exists():
                    video_path.unlink()
                target_height = int(metadata[video_id]["res"][0])
                download_info.update(
                    download_h264_video(
                        video_id,
                        video_path,
                        target_height,
                        args.retries,
                    )
                )

            extraction = extract_frames(
                video_path=video_path,
                frame_rows=frames,
                output_dir=output_dir,
                source_fps=float(metadata[video_id]["fps"]),
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite_images,
            )
            if extraction["failed_frames"]:
                raise RuntimeError(
                    f"failed to decode {len(extraction['failed_frames'])}/"
                    f"{len(frames)} annotated frames"
                )
            status["completed"][video_id] = {**download_info, **extraction}
            status["failed"].pop(video_id, None)
            print(
                f"  done: written={extraction['written']} cached={extraction['cached']}",
                flush=True,
            )
            if args.delete_videos:
                video_path.unlink(missing_ok=True)
        except Exception as error:
            status["failed"][video_id] = f"{type(error).__name__}: {error}"
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
        write_status(status_path, status)

    print(
        json.dumps(
            {
                "requested": status["requested"],
                "completed": len(status["completed"]),
                "failed": len(status["failed"]),
                "status_json": str(status_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
