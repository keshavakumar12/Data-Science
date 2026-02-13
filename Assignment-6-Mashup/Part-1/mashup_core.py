from __future__ import annotations

import logging
import os
import re
import tempfile
import warnings
import subprocess
from pathlib import Path
from typing import Iterable, List
import time

warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv",
    category=RuntimeWarning,
    module="pydub.utils",
)

import imageio_ffmpeg
import yt_dlp


log = logging.getLogger(__name__)


class MashupError(Exception):
    """Custom exception for mashup-related failures."""


def _coerce_positive_int(value: str, name: str) -> int:
    try:
        as_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if as_int <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return as_int


def validate_inputs(singer: str, n_videos: int, clip_seconds: int) -> None:
    if not singer or not singer.strip():
        raise ValueError("Singer name cannot be empty.")
    if n_videos <= 10:
        raise ValueError("Number of videos must be greater than 10.")
    if clip_seconds <= 20:
        raise ValueError("Audio duration must be greater than 20 seconds.")


def setup_audio_backend() -> None:
    """
    Ensure ffmpeg is available. imageio_ffmpeg will fetch a local
    ffmpeg binary if one is not already available on the system PATH.
    """
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ.setdefault("FFMPEG_BINARY", ffmpeg_path)
        os.environ.setdefault("FFPROBE_BINARY", ffmpeg_path)
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_path)
        os.environ.setdefault("FFMPEG_LOCATION", ffmpeg_path)
        # put ffmpeg directory on PATH so subprocess calls can find it
        os.environ["PATH"] = str(Path(ffmpeg_path).parent) + os.pathsep + os.environ.get("PATH", "")
        log.debug("Using ffmpeg binary at %s", ffmpeg_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise MashupError("Unable to locate or download ffmpeg.") from exc


def _search_candidates(singer: str, n_videos: int) -> List[dict]:
    search_query = f"ytsearch{max(n_videos * 2, 20)}:{singer} songs"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "default_search": "auto",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        search_result = ydl.extract_info(search_query, download=False)
    entries = search_result.get("entries") or []
    return [entry for entry in entries if entry]


def download_audio_from_search(
    singer: str, n_videos: int, workdir: Path
) -> List[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    candidates = _search_candidates(singer, n_videos)
    if not candidates:
        raise MashupError("No videos found for the provided singer name.")

    def _wait_for_stable(path: Path, attempts: int = 20, delay: float = 0.5) -> None:
        last_size = -1
        for _ in range(attempts):
            if not path.exists():
                time.sleep(delay)
                continue
            size = path.stat().st_size
            if size == last_size and size > 0:
                return
            last_size = size
            time.sleep(delay)
        raise MashupError(f"File did not stabilize: {path}")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    downloaded: List[Path] = []
    for entry in candidates:
        if len(downloaded) >= n_videos:
            break

        url = entry.get("webpage_url")
        video_id = entry.get("id")
        if not url or not video_id:
            continue

        outtmpl = str(workdir / f"{video_id}.%(ext)s")
        opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "noplaylist": True,
            "outtmpl": outtmpl,
            "retries": 3,
            "ffmpeg_location": ffmpeg_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            # set to remove original video after audio extraction (default False)
            "keepvideo": False,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:  # continue on individual download failures
            log.warning("Failed to download %s: %s", url, exc)
            continue

        # prefer the extracted mp3; fallback to any file with the id
        matched = list(workdir.glob(f"{video_id}.mp3"))
        if not matched:
            matched = list(workdir.glob(f"{video_id}.*"))
        if matched:
            target = matched[0]
            try:
                _wait_for_stable(target)
                downloaded.append(target)
            except Exception as exc:
                log.warning("Skipping locked/unready file %s: %s", target, exc)
                continue

    if len(downloaded) < n_videos:
        raise MashupError(
            f"Downloaded only {len(downloaded)} of {n_videos} requested videos. "
            "Try again with a smaller number or a different singer."
        )
    return downloaded


def trim_audios(
    audio_paths: Iterable[Path], clip_seconds: int, target_dir: Path
) -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    trimmed_files: List[Path] = []
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    def _wait_for_unlock(file_path: Path, attempts: int = 60, delay: float = 1.0) -> None:
        for _ in range(attempts):
            try:
                with file_path.open("rb"):
                    return
            except PermissionError:
                time.sleep(delay)
        raise PermissionError(f"File remains locked after retries: {file_path}")

    for path in audio_paths:
        try:
            _wait_for_unlock(path)
            trimmed_path = target_dir / f"{path.stem}_trimmed.mp3"
            cmd = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "0",
                "-t",
                str(clip_seconds),
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "mp3",
                "-b:a",
                "192k",
                str(trimmed_path),
            ]
            subprocess.run(cmd, check=True)
            trimmed_files.append(trimmed_path)
        except Exception as exc:
            log.warning("Skipping locked/unreadable file %s: %s", path, exc)
            continue

    if not trimmed_files:
        raise MashupError("No audio files were trimmed.")
    return trimmed_files


def merge_audios(trimmed_paths: Iterable[Path], output_path: Path) -> Path:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    trimmed_list = [Path(p) for p in trimmed_paths]
    if not trimmed_list:
        raise MashupError("No trimmed audio files to merge.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=output_path.parent, suffix=".txt", newline=""
    ) as list_file:
        for path in trimmed_list:
            safe_path = path.resolve().as_posix().replace("'", "'\\''")
            list_file.write(f"file '{safe_path}'\n")
        list_path = Path(list_file.name)

    try:
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass

    return output_path


def create_mashup(
    singer: str, n_videos: int, clip_seconds: int, output_path: Path | str
) -> Path:
    """
    Public API to create the mashup audio file.
    """
    n_videos = _coerce_positive_int(n_videos, "Number of videos")
    clip_seconds = _coerce_positive_int(clip_seconds, "Audio duration")
    validate_inputs(singer, n_videos, clip_seconds)

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp3":
        output_path = output_path.with_suffix(".mp3")

    setup_audio_backend()

    with tempfile.TemporaryDirectory(prefix="mashup_") as temp_root:
        temp_root = Path(temp_root)
        download_dir = temp_root / "downloads"
        trimmed_dir = temp_root / "trimmed"

        downloads = download_audio_from_search(singer, n_videos, download_dir)
        # pause to let OS/AV finish touching the downloaded files (Windows)
        time.sleep(5.0)
        trimmed = trim_audios(downloads, clip_seconds, trimmed_dir)
        merged_path = merge_audios(trimmed, output_path)

    return merged_path


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.match(email.strip()))
