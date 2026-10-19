"""Sequential SimulStreaming Canary runner for Earnings25 audio."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SIMULSTREAMING_ROOT = WORKSPACE_ROOT / "SimulStreaming"
DEFAULT_ARCHIVE_URL = "https://zenodo.org/records/18762168/files/earnings25.zip?download=1"
DEFAULT_EXTRACT_DIR = WORKSPACE_ROOT / "data"
DEFAULT_ARCHIVE = DEFAULT_EXTRACT_DIR / "earnings25.zip"
DEFAULT_AUDIO_DIR = DEFAULT_EXTRACT_DIR / "earnings-25" / "testset-full" / "audio"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "canary_output"


if str(SIMULSTREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULSTREAMING_ROOT))

from simulstreaming_canary import SimulCanaryOnline, simul_asr_factory  # noqa: E402
from simulstreaming.whisper.whisper_streaming.whisper_online_main import (  # noqa: E402
    load_audio,
    load_audio_chunk,
)


@dataclass
class SegmentRecord:
    start_ms: int
    end_ms: int
    emission_ms: float
    text: str


@dataclass
class AudioJob:
    audio_path: Path
    output_path: Path
    duration_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimulStreaming Canary sequentially on Earnings25 audio.")
    parser.add_argument("--download", action="store_true", help="Download and extract the dataset if needed.")
    parser.add_argument("--download-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to output_{target language}, for example output_cs.",
    )
    parser.add_argument("--start-hours", type=float, default=0.0, help="Start of the cumulative audio window.")
    parser.add_argument("--hours", type=float, default=None, help="Length of the cumulative audio window.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="cs", help="Use cs for Czech.")
    parser.add_argument(
        "--task",
        choices=["asr", "ast", "transcribe", "translate", "s2t_translation"],
        default="translate",
    )
    parser.add_argument("--model-path", default=None, help="Optional local Canary .nemo model.")
    parser.add_argument("--min-chunk-size", type=float, default=5.0)
    parser.add_argument("--audio-max-len", type=float, default=15.0)
    parser.add_argument("--frame-threshold", type=int, default=4)
    parser.add_argument("--max-context-len", type=int, default=256)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--decoder", choices=["beam", "greedy"], default="greedy")
    parser.add_argument("--strip-incomplete-words", action="store_true")
    parser.add_argument("--decoder-context", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "-l",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    return parser


def download_and_extract(url: str, archive: Path, extract_dir: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading dataset to {archive} ...", flush=True)
        urllib.request.urlretrieve(url, archive)
    else:
        print(f"Using existing archive: {archive}", flush=True)

    expected_audio = extract_dir / "earnings-25" / "testset-full" / "audio"
    marker = extract_dir / ".earnings25_extracted"
    if expected_audio.exists() and any(expected_audio.glob("*.mp3")):
        print(f"Using existing extraction: {extract_dir}", flush=True)
        return

    print(f"Extracting {archive} ...", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(extract_dir)
    marker.write_text("ok\n", encoding="ascii")


def collect_audio_files(audio_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.mp3" if recursive else "*.mp3"
    files = sorted(path for path in audio_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No MP3 files found under {audio_dir}")
    return files


def get_duration(path: Path) -> float:
    return float(__import__("librosa").get_duration(path=str(path)))


def select_jobs(
    audio_files: list[Path],
    output_dir: Path,
    start_hours: float,
    hours: float | None,
    overwrite: bool,
) -> list[AudioJob]:
    if start_hours < 0:
        raise ValueError("--start-hours must be non-negative")
    if hours is not None and hours <= 0:
        raise ValueError("--hours must be greater than zero")

    window_start = start_hours * 3600.0
    window_end = None if hours is None else window_start + hours * 3600.0
    cumulative = 0.0
    jobs: list[AudioJob] = []

    for audio_path in audio_files:
        duration = get_duration(audio_path)
        segment_start = cumulative
        cumulative += duration

        if segment_start < window_start:
            continue
        if window_end is not None and segment_start >= window_end:
            break

        output_path = output_dir / f"{audio_path.stem}.json"
        if output_path.exists() and not overwrite:
            continue
        jobs.append(AudioJob(audio_path, output_path, duration))

    selected_hours = sum(job.duration_seconds for job in jobs) / 3600.0
    print(
        f"Selected {len(jobs)} files ({selected_hours:.2f} hours), "
        f"window {start_hours:.2f}h to {('end' if hours is None else f'{start_hours + hours:.2f}h')}.",
        flush=True,
    )
    return jobs


def run_one_file(asr, job: AudioJob, args: argparse.Namespace) -> dict:
    online = SimulCanaryOnline(asr)
    online.init(offset=0.0)

    duration = job.duration_seconds
    begin = 0.0
    end = min(duration, begin + args.min_chunk_size)
    segments: list[SegmentRecord] = []

    while True:
        online.insert_audio_chunk(load_audio_chunk(str(job.audio_path), begin, end))
        try:
            output = online.process_iter()
        except AssertionError:
            output = None

        if output and output.get("text"):
            segments.append(
                SegmentRecord(
                    start_ms=round(output["start"] * 1000),
                    end_ms=round(output["end"] * 1000),
                    emission_ms=end * 1000.0,
                    text=output["text"],
                )
            )

        if end >= duration:
            break
        begin = end
        end = min(duration, end + args.min_chunk_size)

    final_output = online.finish()
    if final_output and final_output.get("text"):
        segments.append(
            SegmentRecord(
                start_ms=round(final_output["start"] * 1000),
                end_ms=round(final_output["end"] * 1000),
                emission_ms=0.0,
                text=final_output["text"],
            )
        )

    return {
        "audio_path": str(job.audio_path),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "task": args.task,
        "duration_seconds": job.duration_seconds,
        "final_text": " ".join(segment.text for segment in segments).strip(),
        "segments": [asdict(segment) for segment in segments],
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s\t%(message)s")
    logging.getLogger("numba").setLevel(logging.WARNING)

    if args.output_dir is None:
        args.output_dir = WORKSPACE_ROOT / f"output_{args.target_lang}"

    if args.download or not args.audio_dir.exists() or not any(args.audio_dir.glob("*.mp3")):
        download_and_extract(args.download_url, args.archive, args.extract_dir)

    audio_files = collect_audio_files(args.audio_dir, args.recursive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(audio_files, args.output_dir, args.start_hours, args.hours, args.overwrite)
    if not jobs:
        print("Nothing to process.")
        return

    asr_args = argparse.Namespace(
        model_path=args.model_path,
        decoder=args.decoder,
        frame_threshold=args.frame_threshold,
        max_context_len=args.max_context_len,
        audio_max_len=args.audio_max_len,
        beams=args.beams,
        task=args.task,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        strip_incomplete_words=args.strip_incomplete_words,
        decoder_context=args.decoder_context,
        log_level=args.log_level,
    )
    asr, _ = simul_asr_factory(asr_args)
    print(f"Canary model device: {next(asr.model.parameters()).device}", flush=True)

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Processing {job.audio_path.name}", flush=True)
        result = run_one_file(asr, job, args)
        temporary_path = job.output_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(job.output_path)
        print(f"[{index}/{len(jobs)}] Saved {job.output_path}", flush=True)


if __name__ == "__main__":
    main()
