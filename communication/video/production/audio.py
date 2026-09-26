"""Synthesize the approved English narration and fit it to the ten scene slots.

Run through scripts/video-audio.sh. No narration is silently truncated.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import sys
import subprocess
import textwrap
import wave

import edge_tts

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from communication.video.film_plan import DURATIONS, CHAPTERS
OUT = ROOT / "communication/video/rendered/v2/audio"
VOICE = "en-US-AndrewNeural"


@dataclass
class Boundary:
    start: float
    duration: float
    text: str


@dataclass
class Speech:
    scene: int
    source_hash: str
    voice: str
    raw_seconds: float
    slot_seconds: float
    tempo: float
    onset: float
    boundaries: list[Boundary]


def command(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


def duration(path: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip())


def stamp(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def fit_audio(raw: Path, slot: int, tempo: float, onset: float) -> None:
    expected_samples = slot * 48000
    filters = (
        f"atempo={tempo:.7f},loudnorm=I=-16:TP=-1.5:LRA=11,"
        f"aresample=48000,asetpts=PTS-STARTPTS,adelay={round(onset * 1000)}:all=1,"
        f"apad=whole_len={expected_samples},atrim=end_sample={expected_samples},asetpts=PTS-STARTPTS"
    )
    target = raw.with_suffix(".wav")
    command(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", filters,
             "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(target)])
    with wave.open(str(target)) as audio:
        if audio.getnframes() != expected_samples or audio.getframerate() != 48000:
            raise RuntimeError(f"Audio sample count is not exact for {target.name}")


async def synthesize(index: int, paragraph: str, force: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"s{index + 1:02}.mp3"
    meta = raw.with_suffix(".json")
    identity = hashlib.sha256((VOICE + paragraph).encode()).hexdigest()
    if not force and raw.exists() and meta.exists() and raw.with_suffix(".wav").exists():
        previous = json.loads(meta.read_text())
        if previous["source_hash"] == identity:
            fit_audio(raw, DURATIONS[index], previous["tempo"], previous["onset"])
            print(f"S{index + 1:02}: cached voice, exact sample-count refit", flush=True)
            return
    boundaries: list[Boundary] = []
    temporary = raw.with_suffix(".partial.mp3")
    for attempt in range(2):
        try:
            boundaries.clear()
            with temporary.open("wb") as audio:
                speech = edge_tts.Communicate(paragraph, VOICE, rate="+0%")
                async for chunk in speech.stream():
                    if chunk["type"] == "audio":
                        audio.write(chunk["data"])
                    elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                        boundaries.append(Boundary(
                            float(chunk["offset"]) / 1e7,
                            float(chunk["duration"]) / 1e7,
                            html.unescape(str(chunk["text"]))
                        ))
            if temporary.stat().st_size < 1000:
                raise RuntimeError("No usable speech returned")
            temporary.replace(raw)
            break
        except Exception:
            if attempt == 1:
                raise
            await asyncio.sleep(2)
    raw_duration = duration(raw)
    slot = DURATIONS[index]
    onset = 0.35
    tempo = max(1.0, raw_duration / (slot - 1.1))
    if tempo > 1.22:
        raise RuntimeError(f"S{index + 1}: narration too long for natural speech ({raw_duration:.1f}s)")
    fit_audio(raw, slot, tempo, onset)
    result = Speech(index + 1, identity, VOICE, raw_duration, slot, tempo, onset, boundaries)
    meta.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"S{index + 1:02}: voice={raw_duration:.2f}s / slot={slot}s / tempo={tempo:.3f}", flush=True)


def assemble_subtitles(paragraphs: list[str]) -> None:
    cues: list[str] = []
    offset = 0.0
    for i in range(len(paragraphs)):
        data = json.loads((OUT / f"s{i + 1:02}.json").read_text())
        # Derive phrase timings from the synthesizer's reported speech boundaries.
        words: list[Boundary] = []
        for item in data["boundaries"]:
            tokens = item["text"].split()
            for j, token in enumerate(tokens):
                unit = item["duration"] / max(1, len(tokens))
                words.append(Boundary(item["start"] + j * unit, unit, token))
        if not words:
            raise RuntimeError("Narration has no timing metadata")
        groups: list[list[Boundary]] = []
        current: list[Boundary] = []
        for word in words:
            current.append(word)
            count = len(" ".join(w.text for w in current))
            if count >= 56 or word.text.endswith((".", "?", "!", ";")):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        for group in groups:
            start = offset + data["onset"] + group[0].start / data["tempo"]
            end = offset + data["onset"] + (group[-1].start + group[-1].duration) / data["tempo"]
            end = min(offset + DURATIONS[i] - 0.15, end + 0.08)
            line = textwrap.fill(" ".join(w.text for w in group), width=52)
            cues.append(f"{len(cues) + 1}\n{stamp(start)} --> {stamp(end)}\n{line}\n")
        offset += DURATIONS[i]
    (OUT.parent / "Learning_While_Acting.en.srt").write_text("\n".join(cues), encoding="utf-8")
    listing = OUT / "audio-concat.txt"
    listing.write_text("".join(f"file '{OUT / f's{i + 1:02}.wav'}'\n" for i in range(len(CHAPTERS))))
    command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c:a", "pcm_s16le", str(OUT.parent / "narration.en.wav")])
    with wave.open(str(OUT.parent / "narration.en.wav")) as audio:
        if audio.getnframes() != 180 * 48000:
            raise RuntimeError("The complete narration must be exactly 180 seconds")
    print("Narration and subtitle timeline assembled: exactly 8640000 samples / 180 seconds.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paragraphs = [s.strip() for s in (ROOT / "communication/video/narration-v4.en.txt").read_text().split("\n\n") if s.strip()]
    if len(paragraphs) != len(CHAPTERS):
        raise ValueError("The approved narration must contain one paragraph per chapter")
    for i, paragraph in enumerate(paragraphs[:1] if args.test else paragraphs):
        await synthesize(i, paragraph, args.force)
    if not args.test:
        assemble_subtitles(paragraphs)


if __name__ == "__main__":
    asyncio.run(main())
