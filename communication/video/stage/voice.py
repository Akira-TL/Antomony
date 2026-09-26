"""Build v3 English audio; retain the four approved core recordings verbatim.

Sentence cuts use the synthesizer's boundaries, not forced phoneme alignment.
All output is under rendered/v3. The v2 media and timeline are never rewritten.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import shutil
import textwrap
from typing import TypedDict, cast

import edge_tts
import numpy as np

from communication.video.production.retime_audio import load, save, run, timestamp
from communication.video.film_plan import CHAPTERS as V2_CHAPTERS
from communication.video.stage.plan import CHAPTERS, NARRATION, OUT, V2, RATE, VOICE, ROOT, PRESERVED_SHA256


@dataclass(frozen=True)
class Boundary:
    start: float
    duration: float
    text: str


class Cache(TypedDict):
    identity: str
    voice: str
    text: str
    boundaries: list[dict[str, float | str]]


@dataclass(frozen=True)
class Cue:
    chapter: int
    sentence: int
    start: float
    end: float
    tempo: float
    text: str
    reused: bool = False


def verify_v2() -> None:
    master = V2 / "Learning_While_Acting_v2_1080p60.mp4"
    if hashlib.sha256(master.read_bytes()).hexdigest() != PRESERVED_SHA256:
        raise RuntimeError("Approved v2 master identity changed; stop rather than overwrite it")


async def raw_speech(index: int, paragraph: str) -> tuple[np.ndarray, list[Boundary]]:
    directory = OUT / "audio"
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"p{index:02}"
    raw = stem.with_suffix(".mp3")
    meta = stem.with_suffix(".json")
    identity = hashlib.sha256((VOICE + "\n" + paragraph).encode()).hexdigest()
    cached = cast(Cache, json.loads(meta.read_text())) if meta.exists() else None
    if cached is None or cached["identity"] != identity or not raw.exists():
        partial = stem.with_suffix(".partial.mp3")
        boundaries: list[Boundary] = []
        for attempt in range(2):
            try:
                boundaries.clear()
                speech = edge_tts.Communicate(paragraph, VOICE, rate="+0%")
                with partial.open("wb") as audio:
                    async for chunk in speech.stream():
                        if chunk["type"] == "audio":
                            audio.write(chunk["data"])
                        elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                            boundaries.append(Boundary(float(chunk["offset"]) / 1e7,
                                                       float(chunk["duration"]) / 1e7,
                                                       html.unescape(str(chunk["text"]))))
                if partial.stat().st_size < 1000 or not boundaries:
                    raise RuntimeError("Missing speech audio or timing metadata")
                partial.replace(raw)
                break
            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(2)
        cached = cast(Cache, {"identity": identity, "voice": VOICE, "text": paragraph,
                              "boundaries": [asdict(b) for b in boundaries]})
        meta.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    boundaries = [Boundary(float(b["start"]), float(b["duration"]), str(b["text"]))
                  for b in cached["boundaries"]]
    decoded = stem.with_name(stem.name + "-raw.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ar", str(RATE), "-ac", "1",
         "-c:a", "pcm_s16le", str(decoded)])
    return load(decoded), boundaries


async def main() -> None:
    verify_v2()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    paragraphs = [p.strip() for p in NARRATION.read_text().split("\n\n") if p.strip()]
    previous = [p.strip() for p in (ROOT / "communication/video/narration-v4.en.txt").read_text().split("\n\n") if p.strip()]
    if len(paragraphs) != len(CHAPTERS):
        raise ValueError("One approved paragraph is required per v3 chapter")
    old_timeline = json.loads((V2 / "speech-visual-timeline.json").read_text())
    timeline: list[Cue] = []
    chunks: list[np.ndarray] = []
    offset = 0
    for i, (chapter, paragraph) in enumerate(zip(CHAPTERS, paragraphs), 1):
        target = OUT / f"audio/p{i:02}-timed.wav"
        if chapter.reuse_chapter is not None:
            old = chapter.reuse_chapter
            if paragraph != previous[old - 1] or chapter.seconds != V2_CHAPTERS[old - 1].seconds:
                raise ValueError(f"Core paragraph {old} changed; cannot silently reuse its audio")
            source = V2 / f"audio/s{old:02}-timed.wav"
            shutil.copy2(source, target)
            samples = load(target)
            old_offset = sum(c.seconds for c in V2_CHAPTERS[:old - 1])
            for cue in old_timeline:
                if cue["chapter"] == old:
                    timeline.append(Cue(i, int(cue["sentence"]),
                                        offset + float(cue["start"]) - old_offset,
                                        offset + float(cue["end"]) - old_offset,
                                        float(cue["tempo"]), str(cue["text"]), True))
            print(f"{chapter.scene}: reused approved voice, {chapter.seconds}s", flush=True)
        else:
            source_samples, boundaries = await raw_speech(i, paragraph)
            if len(boundaries) != len(chapter.starts):
                raise ValueError(f"{chapter.scene}: {len(boundaries)} speech sentences != {len(chapter.starts)} visual cues")
            cuts = [0]
            for left, right in zip(boundaries, boundaries[1:]):
                cuts.append(round((left.start + left.duration + right.start) / 2 * RATE))
            cuts.append(len(source_samples))
            samples = np.zeros(chapter.seconds * RATE, dtype=np.int16)
            for j, (boundary, start) in enumerate(zip(boundaries, chapter.starts)):
                segment = source_samples[cuts[j]:cuts[j + 1]]
                next_start = chapter.starts[j + 1] if j + 1 < len(chapter.starts) else chapter.seconds - .12
                available = next_start - start - .035
                ratio = max(1., len(segment) / RATE / available)
                if ratio > 1.18:
                    raise ValueError(f"{chapter.scene} sentence {j}: adjust the visual beat, not a rushed voice ({ratio:.3f})")
                if ratio > 1.0:
                    before = OUT / f"audio/p{i:02}-part{j}-raw.wav"
                    after = OUT / f"audio/p{i:02}-part{j}-fit.wav"
                    save(before, segment)
                    run(["ffmpeg", "-y", "-v", "error", "-i", str(before), "-af", f"atempo={ratio * 1.007:.8f}",
                         "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", str(after)])
                    segment = load(after)
                begin = round(start * RATE)
                end = begin + len(segment)
                if end >= round(next_start * RATE):
                    raise ValueError(f"Overlapping speech in {chapter.scene}")
                fade = min(120, len(segment) // 4)
                floating = segment.astype(float)
                floating[:fade] *= np.linspace(0, 1, fade)
                floating[-fade:] *= np.linspace(1, 0, fade)
                samples[begin:end] = np.clip(floating, -32768, 32767).astype(np.int16)
                timeline.append(Cue(i, j, offset + start, offset + end / RATE, ratio, boundary.text))
            save(target, samples)
            print(f"{chapter.scene}: {len(boundaries)} sentences placed, {chapter.seconds}s", flush=True)
        if len(samples) != chapter.seconds * RATE:
            raise ValueError("Chapter audio sample count mismatch")
        chunks.append(samples)
        offset += chapter.seconds
    combined = np.concatenate(chunks)
    if len(combined) != 180 * RATE:
        raise ValueError("Narration is not exactly 180 seconds")
    unnormalized = OUT / "audio/combined.wav"
    save(unnormalized, combined)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(unnormalized), "-af",
         "loudnorm=I=-17:TP=-1.5:LRA=11,aresample=48000,asetpts=PTS-STARTPTS,apad=whole_len=8640000,atrim=end_sample=8640000",
         "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", str(OUT / "narration.en.wav")])
    if len(load(OUT / "narration.en.wav")) != 180 * RATE:
        raise ValueError("Normalized narration sample count mismatch")
    subtitles = [f"{i}\n{timestamp(c.start)} --> {timestamp(c.end)}\n{textwrap.fill(c.text, width=58)}\n"
                 for i, c in enumerate(timeline, 1)]
    (OUT / "Learning_While_Acting.en.srt").write_text("\n".join(subtitles), encoding="utf-8")
    (OUT / "speech-visual-timeline.json").write_text(json.dumps([asdict(c) for c in timeline], indent=2))
    verify_v2()
    print("v3 voice ready: 180 seconds; four core recordings unchanged; SRT sidecar only.")


if __name__ == "__main__":
    asyncio.run(main())
