"""Assemble, inspect, package and open the independent v3 stage film.

Four approved v2 video chapters are copied without decoding/re-encoding. New
chapters are frame-normalized. Only video and narration are muxed into the MP4.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import wave
import zipfile

from communication.video.stage.plan import CHAPTERS, ROOT, OUT, V2, FPS, RATE, PRESERVED_SHA256
from communication.video.production.assemble import metadata

FILENAME = "Learning_While_Acting_v3_Stage_1080p60.mp4"
BASE_REF = "77ec519"
PRESERVED_SOURCES = (
    "communication/video/film_plan.py", "communication/video/narration-v4.en.txt",
    "communication/video/scenes/ant_drawing.py", "communication/video/scenes/ants_learning.py",
    "communication/video/production/toy_motion.py", "communication/video/production/audio.py",
    "communication/video/production/retime_audio.py", "communication/video/production/assemble.py",
    "scripts/video/render.sh", "scripts/video/open-review.sh",
)


@dataclass(frozen=True)
class Receipt:
    scene: str
    seconds: int
    frames: int
    origin: str
    source_sha256: str
    normalized_sha256: str
    exact_copy: bool


def run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args[0]}\n{result.stderr[-6000:]}")


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def preserved() -> None:
    if sha(V2 / "Learning_While_Acting_v2_1080p60.mp4") != PRESERVED_SHA256:
        raise RuntimeError("The approved v2 master was modified")
    for name in PRESERVED_SOURCES:
        original = subprocess.check_output(["git", "show", f"{BASE_REF}:{name}"], cwd=ROOT)
        if original != (ROOT / name).read_bytes():
            raise RuntimeError(f"A preserved v2 source was modified: {name}")


def check() -> None:
    preserved()
    final = OUT / FILENAME
    data = metadata(final)
    if sorted(s["codec_type"] for s in data["streams"]) != ["audio", "video"]:
        raise RuntimeError("The video must contain only one video and one audio stream")
    visual = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    if (visual.get("width"), visual.get("height"), visual.get("r_frame_rate")) != (1920, 1080, "60/1"):
        raise RuntimeError("The stage master must be 1080p60")
    if int(visual.get("nb_frames", "0")) != 10800:
        raise RuntimeError("The stage master must contain exactly 10800 frames")
    if abs(float(audio.get("duration", "0")) - 180.) > .04:
        raise RuntimeError("The audio is not 180 seconds")
    if abs(float(data["format"]["duration"]) - 180.) > .04:
        raise RuntimeError("The container is not 180 seconds")
    with wave.open(str(OUT / "narration.en.wav")) as source:
        if source.getnframes() != RATE * 180 or source.getframerate() != RATE:
            raise RuntimeError("Narration does not have an exact sample count")
    for chapter in CHAPTERS:
        if chapter.reuse_chapter is not None:
            expected = V2 / f"normalized-hd/{chapter.scene}.mp4"
            actual = OUT / f"normalized-hd/{chapter.scene}.mp4"
            if sha(expected) != sha(actual):
                raise RuntimeError(f"Approved core clip {chapter.scene} was changed")
            index = CHAPTERS.index(chapter) + 1
            if sha(V2 / f"audio/s{chapter.reuse_chapter:02}-timed.wav") != sha(OUT / f"audio/p{index:02}-timed.wav"):
                raise RuntimeError(f"Approved core voice {chapter.scene} was changed")
    run(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"])
    print("PASS: 180s / 10800 frames / 1080p60 / audio+video only / full decode.")
    print("PASS: v2 master and source files unchanged; four core clips and voices exact copies.")


def player() -> None:
    buttons = []
    position = 0
    for chapter in CHAPTERS:
        buttons.append(f'<button data-time="{position}"><time>{position // 60:02}:{position % 60:02}</time>{html.escape(chapter.title)}</button>')
        position += chapter.seconds
    template = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning While Acting | Stage cut v3</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0e16;color:#f2eee6;font-family:system-ui,sans-serif}main{max-width:1480px;padding:28px;margin:auto}small{color:#57d6bd;letter-spacing:2px}h1{font-size:38px;font-weight:500;margin:12px 0}p{color:#a4a9b5;line-height:1.6}video{display:block;width:100%;max-height:78vh;background:#0b0e16;margin:22px 0}.chapters{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:8px}button{display:flex;gap:14px;background:#151c28;border:1px solid #2c3849;color:#f2eee6;padding:14px;text-align:left;font:14px system-ui;cursor:pointer}button:hover{border-color:#57d6bd}time{color:#57d6bd;white-space:nowrap}footer{font-size:13px;color:#8995a6;line-height:1.8;margin:24px 0}a{color:#70b8e8}
</style></head><body><main><small>MATHHACKSON / STAGE CUT 03</small><h1>Learning While Acting</h1><p>3:00 · 1080p60 · English narration · No embedded or burned-in captions<br>Question-led introduction, moving-ant demonstrations, and a return to the architecture.</p><video id="film" controls preload="metadata" src="__FILE__"></video><div class="chapters">__CHAPTERS__</div><footer>Architecture proposal and teaching simulations, not benchmark results. The approved v2 has been preserved.<br>Voice: synthetic Microsoft en-US-AndrewNeural. Original ManimGL scenes; no copied 3Blue1Brown footage or cloned voice.<br><a href="Learning_While_Acting.en.srt" download>Separate English subtitles</a> · <a href="ManimGL_sources_v3.zip" download>Source package</a></footer></main><script>
const video=document.getElementById('film');for(const button of document.querySelectorAll('[data-time]'))button.onclick=()=>{video.currentTime=Number(button.dataset.time);video.play();video.scrollIntoView({behavior:'smooth',block:'center'});};
</script></body></html>'''
    (OUT / "index.html").write_text(template.replace("__FILE__", FILENAME).replace("__CHAPTERS__", "\n".join(buttons)), encoding="utf-8")


def assemble() -> None:
    preserved()
    normalized = OUT / "normalized-hd"
    normalized.mkdir(parents=True, exist_ok=True)
    receipts: list[Receipt] = []
    listing: list[str] = []
    for chapter in CHAPTERS:
        reused = chapter.reuse_chapter is not None
        source = (V2 / "normalized-hd" if reused else OUT / "clips-hd") / (chapter.scene + ".mp4")
        target = normalized / source.name
        data = metadata(source)
        stream = next(s for s in data["streams"] if s["codec_type"] == "video")
        if (stream.get("width"), stream.get("height"), stream.get("r_frame_rate")) != (1920, 1080, "60/1"):
            raise RuntimeError(f"Bad source geometry: {source}")
        if abs(float(data["format"]["duration"]) - chapter.seconds) > 2 / FPS:
            raise RuntimeError(f"Bad source duration: {source}")
        if reused:
            shutil.copy2(source, target)
        else:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-an", "-vf",
                 f"trim=end_frame={chapter.seconds * FPS},setpts=PTS-STARTPTS", "-c:v", "libx264",
                 "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", str(target)])
        frames = int(metadata(target)["streams"][0].get("nb_frames", "0"))
        if frames != chapter.seconds * FPS:
            raise RuntimeError(f"Wrong frame count: {source.name}")
        receipts.append(Receipt(chapter.scene, chapter.seconds, frames,
                                str(source.relative_to(ROOT)), sha(source), sha(target), reused))
        listing.append(f"file '{target}'\n")
    concat = OUT / "concat-hd.txt"
    concat.write_text("".join(listing))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-i", str(OUT / "narration.en.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-metadata",
         "title=Learning While Acting | Stage cut v3 | MathHackson", str(OUT / FILENAME)])
    check()
    manifest = {
        "title": "Learning While Acting", "version": "v3-stage", "duration": 180,
        "width": 1920, "height": 1080, "fps": FPS, "frames": 10800,
        "status": "architecture proposal and teaching simulations, not benchmark results",
        "captions": "sidecar SRT only", "voice": "synthetic Microsoft en-US-AndrewNeural",
        "preserved_v2_sha256": PRESERVED_SHA256, "sha256": sha(OUT / FILENAME),
        "bytes": (OUT / FILENAME).stat().st_size,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "chapters": [asdict(c) for c in receipts],
    }
    (OUT / "render-manifest.json").write_text(json.dumps(manifest, indent=2))
    player()
    print(f"Ready: {OUT / FILENAME}\n{manifest['bytes'] / 1048576:.2f} MiB")


def package() -> None:
    names = subprocess.check_output(["git", "ls-files", "communication/video", "scripts/video"], cwd=ROOT, text=True).splitlines()
    allowed = {".py", ".sh", ".md", ".txt"}
    destination = OUT / "ManimGL_sources_v3.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = ROOT / name
            if path.is_file() and path.suffix in allowed and "rendered" not in path.parts:
                archive.write(path, name)
    print(f"Sources: {destination}; code and narration only, no fonts or model files.")


def review() -> None:
    check()
    package()
    profile = subprocess.check_output(["cmd.exe", "/c", "echo %USERPROFILE%"], text=True, stderr=subprocess.DEVNULL).strip()
    user_root = Path(subprocess.check_output(["wslpath", "-u", profile], text=True).strip())
    downloads = user_root / "Downloads"
    if not downloads.is_dir():
        raise RuntimeError("Windows Downloads directory could not be resolved")
    destination = downloads / "MathHackson_Learning_While_Acting_v3"
    if destination.exists():
        destination = downloads / (destination.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir()
    for name in (FILENAME, "index.html", "render-manifest.json", "Learning_While_Acting.en.srt",
                 "speech-visual-timeline.json", "ManimGL_sources_v3.zip"):
        shutil.copy2(OUT / name, destination / name)
    if sha(destination / FILENAME) != sha(OUT / FILENAME):
        raise RuntimeError("Windows copy does not match rendered master")
    windows = subprocess.check_output(["wslpath", "-w", str(destination / "index.html")], text=True).strip()
    escaped = windows.replace("'", "''")
    run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", f"Start-Process -FilePath '{escaped}'"])
    log = ROOT / "logs/video-v3/windows-review-path.txt"
    log.write_text(str(destination) + "\n")
    print(f"Opened: {windows}")
    print("Video: " + subprocess.check_output(["wslpath", "-w", str(destination / FILENAME)], text=True).strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("assemble", "check", "review", "package"))
    action = parser.parse_args().action
    {"assemble": assemble, "check": check, "review": review, "package": package}[action]()
