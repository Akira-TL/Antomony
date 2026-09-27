"""Independent v3 timing; the approved v2 plan and media are read-only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "communication/video/rendered/v3"
V2 = ROOT / "communication/video/rendered/v2"
NARRATION = ROOT / "communication/video/plans/narration-stage-v3.en.txt"
PRESERVED_SHA256 = "4e7d70eaf9a4336972ff84a09a01d6a132ef57692958e24ff324708fcb61bedf"
VOICE = "en-US-AndrewNeural"
RATE = 48000
FPS = 60


@dataclass(frozen=True)
class Chapter:
    scene: str
    title: str
    seconds: int
    starts: tuple[float, ...] = ()
    reuse_chapter: int | None = None


CHAPTERS = (
    Chapter("B00TheQuestion", "Training finished. Learning finished?", 30,
            (.4, 1.95, 4.6, 8.5, 10.7, 18.1, 22.9, 24.55)),
    Chapter("B01FollowTheTrail", "A familiar skill, a changing world", 14, (.4, 3.7, 7.0)),
    Chapter("B02TheTwin", "Same starting rule. Different learning.", 12, (.4, 1.85, 4.4, 6.95, 9.4)),
    Chapter("B03OneCorrection", "One number changes the movement", 15, (.4, 2.45, 7.7, 11.7)),
    Chapter("A04LearnFromTheGap", "A local correction", 20, reuse_chapter=4),
    Chapter("A05NoiseOrChange", "A gust, or a lasting change?", 24, reuse_chapter=5),
    Chapter("A06LearningTheChoice", "The update rule learns too", 28, reuse_chapter=6),
    Chapter("B07Withdraw", "Undo the change, not the past", 9, (.4, 3.0, 6.1)),
    Chapter("A08KeepWalking", "The next action cannot wait", 20, reuse_chapter=8),
    Chapter("B09TheArchitecture", "Learning While Acting", 8, (.4, 2.3)),
)

if sum(c.seconds for c in CHAPTERS) != 180:
    raise ValueError("The stage film must be exactly three minutes")

NEW_SCENES = tuple(c.scene for c in CHAPTERS if c.reuse_chapter is None)
