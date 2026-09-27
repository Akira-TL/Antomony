"""One timing source for the animated-ant film."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    scene: str
    title: str
    seconds: int


CHAPTERS = (
    Chapter("A01FollowTheTrail", "What changed?", 18),
    Chapter("A02SameAntDifferentLearning", "Watch the paths separate", 20),
    Chapter("A03WeightBecomesMovement", "A number becomes a movement", 22),
    Chapter("A04LearnFromTheGap", "The prediction and the gap", 20),
    Chapter("A05NoiseOrChange", "A gust, or a new rule?", 24),
    Chapter("A06LearningTheChoice", "Learning which changes to make", 28),
    Chapter("A07UndoOneChange", "Undo a change, not the past", 16),
    Chapter("A08KeepWalking", "The next step cannot wait", 20),
    Chapter("A09ReturnToTheAnts", "Learning While Acting", 12),
)
DURATIONS = tuple(ch.seconds for ch in CHAPTERS)
assert sum(DURATIONS) == 180

# Visual beat times. Speech is separated at the synthesizer's sentence boundaries,
# with original pauses preserved inside each sentence and room to watch between them.
SENTENCE_STARTS = (
    (0.40, 1.80, 2.55, 4.65, 6.50, 13.00),
    (0.40, 2.20, 5.00, 8.90, 12.00),
    (0.40, 2.45, 3.85, 6.45, 11.40, 16.60),
    (0.40, 2.25, 4.25, 6.00, 8.00, 13.10),
    (0.40, 3.80, 6.90, 12.10, 14.15, 17.60, 20.85),
    (0.40, 2.60, 4.85, 7.20, 11.00, 13.75, 21.00),
    (0.40, 3.20, 6.90, 12.15),
    (0.40, 2.00, 6.30, 11.85, 16.90),
    (0.40, 4.65),
)
