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
