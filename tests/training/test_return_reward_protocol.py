from pathlib import Path

from mathhackson.training.comparison.continuous import ContinuousPlan


def test_reward_ablation_changes_only_empty_return_reward():
    root = Path(__file__).resolve().parents[2]
    on = ContinuousPlan.model_validate_json((root / '.research/protocols/return-reward-on.json').read_bytes())
    off = ContinuousPlan.model_validate_json((root / '.research/protocols/return-reward-off.json').read_bytes())
    assert on.environment.return_reward == 2.
    assert off.environment.return_reward == 0.
    assert on.model_copy(update={'environment': off.environment}) == off
    assert on.environment.model_copy(update={'return_reward': 0.}) == off.environment
    assert on.seeds == (19301, 19302)
    assert on.respawn and on.environment.ants == 32 and on.environment.horizon == 4096
    assert on.checkpoint_every == 256
    assert len(on.conditions) == 1 and on.conditions[0].name == 'moving-danger'
