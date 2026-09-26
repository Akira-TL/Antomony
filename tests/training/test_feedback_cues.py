import dataclasses

import pytest
import torch

from mathhackson.training.foraging.feedback_cues import CueConfig, cue_episode, plastic_features


def test_inputs_do_not_reveal_association_or_future_feedback():
    config = CueConfig(steps=16, batch=4)
    reference = cue_episode(731, config, condition="reference")
    association = cue_episode(731, config, condition="association")
    transient = cue_episode(731, config, condition="transient")
    assert torch.equal(reference.observations, association.observations)
    assert torch.equal(reference.observations, transient.observations)
    assert not torch.equal(reference.targets, association.targets)
    assert torch.equal(association.clean_targets, transient.clean_targets)
    assert [f.name for f in dataclasses.fields(reference.input_at(0))] == ["observation", "features"]
    assert torch.equal(association.input_at(0).features, transient.input_at(0).features)
    assert torch.all(transient.reward_flips.sum(dim=0) == 2)


def test_reserved_receptors_and_last_two_shared_slots_remain_zero():
    episode = cue_episode(733, CueConfig(steps=8, batch=2))
    receptors = episode.observations[..., :72].reshape(8, 2, 9, 8)
    assert not receptors[..., 3:].any()
    observed = episode.input_at(0)
    assert not observed.features.reshape(2, 9, 5)[..., 3:].any()
    assert observed.features.abs().max() <= 1.
    assert not plastic_features(observed.observation, channels="novel").any()
    observed.observation.zero_()
    assert episode.observations[0].any()


def test_feedback_follows_chosen_direction_not_a_training_label_input():
    episode = cue_episode(737, CueConfig(steps=8, batch=2))
    target = episode.targets[0]
    assert torch.allclose(episode.feedback(0, target), torch.ones(2), atol=1e-6)
    assert torch.allclose(episode.feedback(0, -target), -torch.ones(2), atol=1e-6)
    with pytest.raises(ValueError, match="单位方向"):
        episode.feedback(0, torch.zeros(2, 2))
    with pytest.raises(ValueError, match="越界"):
        episode.input_at(8)


def test_seed_and_channel_selection_are_explicit_and_reproducible():
    config = CueConfig(steps=8, batch=3)
    first, same, other = cue_episode(743, config), cue_episode(743, config), cue_episode(744, config)
    assert torch.equal(first.observations, same.observations)
    assert torch.equal(first.targets, same.targets)
    assert not torch.equal(first.observations, other.observations)
    with pytest.raises(ValueError, match="完整"):
        CueConfig(steps=9)
    with pytest.raises(ValueError, match="通道"):
        plastic_features(first.observations[0], channels="unknown")
    broken = first.observations[0].clone()
    broken[0, 0] = float("nan")
    with pytest.raises(ValueError, match="有限"):
        plastic_features(broken, channels="basic")
