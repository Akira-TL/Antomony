import pytest
import torch

from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.reward import actor_critic_losses, direction_distribution, discounted_returns


def test_discounted_rewards_include_bootstrap_and_terminal_zero():
    torch.testing.assert_close(discounted_returns([1., 2.], 4., .5), torch.tensor([3., 4.]))
    torch.testing.assert_close(discounted_returns([1., 2.], 0., .5), torch.tensor([2., 2.]))
    with pytest.raises(ValueError):
        discounted_returns([float("nan")], 0.)


def test_direction_sampling_covers_all_angles_and_prefers_learned_direction():
    probabilities = direction_distribution(torch.tensor([1., 0.])).probs
    assert probabilities.argmax().item() == 0
    assert bool((probabilities >= .2 / 16. - 1e-7).all())
    assert probabilities.sum().item() == pytest.approx(1.)


def test_actor_cannot_reduce_loss_by_changing_value_baseline():
    log_probability = torch.tensor([-1., -2.], requires_grad=True)
    values = torch.tensor([.5, .5], requires_grad=True)
    actor, critic = actor_critic_losses(log_probability, values, torch.tensor([1., 2.]))
    actor.backward()
    assert values.grad is None
    torch.testing.assert_close(log_probability.grad, torch.tensor([-.25, -.75]))
    critic.backward()
    assert values.grad is not None


def test_reward_phase_opens_memory_but_keeps_reserved_channels_frozen():
    model = ForagingPolicy(71)
    model.set_phase("reward")
    assert model.recent_memory.weight.requires_grad
    assert model.sparse_memory.weight.requires_grad
    assert model.value.weight.requires_grad
    assert all(not p.requires_grad and not p.any() for p in model.reserved_parameters())
