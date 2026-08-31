"""Tests for the model and its losses."""

from __future__ import annotations

import pytest
import torch

from model.losses import (
    compute_pos_weight,
    gaussian_nll,
    multitask_loss,
    smoothness,
    stationary_bce,
    yaw_rate_mae,
)
from model.resnet1d import (
    IN_CHANNELS,
    LOGVAR_MAX,
    LOGVAR_MIN,
    WINDOW_SAMPLES,
    ResNet1D,
    count_parameters,
)


# -- architecture ----------------------------------------------------------

def test_output_shapes():
    from model.resnet1d import test_shapes
    out = test_shapes(batch=4, verbose=False)
    for t in out.values():
        assert t.shape == (4,)


def test_rejects_wrong_channel_count():
    m = ResNet1D()
    with pytest.raises(ValueError, match="expected"):
        m(torch.randn(2, 3, WINDOW_SAMPLES))


def test_mu_is_non_negative_even_for_extreme_input():
    m = ResNet1D().eval()
    with torch.no_grad():
        out = m(torch.randn(8, IN_CHANNELS, WINDOW_SAMPLES) * 100)
    assert (out["mu"] >= 0).all()


def test_logvar_is_clamped():
    m = ResNet1D().eval()
    with torch.no_grad():
        out = m(torch.randn(8, IN_CHANNELS, WINDOW_SAMPLES) * 50)
    assert (out["logvar"] >= LOGVAR_MIN).all()
    assert (out["logvar"] <= LOGVAR_MAX).all()


def test_parameter_count_is_reasonable():
    total, trainable = count_parameters(ResNet1D())
    assert total == trainable
    assert 1e6 < total < 1e7, f"{total} parameters"


def test_variable_window_length_still_pools():
    """Global average pooling should make the head length-agnostic."""
    m = ResNet1D().eval()
    with torch.no_grad():
        out = m(torch.randn(2, IN_CHANNELS, 200))
    assert out["mu"].shape == (2,)


# -- losses ----------------------------------------------------------------

def test_gaussian_nll_prefers_the_correct_mean():
    target = torch.full((32,), 10.0)
    lv = torch.zeros(32)
    good = gaussian_nll(torch.full((32,), 10.0), lv, target)
    bad = gaussian_nll(torch.full((32,), 25.0), lv, target)
    assert good < bad


def test_gaussian_nll_rewards_honest_uncertainty():
    """A wrong prediction should be cheaper when the model admits doubt."""
    target = torch.full((32,), 10.0)
    mu = torch.full((32,), 20.0)
    confident = gaussian_nll(mu, torch.full((32,), -2.0), target)
    humble = gaussian_nll(mu, torch.full((32,), 2.0), target)
    assert humble < confident


def test_pos_weight_counters_class_imbalance():
    """With 5% positives the weight should be ~19."""
    y = torch.zeros(1000)
    y[:50] = 1.0
    assert compute_pos_weight(y) == pytest.approx(19.0)


def test_pos_weight_survives_a_batch_with_no_positives():
    assert compute_pos_weight(torch.zeros(64)) == 1.0


def test_pos_weight_makes_all_moving_prediction_expensive():
    """The failure mode it exists to prevent: collapse to 'always moving'."""
    y = torch.zeros(1000)
    y[:50] = 1.0
    logit = torch.full((1000,), -5.0)          # confidently "moving" everywhere
    unweighted = stationary_bce(logit, y)
    weighted = stationary_bce(logit, y, pos_weight=compute_pos_weight(y))
    assert weighted > unweighted * 5


def test_yaw_mae_ignores_invalid_windows():
    pred = torch.zeros(4)
    target = torch.tensor([1.0, float("nan"), 1.0, float("nan")])
    assert yaw_rate_mae(pred, target) == pytest.approx(1.0)


def test_yaw_mae_returns_zero_when_nothing_is_valid():
    out = yaw_rate_mae(torch.zeros(4, requires_grad=True),
                       torch.full((4,), float("nan")))
    assert out.item() == 0.0
    out.backward()          # must not break the graph


def test_smoothness_ignores_session_boundaries():
    """A jump across a session change must not be penalised."""
    mu = torch.tensor([1.0, 1.0, 50.0, 50.0])
    sid = torch.tensor([0, 0, 1, 1])
    t0 = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert smoothness(mu, sid, t0) == pytest.approx(0.0)


def test_smoothness_penalises_within_session_jumps():
    mu = torch.tensor([1.0, 5.0])
    sid = torch.tensor([0, 0])
    t0 = torch.tensor([0.0, 1.0])
    assert smoothness(mu, sid, t0) == pytest.approx(4.0)


def test_smoothness_ignores_time_gaps():
    """Windows either side of a gap are not consecutive in time."""
    mu = torch.tensor([1.0, 40.0])
    sid = torch.tensor([0, 0])
    t0 = torch.tensor([0.0, 90.0])
    assert smoothness(mu, sid, t0) == pytest.approx(0.0)


def test_multitask_loss_reports_every_component():
    from model.losses import test_shapes
    losses = test_shapes(batch=8, verbose=False)
    assert set(losses) == {"total", "displacement", "stationary", "yaw",
                           "smoothness"}
    for v in losses.values():
        assert torch.isfinite(v)


def test_loss_weights_are_applied():
    out = {"mu": torch.ones(4), "logvar": torch.zeros(4),
           "stationary_logit": torch.zeros(4), "yaw_rate": torch.zeros(4)}
    tgt = {"displacement": torch.ones(4), "is_stationary": torch.zeros(4),
           "yaw_rate": torch.full((4,), float("nan")),
           "session_id": torch.zeros(4, dtype=torch.long),
           "t0": torch.arange(4, dtype=torch.float32)}
    base = multitask_loss(out, tgt)
    zeroed = multitask_loss(out, tgt, weights={"stationary": 0.0})
    assert zeroed["total"] < base["total"]
