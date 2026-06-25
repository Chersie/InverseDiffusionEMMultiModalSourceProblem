"""Tests for the sanity checks (per practice.pdf p. 15)."""

from __future__ import annotations

import torch

from mpinv.training.sanity import (
    sanity_check_loss_participation,
    sanity_check_optimiser_coverage,
)


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 2))


def test_optimiser_coverage_pass():
    m = _model()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    sanity_check_optimiser_coverage(m, opt)


def test_optimiser_coverage_missing_param():
    m = _model()
    opt = torch.optim.SGD([p for p in list(m.parameters())[:1]], lr=1e-3)
    try:
        sanity_check_optimiser_coverage(m, opt)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_loss_participation_pass():
    m = _model()
    fn = torch.nn.MSELoss()
    sanity_check_loss_participation(m, fn, torch.randn(2, 4), torch.randn(2, 2))


def test_loss_participation_detect_detached():
    m = _model()
    detached_param = torch.nn.Parameter(torch.randn(3))
    m.extra = detached_param  # added but unused
    fn = torch.nn.MSELoss()
    try:
        sanity_check_loss_participation(m, fn, torch.randn(2, 4), torch.randn(2, 2))
    except ValueError:
        return
    raise AssertionError("expected ValueError for detached parameter")
