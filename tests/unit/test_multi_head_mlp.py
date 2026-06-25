"""Tests for :mod:`mpinv.models.multi_head_mlp`.

Covers the architectural contracts called out in
[.cursor/plans/multi-head_per-l_mode_training_57916da2.plan.md](../../.cursor/plans/multi-head_per-l_mode_training_57916da2.plan.md):

- Round-trip: a single-group multi-head model collapses to the legacy MLP at
  matched seeds (proves the index-map and head bookkeeping is bit-exact).
- Partition validation: overlapping or gap-bearing groups raise.
- Zero + freeze contract: zeroed-and-frozen heads emit identically zero into
  their canonical packed slots.
- Index-map canonical packing: per-l outputs land at the slots the
  ``[Re a^E | Im a^E | Re a^M | Im a^M]`` layout dictates.
- Transplant: backbone + per-l head weights are copied, missing heads stay at
  whatever state they were initialised in, frozen flag is set.
"""

from __future__ import annotations

import pytest
import torch

from mpinv.core.packing import flat_index
from mpinv.models.mlp import MLP, MLPConfig
from mpinv.models.multi_head_mlp import (
    MultiHeadMLP,
    MultiHeadMLPConfig,
    build_index_map,
    expected_output_dim,
    head_output_dim,
    transplant_heads,
    validate_groups,
)
from mpinv.models.registry import MODELS


# ---------------------------------------------------------------------------
# Group validation
# ---------------------------------------------------------------------------


def test_multi_head_mlp_registered():
    assert "multi_head_mlp" in MODELS


def test_validate_groups_default_partition():
    out = validate_groups([[1], [2], [3], [4], [5]], l_max=5)
    assert out == [[1], [2], [3], [4], [5]]


def test_validate_groups_canonical_order():
    # Non-canonical input → canonicalised order.
    out = validate_groups([[3, 1, 2], [5, 4]], l_max=5)
    assert out == [[1, 2, 3], [4, 5]]


def test_validate_groups_rejects_overlap():
    with pytest.raises(ValueError, match="more than one group"):
        validate_groups([[1, 2], [2, 3]], l_max=3)


def test_validate_groups_rejects_gap():
    with pytest.raises(ValueError, match="missing l"):
        validate_groups([[1], [3]], l_max=3)


def test_validate_groups_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        validate_groups([[1], [2], [4]], l_max=3)


def test_validate_groups_rejects_empty_group():
    with pytest.raises(ValueError, match="empty group"):
        validate_groups([[1, 2, 3], []], l_max=3)


# ---------------------------------------------------------------------------
# Output dimensions and index-map canonicality
# ---------------------------------------------------------------------------


def test_head_output_dim_per_l():
    # Per-l block width is 4·(2l+1): for L=5 → 12, 20, 28, 36, 44, sum=140.
    expected_widths = [12, 20, 28, 36, 44]
    assert sum(expected_widths) == expected_output_dim(5)
    for l, w in zip(range(1, 6), expected_widths, strict=True):
        assert head_output_dim([l]) == w


def test_index_map_l1_canonical_positions():
    """For l=1, the canonical packed positions are
    ``q*K + (l-1)(l+1) + (m+l)`` = ``q*K + 0 + (m+1)`` for m∈{-1,0,1}, q∈{0..3}.
    With K = l_max·(l_max+2) = 35 (L=5), the index map for group=[1] is
    [0,1,2, 35,36,37, 70,71,72, 105,106,107].
    """
    idx = build_index_map([1], l_max=5).tolist()
    assert idx == [0, 1, 2, 35, 36, 37, 70, 71, 72, 105, 106, 107]


def test_index_map_disjoint_across_groups():
    L = 5
    K = L * (L + 2)
    union: list[int] = []
    for l in range(1, L + 1):
        union.extend(build_index_map([l], l_max=L).tolist())
    union.sort()
    # Default groups partition exactly the full 4K-long packed vector.
    assert union == list(range(4 * K))


def test_index_map_uses_packing_flat_index_offsets():
    """Verify the (l, m) -> per-quarter offset matches ``packing.flat_index``."""
    L = 5
    K = L * (L + 2)
    for l in range(1, L + 1):
        idx = build_index_map([l], l_max=L).tolist()
        per_quarter = len(idx) // 4
        for q in range(4):
            for j, m in enumerate(range(-l, l + 1)):
                expected = q * K + flat_index(l, m, l_max=L)
                assert idx[q * per_quarter + j] == expected, (
                    f"mismatch q={q} l={l} m={m}: {idx[q*per_quarter+j]} != {expected}"
                )


# ---------------------------------------------------------------------------
# Forward and round-trip
# ---------------------------------------------------------------------------


def _make_mh(L: int, *, groups=None, hidden=8, layers=2, activation="silu"):
    return MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4,
        output_dim=expected_output_dim(L),
        l_max=L,
        groups=groups,
        hidden_size=hidden,
        n_hidden_layers=layers,
        activation=activation,
    ))


def test_forward_shape_and_default_groups():
    L = 5
    m = _make_mh(L)
    assert m.n_heads == 5
    assert m.head_output_widths() == [12, 20, 28, 36, 44]
    y = m(torch.randn(7, 4))
    assert y.shape == (7, expected_output_dim(L))


def test_round_trip_full_band_matches_mlp():
    """A single-group MultiHeadMLP that covers all l-bands produces bit-identical
    output to the legacy :class:`MLP` for matched seeds and inputs.

    This proves the scatter index_map for a "full" group is the identity, and
    that the backbone factoring in :func:`mpinv.models.mlp.make_backbone` is
    exact (no drift in widths, activations, dropout placement, etc.).
    """
    L = 4
    out_dim = expected_output_dim(L)
    torch.manual_seed(123)
    mlp = MLP(MLPConfig(
        input_dim=8, output_dim=out_dim,
        hidden_size=16, n_hidden_layers=3,
        architecture="flat", activation="silu",
    ))
    torch.manual_seed(123)
    mh = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=8, output_dim=out_dim, l_max=L,
        groups=[[1, 2, 3, 4]],
        hidden_size=16, n_hidden_layers=3,
        architecture="flat", activation="silu",
    ))
    x = torch.randn(5, 8)
    assert torch.equal(mlp(x), mh(x))


def test_round_trip_full_band_residual_architecture():
    """Same identity check for the residual architecture, where backbone widths
    are uniform and the per-block skip changes the parameter graph shape."""
    L = 3
    out_dim = expected_output_dim(L)
    torch.manual_seed(7)
    mlp = MLP(MLPConfig(
        input_dim=12, output_dim=out_dim,
        hidden_size=12, n_hidden_layers=2,
        architecture="residual", activation="gelu",
    ))
    torch.manual_seed(7)
    mh = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=12, output_dim=out_dim, l_max=L,
        groups=[[1, 2, 3]],
        hidden_size=12, n_hidden_layers=2,
        architecture="residual", activation="gelu",
    ))
    x = torch.randn(2, 12)
    assert torch.allclose(mlp(x), mh(x))


# ---------------------------------------------------------------------------
# Lifecycle: zero / freeze / re-init
# ---------------------------------------------------------------------------


def test_zero_init_head_emits_zero_in_canonical_slots():
    L = 5
    m = _make_mh(L)
    target = 2  # head index 2 = l=3 → 7 m-values per quarter, offset (l-1)(l+1)=8.
    m.zero_init_head(target)
    m.set_head_trainable(target, False)
    x = torch.randn(3, 4)
    y = m(x)
    K = L * (L + 2)
    for q in range(4):
        slot = y[:, q * K + 8 : q * K + 8 + 7]
        assert torch.all(slot == 0.0), (
            f"quarter {q} l=3 slot should be zero after zero_init_head + freeze"
        )


def test_zero_then_reinit_makes_weights_nonzero():
    m = _make_mh(3)
    head_idx = 1
    m.zero_init_head(head_idx)
    head = m.heads[head_idx]
    assert torch.all(head.weight == 0.0)
    if head.bias is not None:
        assert torch.all(head.bias == 0.0)
    m.reinit_head(head_idx)
    # Default Linear init draws from kaiming_uniform_ with non-zero range; the
    # probability of an exactly-zero realisation is zero.
    assert torch.any(head.weight != 0.0)


def test_set_head_trainable_flips_requires_grad():
    m = _make_mh(4)
    m.set_head_trainable(2, False)
    assert all(not p.requires_grad for p in m.heads[2].parameters())
    m.set_head_trainable(2, True)
    assert all(p.requires_grad for p in m.heads[2].parameters())


def test_set_backbone_trainable_flips_requires_grad():
    m = _make_mh(3)
    m.set_backbone_trainable(False)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    m.set_backbone_trainable(True)
    assert all(p.requires_grad for p in m.backbone.parameters())


def test_trainable_summary_reports_block_state():
    m = _make_mh(3)
    m.set_backbone_trainable(False)
    m.set_head_trainable(0, False)
    m.set_head_trainable(2, False)
    s = m.trainable_summary()
    assert s == {
        "backbone": False,
        "head_0": False,
        "head_1": True,
        "head_2": False,
    }


# ---------------------------------------------------------------------------
# Output-dim mismatch
# ---------------------------------------------------------------------------


def test_config_rejects_mismatched_output_dim():
    with pytest.raises(ValueError, match="output_dim must equal"):
        MultiHeadMLPConfig(
            input_dim=8, output_dim=999,  # not 4·5·7=140
            l_max=5,
        )


# ---------------------------------------------------------------------------
# Transplant
# ---------------------------------------------------------------------------


def test_transplant_l3_to_l5_copies_matching_heads_and_freezes():
    L_src, L_dst = 3, 5
    src = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(L_src),
        l_max=L_src, hidden_size=8, n_hidden_layers=2, activation="silu",
    ))
    dst = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(L_dst),
        l_max=L_dst, hidden_size=8, n_hidden_layers=2, activation="silu",
    ))
    src_head0_w = src.heads[0].weight.detach().clone()
    info = transplant_heads(src, dst, freeze_src_heads=True)
    assert info["transplanted_head_indices"] == [0, 1, 2]
    assert info["skipped_head_indices"] == [3, 4]
    assert info["backbone_copied"] is True
    assert torch.equal(dst.heads[0].weight, src_head0_w)
    # Source heads frozen in destination.
    for di in [0, 1, 2]:
        assert all(not p.requires_grad for p in dst.heads[di].parameters())
    # Skipped heads still trainable by default.
    for di in [3, 4]:
        assert all(p.requires_grad for p in dst.heads[di].parameters())


def test_transplant_rejects_dst_smaller_than_src():
    src = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(5),
        l_max=5, hidden_size=8, n_hidden_layers=2,
    ))
    dst = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(3),
        l_max=3, hidden_size=8, n_hidden_layers=2,
    ))
    with pytest.raises(ValueError, match="must be >= source l_max"):
        transplant_heads(src, dst)


def test_transplant_rejects_incompatible_backbone():
    src = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(3),
        l_max=3, hidden_size=8, n_hidden_layers=2,
    ))
    dst = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(5),
        l_max=5, hidden_size=16, n_hidden_layers=2,  # different hidden_size
    ))
    with pytest.raises(ValueError, match="backbone-incompatible"):
        transplant_heads(src, dst, copy_backbone=True)
