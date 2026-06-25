"""Multi-head MLP: per-`l`-band heads on a shared backbone.

Motivation
----------
The packed coefficient layout fixed in :mod:`mpinv.core.packing` is

    [Re a^E | Im a^E | Re a^M | Im a^M]   length 4 K = 4 L (L + 2)

with inner ordering ``l = 1..L``, ``m = -l..+l``. For each ``l`` the per-band
block has size ``2 l + 1`` *per quarter*, so the four contiguous-by-stride
slices of width ``2 l + 1`` (one per quarter) describe the contribution of
multipole order ``l`` to the full coefficient vector. A *per-l head* is a
``nn.Linear`` whose output writes those four slices and nothing else, leaving
the slices for other ``l`` values to other heads.

Concretely, for ``L = 5`` the per-l output widths are
``4 · (2 l + 1) = 12, 20, 28, 36, 44`` and they sum to ``4 K = 140`` exactly.

The architecture realised here:

- A **shared backbone** (the existing :func:`mpinv.models.mlp.make_backbone`
  pipeline; same four architectures available: flat / pyramid / bottleneck /
  residual) ending in a hidden vector of width ``hidden_size``.
- A list of **per-group heads**, where a *group* is a (sorted, disjoint, no-gap)
  subset of ``{1, …, L}``. The default is one head per ``l``
  (``[[1], [2], …, [L]]``), but configurable groupings such as
  ``[[1, 2, 3], [4, 5]]`` or ``[[1], [2], [3], [4], [5], [6, 7, 8, 9, 10],
  [11, 12, 13, 14, 15]]`` are supported as long as the union is exactly
  ``{1, …, L}``.
- A precomputed ``index_map`` per head that scatters the head's local output
  slots into the canonical packed-vector positions via ``index_copy_``.

The forward pass therefore produces a ``(B, 4 K)`` tensor in *exactly* the
canonical layout, so the existing physics decoder, losses, and metrics work
unchanged.

Stage-wise training contract
----------------------------
Heads expose three lifecycle methods used by
:mod:`mpinv.training.staged`:

- :meth:`MultiHeadMLP.zero_init_head` — set head weights and bias to zero so the
  head emits identically zero. Used for *future* heads in stage ``k``.
- :meth:`MultiHeadMLP.reinit_head` — call ``head.reset_parameters()`` to restore
  default-distributed weights (PyTorch's ``kaiming_uniform_`` for the weight,
  ``uniform_(-1/√fan_in, +1/√fan_in)`` for the bias — see R4 in the framework
  manifest). Used when activating a previously zero-frozen head.
- :meth:`MultiHeadMLP.set_head_trainable` /
  :meth:`MultiHeadMLP.set_backbone_trainable` — flip ``requires_grad`` on the
  parameters of the chosen head / backbone. The trainer's
  ``sanity_check_optimiser_coverage`` then asserts the optimiser sees every
  trainable parameter exactly once.

Reuse pretrained smaller-L models
---------------------------------
:func:`transplant_heads` copies the backbone state (when structurally
compatible) and the per-l head weights from a smaller-L source model to a
larger-L destination model. Heads in the destination that have no source
counterpart stay zero-initialised. This is the canonical "modes 6-15 starting
from a frozen 1-5" path the user asked for.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from mpinv.core.packing import L_MAX
from mpinv.models.base import BaseModelConfig
from mpinv.models.mlp import MLPConfig, make_backbone
from mpinv.models.registry import register_model

Architecture = Literal["flat", "pyramid", "bottleneck", "residual"]


def _default_groups(l_max: int) -> list[list[int]]:
    """Default partition: one head per `l` in ``1..l_max``."""
    return [[l] for l in range(1, l_max + 1)]


def validate_groups(groups: Sequence[Sequence[int]], l_max: int) -> list[list[int]]:
    """Assert that ``groups`` is a partition of ``{1, …, l_max}`` and return a
    canonicalised copy with each group sorted ascending and groups themselves
    ordered by their minimum element.

    Raises ``ValueError`` on overlap, gaps, out-of-range values, or duplicates.
    """
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1; got {l_max}")
    seen: set[int] = set()
    canonical: list[list[int]] = []
    for raw in groups:
        if not raw:
            raise ValueError("empty group not allowed")
        g = sorted(int(l) for l in raw)
        for l in g:
            if not (1 <= l <= l_max):
                raise ValueError(f"l={l} out of range [1, {l_max}] in group {raw!r}")
            if l in seen:
                raise ValueError(f"l={l} appears in more than one group")
            seen.add(l)
        canonical.append(g)
    expected = set(range(1, l_max + 1))
    missing = expected - seen
    if missing:
        raise ValueError(
            f"groups must partition 1..{l_max}; missing l(s): {sorted(missing)}"
        )
    canonical.sort(key=lambda g: g[0])
    return canonical


def build_index_map(group: Sequence[int], l_max: int) -> torch.LongTensor:
    """Scatter index map for one ``l``-band group.

    The packed vector has 4 quarters of width ``K = l_max (l_max + 2)``: index
    ``q * K + (l - 1)(l + 1) + (m + l)`` is the slot for quarter ``q`` (0 = Re
    a^E, 1 = Im a^E, 2 = Re a^M, 3 = Im a^M), order ``l``, sub-index ``m``.

    The head's *local* output is laid out in the canonical
    "quarter → l (ascending in group) → m (ascending from -l to +l)" order.
    Returns the LongTensor of length ``4 · sum(2 l + 1 for l in group)`` mapping
    each local slot to its global packed position.
    """
    K = l_max * (l_max + 2)
    g = sorted(set(int(l) for l in group))
    indices: list[int] = []
    for q in range(4):
        for l in g:
            offset_l = (l - 1) * (l + 1)
            base = q * K + offset_l
            for m in range(-l, l + 1):
                indices.append(base + (m + l))
    return torch.tensor(indices, dtype=torch.long)


def head_output_dim(group: Sequence[int]) -> int:
    """Width of one head's output: ``4 · sum(2 l + 1 for l in group)``."""
    return int(4 * sum(2 * int(l) + 1 for l in group))


@dataclass(slots=True)
class MultiHeadMLPConfig(BaseModelConfig):
    """Configuration for :class:`MultiHeadMLP`.

    Attributes
    ----------
    input_dim, output_dim : int (inherited)
        ``output_dim`` must equal ``4 · l_max · (l_max + 2)`` (validated in
        ``__post_init__``).
    l_max : int
        Top truncation order. Drives the canonical packed dimension.
    groups : list[list[int]] | None
        L-band partition of ``{1, …, l_max}``. ``None`` defaults to
        ``[[1], [2], …, [l_max]]`` (one head per l).
    hidden_size, n_hidden_layers, architecture, hidden_size_min, dropout,
    use_layer_norm, use_bias, activation
        Mirror :class:`MLPConfig`.
    """

    l_max: int = L_MAX
    groups: list[list[int]] | None = None
    hidden_size: int = 512
    n_hidden_layers: int = 4
    architecture: Architecture = "flat"
    hidden_size_min: int = 64
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_bias: bool = True
    activation: Literal["silu", "relu", "gelu", "elu"] = "silu"

    def __post_init__(self) -> None:
        if self.l_max < 1:
            raise ValueError(f"l_max must be >= 1; got {self.l_max}")
        if self.groups is None:
            self.groups = _default_groups(self.l_max)
        else:
            self.groups = validate_groups(self.groups, self.l_max)
        K = self.l_max * (self.l_max + 2)
        expected_output = 4 * K
        if self.output_dim != expected_output:
            raise ValueError(
                f"output_dim must equal 4·l_max·(l_max+2) = {expected_output} "
                f"for l_max={self.l_max}; got {self.output_dim}"
            )

    def to_mlp_config(self) -> MLPConfig:
        """Project the body knobs onto an :class:`MLPConfig` for backbone reuse."""
        return MLPConfig(
            input_dim=self.input_dim,
            output_dim=expected_output_dim(self.l_max),
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
            architecture=self.architecture,
            hidden_size_min=self.hidden_size_min,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            use_bias=self.use_bias,
            activation=self.activation,
        )


def expected_output_dim(l_max: int) -> int:
    """Canonical packed dimension at truncation ``l_max``: ``4 · l_max · (l_max + 2)``."""
    return 4 * l_max * (l_max + 2)


@register_model("multi_head_mlp")
class MultiHeadMLP(nn.Module):
    """Shared MLP backbone + per-`l`-band linear heads with canonical packed scatter.

    The forward pass builds a ``(B, 4K)`` output by scattering each head's local
    output into its canonical positions:

    .. code-block:: python

        h = self.backbone(x)                      # (B, hidden)
        out = x.new_zeros(x.size(0), 4 * K)        # (B, 4 K)
        for head, idx_map in zip(self.heads, self.index_maps):
            out.index_copy_(1, idx_map, head(h))   # writes one per-l block
        return out

    Heads are zeroable / re-initialisable / freezable independently — see the
    module docstring.
    """

    backbone: nn.Sequential
    heads: nn.ModuleList

    def __init__(self, cfg: MultiHeadMLPConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.groups is not None, "MultiHeadMLPConfig.__post_init__ should set groups"

        self.backbone, hidden_dim = make_backbone(cfg.to_mlp_config())
        self._hidden_dim = hidden_dim

        self.heads = nn.ModuleList()
        # Stable mapping: position i in self.heads <-> cfg.groups[i] (canonical order)
        for group in cfg.groups:
            head = nn.Linear(hidden_dim, head_output_dim(group), bias=cfg.use_bias)
            self.heads.append(head)

        # Buffer: index_map_<i> is a LongTensor of length head_output_dim(groups[i])
        # giving the global packed position for each local head output slot.
        for i, group in enumerate(cfg.groups):
            self.register_buffer(
                f"index_map_{i}",
                build_index_map(group, cfg.l_max),
                persistent=False,
            )

        self._output_dim = expected_output_dim(cfg.l_max)

    @property
    def input_dim(self) -> int:
        return self.cfg.input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def n_heads(self) -> int:
        return len(self.heads)

    @property
    def groups(self) -> list[list[int]]:
        assert self.cfg.groups is not None
        return self.cfg.groups

    def index_map(self, head_idx: int) -> torch.Tensor:
        """Return the scatter index buffer for head ``head_idx``."""
        return self.get_buffer(f"index_map_{head_idx}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = h.new_zeros(h.size(0), self._output_dim)
        for i, head in enumerate(self.heads):
            idx = self.get_buffer(f"index_map_{i}")
            out.index_copy_(1, idx, head(h))
        return out

    # ----- per-head lifecycle -------------------------------------------------

    @torch.no_grad()
    def zero_init_head(self, head_idx: int) -> None:
        """Set head ``head_idx``'s weight (and bias if present) to zero.

        After this call the head emits identically zero for any input. Used for
        "future" heads while stage ``k`` is training the lower bands.
        """
        head: nn.Linear = self.heads[head_idx]  # type: ignore[assignment]
        head.weight.zero_()
        if head.bias is not None:
            head.bias.zero_()

    @torch.no_grad()
    def reinit_head(self, head_idx: int) -> None:
        """Re-initialise head ``head_idx`` to PyTorch's default ``Linear`` init.

        This is the canonical path verified in R4 of the framework manifest:
        ``kaiming_uniform_(weight, a=√5)`` plus uniform bias in
        ``(-1/√fan_in, +1/√fan_in)``. Use when activating a head that was
        previously zeroed (the "non-zero weights for faster convergence" path).
        """
        head: nn.Linear = self.heads[head_idx]  # type: ignore[assignment]
        head.reset_parameters()

    def set_head_trainable(self, head_idx: int, trainable: bool) -> None:
        """Flip ``requires_grad`` on every parameter of head ``head_idx``."""
        for p in self.heads[head_idx].parameters():
            p.requires_grad_(trainable)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Flip ``requires_grad`` on every backbone parameter."""
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)

    def head_parameters(self, head_idx: int) -> Iterable[nn.Parameter]:
        """Yield the parameters of head ``head_idx`` (both weight and bias)."""
        return self.heads[head_idx].parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        """Yield every parameter of the shared backbone."""
        return self.backbone.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- reporting ----------------------------------------------------------

    def head_output_widths(self) -> list[int]:
        """Per-head output width, in canonical group order."""
        return [head_output_dim(g) for g in self.groups]

    def trainable_summary(self) -> dict[str, bool]:
        """Map ``"backbone" | "head_<i>"`` to whether *any* parameter of that
        block is trainable. Useful for logging at stage transitions.
        """
        out: dict[str, bool] = {
            "backbone": any(p.requires_grad for p in self.backbone.parameters())
        }
        for i, head in enumerate(self.heads):
            out[f"head_{i}"] = any(p.requires_grad for p in head.parameters())
        return out


# -----------------------------------------------------------------------------
# Reuse smaller-L pretrained models
# -----------------------------------------------------------------------------


def transplant_heads(
    src: MultiHeadMLP,
    dst: MultiHeadMLP,
    *,
    freeze_src_heads: bool = True,
    copy_backbone: bool = True,
) -> dict[str, list[int] | bool]:
    """Copy backbone state and matching per-l head weights from ``src`` into ``dst``.

    A destination head is matched to a source head iff the two heads cover the
    *exact same* multi-set of l-values (same group, in the same canonical
    order). This is intentionally strict: a destination group ``[1, 2]`` does
    not match source groups ``[[1], [2]]`` because the head linear layers have
    different output widths and therefore incompatible parameter shapes.

    Parameters
    ----------
    src, dst : MultiHeadMLP
        Source (smaller or equal ``l_max``) and destination models. Must share
        the *backbone configuration* if ``copy_backbone=True`` (same hidden
        sizes and architecture); a ``ValueError`` is raised on mismatch.
    freeze_src_heads : bool
        If ``True`` (default), set ``requires_grad=False`` on every transplanted
        head in ``dst`` so the staged trainer does not move them.
    copy_backbone : bool
        If ``True`` (default), copy the backbone ``state_dict`` from ``src`` to
        ``dst``. Requires structural compatibility — same input dim, same
        hidden widths, same activation, same bias / layer-norm flags.

    Returns
    -------
    dict[str, list[int]]
        ``{"transplanted_head_indices": [...], "skipped_head_indices": [...],
        "backbone_copied": bool}`` for downstream logging.
    """
    if dst.cfg.l_max < src.cfg.l_max:
        raise ValueError(
            f"destination l_max ({dst.cfg.l_max}) must be >= source l_max "
            f"({src.cfg.l_max}); cannot truncate during transplant"
        )

    backbone_copied = False
    if copy_backbone:
        # Structural compatibility: identical body knobs, identical input dim.
        for attr in (
            "input_dim",
            "hidden_size",
            "n_hidden_layers",
            "architecture",
            "hidden_size_min",
            "use_layer_norm",
            "use_bias",
            "activation",
        ):
            sv = getattr(src.cfg, attr)
            dv = getattr(dst.cfg, attr)
            if sv != dv:
                raise ValueError(
                    f"backbone-incompatible config: cfg.{attr} differs "
                    f"(src={sv!r}, dst={dv!r})"
                )
        # Note: dropout differs only in the regularisation strength; it does not
        # change parameter shapes, so we tolerate any mismatch and just copy.
        dst.backbone.load_state_dict(src.backbone.state_dict())
        backbone_copied = True

    src_groups = {tuple(g): i for i, g in enumerate(src.groups)}
    transplanted: list[int] = []
    skipped: list[int] = []
    for di, dgroup in enumerate(dst.groups):
        si = src_groups.get(tuple(dgroup))
        if si is None:
            skipped.append(di)
            continue
        src_head: nn.Linear = src.heads[si]  # type: ignore[assignment]
        dst_head: nn.Linear = dst.heads[di]  # type: ignore[assignment]
        if src_head.weight.shape != dst_head.weight.shape:
            raise ValueError(
                f"head-shape mismatch on group {dgroup!r}: "
                f"src.weight {tuple(src_head.weight.shape)} "
                f"vs dst.weight {tuple(dst_head.weight.shape)}"
            )
        with torch.no_grad():
            dst_head.weight.copy_(src_head.weight)
            if dst_head.bias is not None and src_head.bias is not None:
                dst_head.bias.copy_(src_head.bias)
            elif (dst_head.bias is None) != (src_head.bias is None):
                raise ValueError(
                    f"head-bias presence mismatch on group {dgroup!r}: "
                    f"src.bias is None = {src_head.bias is None}, "
                    f"dst.bias is None = {dst_head.bias is None}"
                )
        if freeze_src_heads:
            dst.set_head_trainable(di, False)
        transplanted.append(di)

    return {
        "transplanted_head_indices": transplanted,
        "skipped_head_indices": skipped,
        "backbone_copied": backbone_copied,
    }
