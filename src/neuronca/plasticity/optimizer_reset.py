"""Reset optimizer state for specific parameter indices on slot/edge (re)activation.

Prevents a reactivated neuron from inheriting SGD momentum / Adam moments from a
previous occupant of the same capacity index (Phase 1a plan §Resolved #1).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer


def _zero_state_tensor_slices(
    state_tensor: Tensor,
    dim: int,
    indices: Sequence[int],
) -> None:
    """In-place zero slices of ``state_tensor`` along ``dim`` at ``indices``."""
    if not indices:
        return
    idx = torch.as_tensor(list(indices), device=state_tensor.device, dtype=torch.long)
    # Advanced indexing along ``dim``.
    if dim == 0:
        state_tensor[idx] = 0
    elif dim == 1:
        state_tensor[:, idx] = 0
    else:
        raise ValueError(f"only dim 0 or 1 supported, got {dim}")


def reset_optimizer_state_for_indices(
    optimizer: Optimizer,
    param: Tensor,
    *,
    row_indices: Sequence[int] | None = None,
    col_indices: Sequence[int] | None = None,
    bias_indices: Sequence[int] | None = None,
    bias_param: Tensor | None = None,
) -> None:
    """Zero optimizer state for selected rows/cols of a weight (and optional bias).

    Args:
        optimizer: Torch optimizer that may hold state for ``param`` / ``bias_param``.
        param: Weight parameter ``[out, in]`` (or 1-D bias if only bias_indices used
            with ``param`` as the bias — prefer ``bias_param`` for clarity).
        row_indices: Output-feature indices to scrub (weight dim 0, full rows).
        col_indices: Input-feature indices to scrub (weight dim 1, full cols).
        bias_indices: Bias entries to scrub (uses ``bias_param`` if given, else
            treats ``param`` as 1-D bias).
        bias_param: Optional bias Parameter linked to the same Linear.

    Notes:
        - If a param has no entry in ``optimizer.state`` yet (never stepped), this
          is a no-op for that param — correct, since there is no stale state.
        - Supported state keys: any tensor value matching param shape (SGD
          ``momentum_buffer``, Adam ``exp_avg`` / ``exp_avg_sq``). Scalar state
          (e.g. Adam ``step``) is left unchanged.
    """
    row_indices = list(row_indices or ())
    col_indices = list(col_indices or ())
    bias_indices = list(bias_indices or ())

    def _scrub(p: Tensor, rows: Sequence[int], cols: Sequence[int]) -> None:
        if p not in optimizer.state:
            return
        st = optimizer.state[p]
        for key, val in st.items():
            if not isinstance(val, Tensor):
                continue
            if val.shape != p.shape:
                continue
            if p.ndim == 2:
                if rows:
                    _zero_state_tensor_slices(val, 0, rows)
                if cols:
                    _zero_state_tensor_slices(val, 1, cols)
            elif p.ndim == 1 and rows:
                # 1-D buffer scrubbed via ``rows`` as flat indices.
                _zero_state_tensor_slices(val, 0, rows)

    if param.ndim == 2:
        _scrub(param, row_indices, col_indices)
    elif param.ndim == 1 and bias_indices and bias_param is None:
        _scrub(param, bias_indices, ())

    if bias_param is not None and bias_indices:
        _scrub(bias_param, bias_indices, ())


def reset_linear_neuron_optimizer_state(
    optimizer: Optimizer,
    weight: Tensor,
    bias: Tensor | None,
    *,
    out_index: int | None = None,
    in_index: int | None = None,
) -> None:
    """Convenience: scrub one output row and/or one input column (+ matching bias)."""
    rows = [int(out_index)] if out_index is not None else []
    cols = [int(in_index)] if in_index is not None else []
    bias_idx = rows if bias is not None and out_index is not None else []
    reset_optimizer_state_for_indices(
        optimizer,
        weight,
        row_indices=rows,
        col_indices=cols,
        bias_indices=bias_idx,
        bias_param=bias,
    )


def reset_linear_edge_optimizer_state(
    optimizer: Optimizer,
    weight: Tensor,
    out_index: int,
    in_index: int,
) -> None:
    """Zero optimizer state for a single weight entry (edge) if present.

    SGD/Adam store full-tensor buffers; we zero the scalar location in each.
    """
    if weight not in optimizer.state:
        return
    st = optimizer.state[weight]
    oi, ii = int(out_index), int(in_index)
    for key, val in st.items():
        if isinstance(val, Tensor) and val.shape == weight.shape:
            val[oi, ii] = 0
