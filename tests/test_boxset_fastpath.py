# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BoxSet rectangular fast-path tests for ktdp.load/store and indirect_*.

Covers issue #52: parity between the BoxSet sub-TileRef shortcut on
direct ktdp.load/store and the per-coord scatter that the AffineSet
slow path uses, plus the vectorised _build_indirect_coords_box variant
on the indirect side.
"""

from unittest.mock import patch

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from ktir_cpu.affine import BoxSet
from ktir_cpu.dialects.ktdp_helpers import eval_subscript_expr
from ktir_cpu.dialects.ktdp_ops import ktdp__load
from ktir_cpu.dtypes import bytes_per_elem
from ktir_cpu.grid import CoreContext
from ktir_cpu.ir_types import AccessTile, IndirectAccessTile, MemRef, Tile
from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.ops.memory_ops import (
    MemoryOps,
    _build_indirect_coords_box,
    _eval_subscript_vectorized,
)
from ktir_cpu.parser_ast import parse_affine_map


def _make_ctx():
    hbm = HBMSimulator()
    lx = LXScratchpad(core_id=0)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm), hbm


# ---------------------------------------------------------------------------
# MemoryOps.boxed_load / boxed_store unit tests
# ---------------------------------------------------------------------------

class TestBoxedLoadStoreUnit:
    """Direct sub-TileRef shortcut tested at the helper level (no parser).

    Row-major no-trample is omitted — column-packed strictly contains it
    (stride-inheritance bugs cannot fail col-packed and pass row-major).
    Full-rectangle correctness lives in
    ``TestKtdpDispatchTakesBoxSetPath`` to avoid duplicating the
    happy-path assertion at two layers.
    """

    def test_sub_rectangle_load_returns_box_extent(self):
        """boxed_load on a translated sub-box returns just the box
        contents (in row-major order), not the full parent.  Sole
        unit-level coverage for the read direction; the store
        direction is covered by the col-packed no-trample test."""
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)
        tile_ref = MemRef(
            base_ptr=ptr, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        box = BoxSet(lo=(1, 1), hi=(3, 3))
        boxed = MemoryOps.boxed_load(ctx, tile_ref, box)
        assert boxed.shape == (2, 2)
        np.testing.assert_array_equal(boxed.data, data[1:3, 1:3])

    def test_store_col_packed_does_not_trample(self):
        """Column-packed strides=[1, R]: stride inheritance must come
        from the parent so the sub-tile lands at the right byte
        positions in column-major memory.  Bug here would scatter the
        write across columns and trample sentinels."""
        ctx, hbm = _make_ctx()
        ROWS, COLS = 4, 4
        SENTINEL = np.float16(-3.0)
        elems = ROWS * COLS
        ptr = hbm.allocate(elems * bytes_per_elem("f16"))
        hbm.write(ptr, np.full(elems, SENTINEL, dtype=np.float16))

        # Column-packed: strides=[1, ROWS] means consecutive elements
        # along axis 0 are 1 element apart in memory, and consecutive
        # elements along axis 1 are ROWS elements apart.
        tile_ref = MemRef(
            base_ptr=ptr, shape=(ROWS, COLS), strides=[1, ROWS],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        box = BoxSet(lo=(1, 1), hi=(3, 3))
        payload = np.arange(1, 5, dtype=np.float16).reshape(2, 2)
        MemoryOps.boxed_store(ctx, Tile(payload, "f16", (2, 2)), tile_ref, box)

        # Reconstruct logical view from column-packed bytes.
        flat = hbm.read(ptr, elems, "f16")
        logical = np.empty((ROWS, COLS), dtype=np.float16)
        for r in range(ROWS):
            for c in range(COLS):
                logical[r, c] = flat[r * 1 + c * ROWS]
        np.testing.assert_array_equal(logical[1:3, 1:3], payload)
        mask = np.zeros((ROWS, COLS), dtype=bool)
        mask[1:3, 1:3] = True
        outside = logical[~mask]
        assert np.all(outside == SENTINEL), (
            "Column-packed boxed_store wrote outside the box — stride "
            "inheritance bug."
        )

    def test_empty_box(self):
        """``hi[d] <= lo[d]`` on any axis short-circuits both directions:
        load returns a zero-element tile (no offset pointer fabricated),
        store returns 0 sticks and leaves the parent allocation
        untouched.  One test, two assertions — same failure mode."""
        ctx, hbm = _make_ctx()
        SENTINEL = np.float16(-1.0)
        elems = 16
        ptr = hbm.allocate(elems * bytes_per_elem("f16"))
        hbm.write(ptr, np.full(elems, SENTINEL, dtype=np.float16))

        tile_ref = MemRef(
            base_ptr=ptr, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        box = BoxSet(lo=(2, 2), hi=(2, 4))   # empty on axis 0
        loaded = MemoryOps.boxed_load(ctx, tile_ref, box)
        assert loaded.data.size == 0
        assert loaded.data.shape == (0, 2)

        empty_tile = Tile(np.zeros((0, 2), dtype=np.float16), "f16", (0, 2))
        sticks = MemoryOps.boxed_store(ctx, empty_tile, tile_ref, box)
        assert sticks == 0
        assert np.all(hbm.read(ptr, elems, "f16") == SENTINEL)


# ---------------------------------------------------------------------------
# ktdp.load / ktdp.store wiring + structural fast-path verification
# ---------------------------------------------------------------------------

_VECTOR_ADD_MLIR = """
module {
  func.func @add() attributes {grid = [1, 1]} {
    %X_addr = arith.constant 0 : index
    %Y_addr = arith.constant 1 : index
    %Z_addr = arith.constant 2 : index

    %X = ktdp.construct_memory_view %X_addr, sizes: [8], strides: [1] {
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %Y = ktdp.construct_memory_view %Y_addr, sizes: [8], strides: [1] {
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %Z = ktdp.construct_memory_view %Z_addr, sizes: [8], strides: [1] {
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>

    %c0 = arith.constant 0 : index
    %x_at = ktdp.construct_access_tile %X[%c0]
        : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %y_at = ktdp.construct_access_tile %Y[%c0]
        : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %z_at = ktdp.construct_access_tile %Z[%c0]
        : memref<8xf16> -> !ktdp.access_tile<8xindex>

    %x = ktdp.load %x_at : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %y = ktdp.load %y_at : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %z = arith.addf %x, %y : tensor<8xf16>
    ktdp.store %z, %z_at : tensor<8xf16>, !ktdp.access_tile<8xindex>
    return
  }
}
"""


class TestKtdpDispatchTakesBoxSetPath:
    """End-to-end tests via ktdp.load/store handlers."""

    def test_full_rectangle_via_ktdp_load(self):
        """End-to-end vector add: BoxSet([0, shape)) (the new sentinel)
        must (a) take the structural sub-TileRef shortcut — verified
        by spying on ``_flat_memory_offsets`` — and (b) compute the
        right values.  Two assertions, one fixture: structural fast
        path + happy-path correctness.  Replaces a separate unit-level
        full-rectangle parity test (the helper-level layer is
        sufficiently exercised by sub-rectangle and col-packed tests).
        """
        interp = KTIRInterpreter()
        interp.load(_VECTOR_ADD_MLIR)

        x_in = np.arange(8, dtype=np.float16)
        y_in = np.arange(8, 16, dtype=np.float16)

        seen = []
        real = MemoryOps._flat_memory_offsets

        def _spy(*args, **kwargs):
            seen.append(args)
            return real(*args, **kwargs)

        with patch.object(MemoryOps, "_flat_memory_offsets",
                          staticmethod(_spy)):
            _orig = interp._prepare_execution
            def _seed(grid_shape):
                _orig(grid_shape)
                hbm = interp.memory.hbm
                hbm.write(0, x_in)
                hbm.write(1, y_in)
                hbm.write(2, np.zeros(8, dtype=np.float16))
            interp._prepare_execution = _seed
            interp.execute_function("add")

        assert seen == [], (
            f"BoxSet fast path was bypassed: _flat_memory_offsets was "
            f"called {len(seen)} time(s) — expected 0 because every "
            f"access tile in this fixture is rectangular."
        )
        z_out = interp.memory.hbm.read(2, 8, "f16")
        np.testing.assert_array_equal(z_out, x_in + y_in)

    def test_parser_normalises_omitted_set_to_boxset(self):
        """When MLIR omits access_tile_set, the parser fills in
        BoxSet([0, shape)) so every rectangular access flows through
        a single ``coordinate_set`` shape into the BoxSet fast path."""
        interp = KTIRInterpreter()
        interp.load(_VECTOR_ADD_MLIR)
        func = interp.module.functions["add"]
        access_tile_ops = [
            op for op in func.operations
            if op.op_type == "ktdp.construct_access_tile"
        ]
        assert access_tile_ops, "no construct_access_tile op in fixture"
        for op in access_tile_ops:
            cs = op.attributes.get("coordinate_set")
            assert isinstance(cs, BoxSet), (
                f"coordinate_set should be BoxSet (sentinel retired), "
                f"got {type(cs).__name__}"
            )
            assert cs.lo == (0,) * len(op.attributes["shape"])
            assert tuple(cs.hi) == tuple(op.attributes["shape"])


class TestSlowPathFallbackOnNonIdentityCoordinateOrder:
    """All other fast-path tests verify that the BoxSet branch is
    taken; this one verifies the negative direction — when the guard
    rejects the access (non-identity ``coordinate_order``), execution
    must fall through to the coord-list slow path that calls
    ``_flat_memory_offsets``.  Without this, the guard could silently
    widen (e.g. ``cso is None or True``) and no test would catch it.
    """

    def test_non_identity_order_calls_flat_memory_offsets(self):
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)
        parent_ref = MemRef(
            base_ptr=ptr, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        # Transposing AffineMap — non-identity, so the guard rejects
        # the BoxSet fast path and must use the coord-list scatter.
        transpose = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert not transpose.is_identity()
        access_tile = AccessTile(
            parent_ref=parent_ref, shape=(4, 4),
            base_map=parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>"),
            coordinate_set=BoxSet(lo=(0, 0), hi=(4, 4)),
            coordinate_order=transpose,
        )


        class _Op:
            attributes = {"_result_shape": (4, 4)}
            operands = ["%a"]

        class _Ctx:
            def __init__(self, inner, at):
                self._inner, self._at = inner, at
            def get_value(self, _):
                return self._at
            def __getattr__(self, n):
                return getattr(self._inner, n)

        seen = []
        real = MemoryOps._flat_memory_offsets

        def _spy(*args, **kwargs):
            seen.append(args)
            return real(*args, **kwargs)

        with patch.object(MemoryOps, "_flat_memory_offsets",
                          staticmethod(_spy)):
            ktdp__load(_Op(), _Ctx(ctx, access_tile), env=None)

        assert len(seen) >= 1, (
            "Non-identity coordinate_order must route to the slow path "
            "and call _flat_memory_offsets — guard symmetry check."
        )


class TestExplicitIdentityCoordinateOrder:
    """The dialect handler's guard is ``cso is None or cso.is_identity()``,
    not ``cso is None`` alone — so a non-None identity AffineMap that
    bypasses the parser-side normalisation still triggers the BoxSet
    fast path.  Verifies the guard symmetry across the parser↔handler
    boundary."""

    def test_identity_affine_map_takes_fast_path(self):
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)
        parent_ref = MemRef(
            base_ptr=ptr, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        # Construct an AccessTile directly with a non-None identity map,
        # bypassing the parser's identity → None normalisation.
        identity_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        assert identity_map.is_identity()
        access_tile = AccessTile(
            parent_ref=parent_ref,
            shape=(4, 4),
            base_map=identity_map,
            coordinate_set=BoxSet(lo=(0, 0), hi=(4, 4)),
            coordinate_order=identity_map,
        )

        # Drive the load handler directly.

        class _Op:
            attributes = {"_result_shape": (4, 4)}
            operands = ["%a"]

        seen = []
        real = MemoryOps._flat_memory_offsets

        def _spy(*args, **kwargs):
            seen.append(args)
            return real(*args, **kwargs)

        class _Ctx:
            def __init__(self, inner, at):
                self._inner = inner
                self._at = at
            def get_value(self, _ssa):
                return self._at
            def __getattr__(self, n):
                return getattr(self._inner, n)

        with patch.object(MemoryOps, "_flat_memory_offsets",
                          staticmethod(_spy)):
            tile = ktdp__load(_Op(), _Ctx(ctx, access_tile), env=None)

        np.testing.assert_array_equal(tile.data, data)
        assert seen == [], (
            "Explicit identity coordinate_order should still trigger the "
            "BoxSet fast path (guard: cso is None or cso.is_identity())."
        )


# ---------------------------------------------------------------------------
# _build_indirect_coords vectorised vs slow-path consistency
# ---------------------------------------------------------------------------

class TestIndirectBoxFastPathConsistency:
    """The vectorised _build_indirect_coords_box must produce the same
    coords (in the same order) as the per-pt Python loop slow path on
    fixtures with mixed direct / direct_expr / indirect dim_subscripts.
    Locks the BoxSet column-by-column reconstruction against the
    pt-major / dim-minor consumption contract of the slow path.
    """

    def test_mixed_direct_expr_and_indirect_matches_slow_path(self):
        """direct + direct_expr + indirect mixed in dim_subscripts.
        The per-sub stride/offset slicing in the box variant must match
        the iter()-based pt-major consumption of the slow path when two
        indirect dims share the same index view (paged-tensor pattern).
        """
        # 2-D variable space (3 x 4); two dim_subscripts share one index view.
        N, M = 3, 4
        box = BoxSet(lo=(0, 0), hi=(N, M))
        # For a shared-view fixture, idx_values[0] is consumed pt-major /
        # dim-minor: 2 consumes per pt, total 2*N*M = 24 entries.
        idx_data = np.array(
            [(p * 7 + d) % 16 for p in range(N * M) for d in range(2)],
            dtype=np.int64,
        )
        idx_values = {0: idx_data}

        # Mock index_view_idx 0 as a tiny MemRef placeholder.
        idx_view_stub = MemRef(
            base_ptr=0, shape=(N, M), strides=[M, 1],
            memory_space="HBM", dtype="i32",
        )

        dim_subs = [
            {"kind": "direct", "var_index": 0},
            {"kind": "direct_expr",
             "subscript": ("add", ("dim", 0), ("mul", 2, ("dim", 1)))},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0), ("dim", 1)]},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0), ("dim", 1)]},
        ]

        # Stub IAT with two index_views entries (only [0] is used here).
        parent = MemRef(
            base_ptr=0, shape=(N, M), strides=[M, 1],
            memory_space="HBM", dtype="f16",
        )
        iat = IndirectAccessTile(
            parent_ref=parent.to_tile_ref(),
            shape=(N, M),
            variables_space_set=box,
            dim_subscripts=dim_subs,
            index_views=[idx_view_stub],
            variables_space_order=None,
        )

        fast = _build_indirect_coords_box(iat, idx_values=idx_values, box=box)

        # Slow path reconstruction by hand using the same iter consumption
        # order: per pt, walk dim_subscripts in order; each indirect dim
        # consumes the next entry from its iv_idx's iterator.
        it = iter(idx_data.tolist())
        slow = []
        for p0 in range(N):
            for p1 in range(M):
                pt = (p0, p1)
                row = []
                for sub in dim_subs:
                    if sub["kind"] == "direct":
                        row.append(pt[sub["var_index"]])
                    elif sub["kind"] == "direct_expr":
                        row.append(eval_subscript_expr(sub["subscript"], pt))
                    elif sub["kind"] == "indirect":
                        row.append(int(next(it)))
                slow.append(tuple(row))
        assert fast == slow

    @pytest.mark.parametrize(
        "expr,expected_fn",
        [
            # floordiv/mod take a bare dim *index* in expr[1] (not a
            # sub-expression — the only tags with this divergent
            # convention).  add/mul/dim are covered transitively by
            # ``test_mixed_direct_expr_and_indirect_matches_slow_path``;
            # neg/max are listed here because they're in the affine
            # grammar but not exercised anywhere else.
            (("floordiv", 0, 4), lambda p0, p1: p0 // 4),
            (("mod", 1, 7), lambda p0, p1: p1 % 7),
            (("neg", ("dim", 0)), lambda p0, p1: -p0),
            (("max", ("dim", 0), ("dim", 1)), lambda p0, p1: max(p0, p1)),
        ],
        ids=["floordiv", "mod", "neg", "max"],
    )
    def test_subscript_eval_parity(self, expr, expected_fn):
        """Pin the vectorised evaluator against ``eval_subscript_expr``
        for each grammar tag whose operand convention or branch isn't
        already covered by other tests.  Without this, the two
        evaluators silently drift when the grammar changes.
        """

        pts = np.array(
            [(p0, p1) for p0 in range(4) for p1 in range(5)], dtype=np.int64
        )
        slow = np.array(
            [eval_subscript_expr(expr, tuple(p)) for p in pts], dtype=np.int64,
        )
        fast = _eval_subscript_vectorized(expr, pts)
        expected = np.array(
            [expected_fn(int(p[0]), int(p[1])) for p in pts], dtype=np.int64,
        )
        np.testing.assert_array_equal(fast, slow)
        np.testing.assert_array_equal(fast, expected)

    def test_negative_indirect_index_raises(self):
        """The vectorised path keeps the slow path's IndexError on
        negative indices (NumPy fancy-indexing wraps; we reject)."""

        box = BoxSet(lo=(0,), hi=(3,))
        idx_view_stub = MemRef(
            base_ptr=0, shape=(3,), strides=[1],
            memory_space="HBM", dtype="i32",
        )
        parent = MemRef(
            base_ptr=0, shape=(3,), strides=[1],
            memory_space="HBM", dtype="f16",
        )
        iat = IndirectAccessTile(
            parent_ref=parent.to_tile_ref(),
            shape=(3,),
            variables_space_set=box,
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view_stub],
            variables_space_order=None,
        )
        # One sub on iv_idx 0 → idx_values[0] is length 3, in pt order.
        idx_values = {0: np.array([1, -2, 0], dtype=np.int64)}

        with pytest.raises(IndexError, match="negative"):
            _build_indirect_coords_box(iat, idx_values=idx_values, box=box)
