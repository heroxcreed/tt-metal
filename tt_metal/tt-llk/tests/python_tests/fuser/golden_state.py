# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import TYPE_CHECKING, List, Optional

import torch
from helpers.llk_params import format_dict
from helpers.tilize_untilize import tilize_block, untilize_block

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .operand import Operand


def tile_dimensions(tile_shape) -> tuple:
    return (tile_shape.total_row_dim(), tile_shape.total_col_dim())


def tile_operation(operation: "L1Operation") -> "L1Operation":
    dimensions = tile_dimensions(operation.tile_shape)
    single = copy(operation)
    single.max_output_dimensions = dimensions
    single.block_size = dimensions
    single.block_tiles_x = 1
    single.block_tiles_y = 1
    return single


class OperandTiles:
    def __init__(self, operand: "Operand", tensor: torch.Tensor):
        self.operand = operand
        self.tile_dims = tile_dimensions(operand.tile_shape)
        self.num_faces = operand.tile_shape.total_num_faces()
        self._tiles = tilize_block(
            tensor,
            operand.dimensions,
            operand.data_format,
            num_faces=self.num_faces,
            tile_dimensions=self.tile_dims,
        ).view(operand.tile_count, -1)

    def tile(self, index: int) -> torch.Tensor:
        return untilize_block(
            self._tiles[index].flatten(),
            self.operand.data_format,
            self.tile_dims,
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        ).reshape(self.tile_dims)


class OutputTiles:
    def __init__(self, operand: "Operand"):
        self.operand = operand
        self.tile_dims = tile_dimensions(operand.tile_shape)
        self.num_faces = operand.tile_shape.total_num_faces()
        self._tiles = torch.zeros(
            (operand.tile_count, operand.tile_shape.total_tile_size()),
            dtype=format_dict[operand.data_format],
        )

    def write(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = tilize_block(
            tile.reshape(self.tile_dims),
            self.tile_dims,
            self.operand.data_format,
            num_faces=self.num_faces,
            tile_dimensions=self.tile_dims,
        )[0]

    def finish(self) -> torch.Tensor:
        return untilize_block(
            self._tiles.flatten(),
            self.operand.data_format,
            self.operand.dimensions,
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        )


class L1AccumOutputTiles(OutputTiles):
    """Output collector for L1 accumulation: every block packs into the same output
    tiles, so writes accumulate. The pack plan makes call.out block-independent, so
    each block's tiles land on the first block's positions and sum there (others stay
    zero), matching the batch l1_acc_golden."""

    def write(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = (
            self._tiles[index]
            + tilize_block(
                tile.reshape(self.tile_dims),
                self.tile_dims,
                self.operand.data_format,
                num_faces=self.num_faces,
                tile_dimensions=self.tile_dims,
            )[0]
        )


class TilizedOutputTiles:
    """Output collector for a tilized L1 result (e.g. the tilize unpack path).

    The device writes tilized data straight to L1, and the golden compares it as
    tile-concatenated data reshaped to the operand's shape (which stacks each tile's
    tilized block into consecutive rows) — NOT the row-major, side-by-side arrangement
    an untilize would produce. So tiles are stored verbatim and concatenated in tile
    order, mirroring the batch path's reshape of the tilized math result.
    """

    def __init__(self, operand: "Operand"):
        self.operand = operand
        self._tiles = [None] * operand.tile_count

    def write(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = tile.flatten()

    def finish(self) -> torch.Tensor:
        return torch.cat(self._tiles).reshape(self.operand.dimensions)


class UntilizePackOutput:
    """Output collector for the untilize packer.

    The untilize packer's golden is untilize_block over the WHOLE row-major output
    (the original batch untilize_golden), which is NOT the same as untilizing each
    tile: for >1 tile row the whole-tensor untilize reinterprets the row-major flat
    as tile-concatenated. So collect the raw (relu'd) tiles, arrange them into the
    output shape, and untilize the whole thing once at finish.
    """

    def __init__(self, operand: "Operand"):
        self.operand = operand
        self.tile_dims = tile_dimensions(operand.tile_shape)
        self.num_faces = operand.tile_shape.total_num_faces()
        self._tiles = [
            torch.zeros(self.tile_dims, dtype=format_dict[operand.data_format])
            for _ in range(operand.tile_count)
        ]

    def write(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = tile.reshape(self.tile_dims)

    def finish(self) -> torch.Tensor:
        rows, cols = self.tile_dims
        height, width = self.operand.dimensions
        whole = torch.zeros(height, width, dtype=format_dict[self.operand.data_format])
        for i, tile in enumerate(self._tiles):
            ty, tx = divmod(i, self.operand.tile_count_x)
            whole[ty * rows : (ty + 1) * rows, tx * cols : (tx + 1) * cols] = tile
        return untilize_block(
            whole.flatten(),
            self.operand.data_format,
            self.operand.dimensions,
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        )


class SourceRegisters:
    def __init__(self):
        self.a: List[torch.Tensor] = []
        self.b: List[torch.Tensor] = []

    def push(self, tile_a: Optional[torch.Tensor], tile_b: Optional[torch.Tensor]):
        if tile_a is not None:
            self.a.append(tile_a)
        if tile_b is not None:
            self.b.append(tile_b)

    def pop(self):
        return (
            self.a.pop(0) if self.a else None,
            self.b.pop(0) if self.b else None,
        )


class DestBank:
    def __init__(
        self,
        tiles: int,
        tile_dims: tuple,
        dtype,
        block_tiles_x: int = None,
        block_tiles_y: int = None,
    ):
        self.tile_dims = tile_dims
        # Actual extents of the region this bank holds, so block/row-granular
        # goldens can locate a call's row/column within the block (remainder aware).
        self.block_tiles_x = block_tiles_x
        self.block_tiles_y = block_tiles_y
        self._tiles = [torch.zeros(tile_dims, dtype=dtype) for _ in range(tiles)]

    def __len__(self) -> int:
        return len(self._tiles)

    def get(self, index: int) -> torch.Tensor:
        return self._tiles[index].clone()

    def set(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = tile.reshape(self.tile_dims).clone()


class Inputs:
    def __init__(
        self,
        view_a: Optional[OperandTiles],
        view_b: Optional[OperandTiles],
        block_tiles_x: int = None,
        block_tiles_y: int = None,
    ):
        self.view_a = view_a
        self.view_b = view_b
        # Actual extents of the region, so row/block-granular unpack goldens can
        # gather all the tiles a single call covers (remainder aware).
        self.block_tiles_x = block_tiles_x
        self.block_tiles_y = block_tiles_y

    def tile_a(self, index) -> Optional[torch.Tensor]:
        if index is None or self.view_a is None:
            return None
        return self.view_a.tile(index)

    def tile_b(self, index) -> Optional[torch.Tensor]:
        if index is None or self.view_b is None:
            return None
        return self.view_b.tile(index)
