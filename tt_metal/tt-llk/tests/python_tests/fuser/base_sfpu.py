# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .block_data import BlockData
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation
    from .sfpu_node import SfpuNode

from .golden import Golden
from .indexing import InvocationGranularity


class Sfpu(Golden):
    """Base class for fused test SFPU code generators.

    Subclasses represent specific SFPU operations (e.g. UnarySfpu, BinarySfpu)
    and override methods to emit the C++ LLK calls that configure and drive the
    SFPU Unit, plus a Python golden function for test validation.

    Unlike Fpu, SFPU operates on dest register data that was already computed
    by a prior FPU stage or loaded via datacopy. It has no unpacker — SfpuNode
    has no unpacker field at all.

    The lifecycle called by the pipeline is:
        init() -> planned calls to calculate() -> uninit()

    Entirely skipped during UNPACK_ISOLATE, PACK_ISOLATE, and L1_CONGESTION perf runs.

    To create a new SFPU:
        1. Subclass Sfpu
        2. Override get_headers() with the required LLK header files
        3. Override init(), calculate(), uninit() to emit the C++ LLK calls
        4. Override golden() to compute the expected SFPU result, calling
           self.unary_sfpu_golden() or self.binary_sfpu_golden() as needed
    """

    granularity = InvocationGranularity.BLOCK

    input_count = 1

    def golden(
        self,
        call,
        dest,
        compute_unit: "SfpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Apply the SFPU op in place across the whole dest bank (one block).

        SFPU is block-granular: one call transforms every tile the block holds.
        The dest bank stores untilized tiles; they are tilized, run through
        _batch_golden per block, and written back untilized.
        """
        from helpers.tilize_untilize import tilize_block, untilize_block

        tile_shape = operation.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        num_faces = tile_shape.total_num_faces()
        fmt = config.sentinel.golden_math_format

        tile_count = len(dest)
        tilized = [
            tilize_block(
                dest.get(i), tile_dims, fmt, num_faces, tile_dimensions=tile_dims
            ).flatten()
            for i in range(tile_count)
        ]
        block_tensor = torch.cat(tilized)
        block_dims = (tile_count * tile_dims[0], tile_dims[1])

        result = self._batch_golden(
            block_tensor, operation, config, compute_unit, block_dims, tile_count
        )

        tile_size = tilized[0].numel()
        result = result.reshape(tile_count, tile_size)
        for i in range(tile_count):
            dest.set(
                i,
                untilize_block(
                    result[i].flatten(),
                    fmt,
                    tile_dims,
                    tile_dimensions=tile_dims,
                    num_faces=num_faces,
                ).reshape(tile_dims),
            )

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the SFPU before calculation.

        Called once per block. Override to emit the
        _llk_math_eltwise_*_sfpu_init_<>() call.
        """
        return ""

    def calculate(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that performs the SFPU operation.

        Called once per block between init() and uninit().
        Override to emit the sfpu calls.
        """
        return ""

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the SFPU after calculation.

        Called once per block after calculate(). Override if the SFPU
        requires explicit cleanup.
        """
        return ""

    def _batch_golden(
        self,
        tensor: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        """Compute the golden SFPU result for one block of tilized dest data.

        batch_dims and batch_tile_cnt describe the block's tile layout. Called per
        block by golden() above; override with the op's SFPU math.
        """
        return tensor

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this SFPU.

        These headers are #included in the generated test source file.
        Override to return the headers that declare the SFPU functions
        used by init(), calculate() and uninit().
        """
        return []

    def __str__(self) -> str:
        return f"{self.__name__}"
