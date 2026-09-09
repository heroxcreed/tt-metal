# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .pack_node import PackNode

from helpers.llk_params import PackerReluType

from .golden import Golden
from .golden_state import tile_operation
from .indexing import InvocationGranularity


class Packer(Golden):
    """Base class for fused test packer code generators.

    Subclasses override methods to emit the C++ LLK calls that configure and
    drive the Pack thread, plus a Python golden function for test validation.

    The pack lifecycle is driven by the planned call nest, which iterates
    over tiles in the block and calls pack() for each one:
        init() -> pack_loop() [which calls pack()] -> uninit()

    To create a new packer:
        1. Subclass Packer
        2. Override get_headers() with the required LLK header files
        3. Override init(), pack(), uninit() to emit the C++ LLK calls
        4. Override golden() to compute the expected pack result, calling
           self.relu_golden() as needed (L1 accumulation is handled by the
           L1AccumOutputTiles output collector)
    """

    # Controls the tile iteration pattern for the pack loop.
    granularity = InvocationGranularity.TILE

    # Set `per_block_init = True` if init() needs block dimensions and must
    # be called per-block inside the batch loop rather than hoisted out.
    per_block_init: bool = False

    pack_mode: str = "PackMode::Default"

    # Set True on packers that untilize dest to row-major L1: the output golden is
    # one untilize over the whole tensor (UntilizePackOutput), not per tile.
    untilizes_l1_output: bool = False

    def golden(
        self,
        call,
        dest,
        output,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        tile = dest.get(call.dest)
        if pack_node.pack_relu != PackerReluType.NoRelu:
            tile = self.relu_golden(tile, config, tile_operation(operation), pack_node)
        # L1 accumulation is handled by the output collector (L1AccumOutputTiles).
        output.write(call.out, tile)

    requires_dest_remap: bool = False

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this packer.

        These headers are #included in the generated test source file. Override to
        return the headers that declare the _llk_pack_*_ functions used by init(),
        pack(), and uninit().
        """
        return []

    def init(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the packer before the pack loop.

        Called once per block. Override to emit the _llk_pack_init_<>()
        calls with the appropriate parameters
        """
        return ""

    def pack(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that packs a single tile from dest to L1.

        Called for each planned invocation. Use
        block.tile_id_block for the dest register index and
        block.tile_id_global for the L1 output buffer index.
        Override to emit the _llk_pack_<>() call.
        """
        return ""

    def uninit(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that uninitializes the packer after the pack loop.

        Called once per block after the pack loop completes. Override if the
        packer requires explicit cleanup.
        """
        return ""
