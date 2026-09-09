# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from .fpu_node import FpuNode
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation
    from .operand import Operand
    from .pack_node import PackNode
    from .sfpu_node import SfpuNode

from helpers.golden_generators import (
    BinarySFPUGolden,
    BroadcastGolden,
    DataCopyGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    PackGolden,
    ReduceGolden,
    TransposeGolden,
    UnarySFPUGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    EltwiseBinaryReuseDestType,
    PackerReluType,
    ReduceDimension,
    ReducePool,
    Transpose,
)
from helpers.tilize_untilize import tilize_block, untilize_block

from .golden_state import tile_operation
from .indexing import InvocationGranularity


def _ensure_srcs(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, dimensions: tuple
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tensor_a is None:
        tensor_a = torch.zeros(dimensions)
    if tensor_b is None:
        tensor_b = torch.zeros(dimensions)
    return tensor_a, tensor_b


def _call_count(node: "FpuNode", block_tiles_x: int) -> int:
    return 1 if getattr(node, "custom", False) else block_tiles_x


class Golden:
    """Golden result helpers shared by compute unit base classes."""

    def tilize_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> torch.Tensor:
        src = node.src_a
        return tilize_block(
            tensor,
            src.dimensions,
            src.data_format,
            src.tile_shape.total_num_faces(),
            tile_dimensions=[
                src.tile_shape.total_row_dim(),
                src.tile_shape.total_col_dim(),
            ],
            face_r_dim=src.tile_shape.face_r_dim,
        )

    def transpose_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        use_srcb: bool = False,
    ) -> torch.Tensor:
        operand = node.src_b if use_srcb else node.src_a
        dimensions = (
            operand.dimensions
            if operand is not None
            else operation.max_output_dimensions
        )
        tile_count = (
            operand.tile_count
            if operand is not None
            else (dimensions[0] // operation.tile_shape.total_row_dim())
            * (dimensions[1] // operation.tile_shape.total_col_dim())
        )
        t_matrix = get_golden_generator(TransposeGolden)
        if node.transpose_faces == Transpose.Yes:
            tensor = t_matrix.transpose_faces_multi_tile(
                tensor,
                config.sentinel.golden_math_format,
                tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=dimensions,
            )
        if node.transpose_within_face == Transpose.Yes:
            tensor = t_matrix.transpose_within_faces_multi_tile(
                tensor,
                config.sentinel.golden_math_format,
                tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=dimensions,
            )
        return tensor

    def broadcast_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        operand: "Operand" = None,
        per_block: bool = False,
    ) -> torch.Tensor:
        operand = operand or node.src_b
        if node.broadcast_type == BroadcastType.None_:
            return tensor

        tile_shape = operand.tile_shape
        num_faces = tile_shape.total_num_faces()
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())

        tilized = tilize_block(
            tensor,
            operand.dimensions,
            operand.data_format,
            num_faces,
            tile_dimensions=tile_dims,
        )
        broadcast_generator = get_golden_generator(BroadcastGolden)
        broadcast = broadcast_generator(
            node.broadcast_type,
            tilized,
            operand.data_format,
            num_faces,
            operand.tile_count,
            tile_shape.face_r_dim,
        )

        if per_block:
            tiles = broadcast.view(operand.tile_count_y, operand.tile_count_x, -1)
            for bx in range(0, operand.tile_count_x, operation.block_tiles_x):
                block = tiles[:, bx : bx + operation.block_tiles_x]
                block[:] = block[:, :1]
        elif node.broadcast_tile is not None:
            tiles = broadcast.view(operand.tile_count, -1)
            tiles[:] = tiles[node.broadcast_tile].clone()

        return untilize_block(
            broadcast,
            operand.data_format,
            operand.dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )

    def broadcast_tile_golden(
        self,
        tile: torch.Tensor,
        operation: "L1Operation",
        node: "FpuNode",
        operand: "Operand",
    ) -> torch.Tensor:
        if node.broadcast_type == BroadcastType.None_:
            return tile

        tile_shape = operation.tile_shape
        num_faces = tile_shape.total_num_faces()
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        tilized = tilize_block(
            tile,
            tile_dims,
            operand.data_format,
            num_faces,
            tile_dimensions=tile_dims,
        )
        broadcast = get_golden_generator(BroadcastGolden)(
            node.broadcast_type,
            tilized,
            operand.data_format,
            num_faces,
            1,
            tile_shape.face_r_dim,
        )
        return untilize_block(
            broadcast,
            operand.data_format,
            tile_dims,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        ).reshape(tile_dims)

    def transpose_tile_golden(
        self,
        tile: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> torch.Tensor:
        if node.transpose_faces == Transpose.No and (
            node.transpose_within_face == Transpose.No
        ):
            return tile
        tile_shape = operation.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        t_matrix = get_golden_generator(TransposeGolden)
        result = tile
        if node.transpose_faces == Transpose.Yes:
            result = t_matrix.transpose_faces_multi_tile(
                result,
                config.sentinel.golden_math_format,
                1,
                tilize=True,
                untilize=True,
                input_dimensions=tile_dims,
            )
        if node.transpose_within_face == Transpose.Yes:
            result = t_matrix.transpose_within_faces_multi_tile(
                result,
                config.sentinel.golden_math_format,
                1,
                tilize=True,
                untilize=True,
                input_dimensions=tile_dims,
            )
        return result.reshape(tile_dims)

    def reuse_dest_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            return None, tensor_a
        return tensor_a, tensor_b

    def eltwise_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        accumulate_on_dest: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        math_fidelity = node.math_fidelity

        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_a = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
            tensor_b = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        generate_golden = get_golden_generator(EltwiseBinaryGolden)
        golden_tensor = generate_golden(
            node.fpu.operation,
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            tile_shape=operation.tile_shape,
        ).reshape(operation.max_output_dimensions)

        if accumulate_on_dest or node.acc_to_dest == AccToDest.Yes:
            golden_tensor = golden_tensor + tensor_dst

        return (tensor_a, tensor_b, golden_tensor)

    def datacopy_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if node.broadcast_type != BroadcastType.None_:
            source_tensor = tensor_b
        else:
            source_tensor = tensor_a

        golden_generator = get_golden_generator(DataCopyGolden)
        golden_tensor = golden_generator(
            source_tensor,
            config.sentinel.golden_math_format,
            num_faces=operation.tile_shape.total_num_faces(),
            input_dimensions=operation.max_output_dimensions,
            face_r_dim=operation.tile_shape.face_r_dim,
            tile_shape=operation.tile_shape,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def matmul_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        math_fidelity = node.math_fidelity

        generate_golden = get_golden_generator(MatmulGolden)
        golden = generate_golden(
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            input_A_dimensions=node.src_a.dimensions,
            input_B_dimensions=node.src_b.dimensions,
            tilize=False,
            input_A_format=node.src_a.data_format,
            input_B_format=node.src_b.data_format,
        )

        return (tensor_a, tensor_b, golden)

    def reduce_call_golden(
        self,
        call,
        srcs,
        dest,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Reduce one src tile and place/accumulate it into its dest slot.

        Each tile is reduced fresh (against a zero dest) via reduce_golden, which is
        identical to the batch path for the independent-tile case. For reduce_to_tile
        every call targets the same dest slot, so results are accumulated across the
        block's tiles WITHOUT re-reducing the accumulator: max pools with maximum(),
        sum/average add. That composition matches the batch fold — span and scaler are
        constant, so summing per-tile (avg*span*scaler) equals folding then scaling.
        """
        tensor_a, tensor_b = srcs.pop()
        single = tile_operation(operation)
        dimensions = single.max_output_dimensions
        tensor_a, tensor_b = _ensure_srcs(tensor_a, tensor_b, dimensions)
        _, _, reduced = self.reduce_golden(
            tensor_a, tensor_b, torch.zeros(dimensions), config, single, node
        )
        reduced = reduced.reshape(dimensions)

        if not node.reduce_to_tile:
            dest.set(call.dest, reduced)
            return

        current = dest.get(call.dest)
        if self.reduce_pool == ReducePool.Max:
            dest.set(call.dest, torch.maximum(current, reduced))
        else:
            dest.set(call.dest, current + reduced)

    def reduce_block_max_call_golden(
        self,
        call,
        srcs,
        dest,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Block-max-row reduce one tile and max it into its block-row's first slot.

        block_max reduces every ct-wide row of tiles to a single tile at the row's
        column-0 slot. This runs per tile: row-reduce the tile (fresh, so the result
        is maximum(row_reduce(tile), 0)), then max it into dest[row_base] where
        row_base is the block-local index of the row's first tile. Other slots stay
        zero — matching the batch fold that writes each row's result to column 0.
        """
        tensor_a, tensor_b = srcs.pop()
        single = tile_operation(operation)
        dimensions = single.max_output_dimensions
        tensor_a, tensor_b = _ensure_srcs(tensor_a, tensor_b, dimensions)
        _, _, reduced = self.reduce_golden(
            tensor_a, tensor_b, torch.zeros(dimensions), config, single, node, True
        )
        reduced = reduced.reshape(dimensions)

        ct = dest.block_tiles_x
        row_base = (call.dest // ct) * ct
        dest.set(row_base, torch.maximum(dest.get(row_base), reduced))

    def reduce_block_max_row_unpack_golden(
        self,
        call,
        inputs,
        srcs,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Push every tile of a block row for the ROW-granular block-max reduce.

        One ROW call covers a whole block row; call.in0 is the row's first tile, and
        the row's tiles are contiguous. Push all block_tiles_x of them so the math
        golden can fold the row.
        """
        for k in range(inputs.block_tiles_x):
            srcs.push(inputs.view_a.tile(call.in0 + k), None)

    def reduce_block_max_row_math_golden(
        self,
        call,
        srcs,
        dest,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Fold a block row to a single max tile at the row's first dest slot.

        Row-reduce each of the row's tiles (fresh → maximum(row_reduce, 0)) and max
        them together into dest[call.dest] (the row's column-0 slot); other slots
        stay zero, matching the batch fold.
        """
        single = tile_operation(operation)
        dimensions = single.max_output_dimensions
        result = None
        for _ in range(dest.block_tiles_x):
            tensor_a, tensor_b = srcs.pop()
            tensor_a, tensor_b = _ensure_srcs(tensor_a, tensor_b, dimensions)
            _, _, reduced = self.reduce_golden(
                tensor_a, tensor_b, torch.zeros(dimensions), config, single, node, True
            )
            reduced = reduced.reshape(dimensions)
            result = reduced if result is None else torch.maximum(result, reduced)
        dest.set(call.dest, result)

    def matmul_unpack_call_golden(
        self,
        call,
        inputs,
        srcs,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Assemble this block's A and B sub-matrices and push them as one src pair.

        A matmul call is block-granular: call.in0 is the block's top-left output
        tile id, from which the block's tile-row/col origin follows. The A block is
        those output rows across all K; the B block is all K across those output
        columns. Both are reassembled untilized so the math golden can run a plain
        matmul over them.
        """
        row_dim = node.src_a.tile_shape.total_row_dim()
        col_dim = node.src_a.tile_shape.total_col_dim()
        out_tiles_x = operation.max_output_dimensions[1] // col_dim
        out_tiles_y = operation.max_output_dimensions[0] // row_dim
        row0 = call.in0 // out_tiles_x
        col0 = call.in0 % out_tiles_x
        rt = min(operation.block_tiles_y, out_tiles_y - row0)
        ct = min(operation.block_tiles_x, out_tiles_x - col0)
        kt = node.src_a.dimensions[1] // col_dim
        a_tiles_x = node.src_a.tile_count_x
        b_tiles_x = node.src_b.tile_count_x

        a_block = torch.cat(
            [
                torch.cat(
                    [
                        inputs.view_a.tile((row0 + ay) * a_tiles_x + ax)
                        for ax in range(kt)
                    ],
                    dim=1,
                )
                for ay in range(rt)
            ],
            dim=0,
        )
        b_block = torch.cat(
            [
                torch.cat(
                    [
                        inputs.view_b.tile(by * b_tiles_x + (col0 + bx))
                        for bx in range(ct)
                    ],
                    dim=1,
                )
                for by in range(kt)
            ],
            dim=0,
        )
        srcs.push(a_block, b_block)

    def matmul_math_call_golden(
        self,
        call,
        srcs,
        dest,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Matmul this block's assembled A and B and write each output tile to dest.

        rt/ct come from the assembled block shapes, so remainder blocks (smaller
        than the nominal block) place their tiles at the same dest indices the
        tile-granular packer reads.
        """
        a_block, b_block = srcs.pop()
        row_dim = node.src_a.tile_shape.total_row_dim()
        col_dim = node.src_a.tile_shape.total_col_dim()
        rt = a_block.shape[0] // row_dim
        ct = b_block.shape[1] // col_dim

        generate_golden = get_golden_generator(MatmulGolden)
        result = generate_golden(
            a_block,
            b_block,
            config.sentinel.golden_math_format,
            node.math_fidelity,
            input_A_dimensions=(a_block.shape[0], a_block.shape[1]),
            input_B_dimensions=(b_block.shape[0], b_block.shape[1]),
            tilize=False,
            input_A_format=node.src_a.data_format,
            input_B_format=node.src_b.data_format,
        ).reshape(a_block.shape[0], b_block.shape[1])

        for ay in range(rt):
            for ax in range(ct):
                tile = result[
                    ay * row_dim : (ay + 1) * row_dim,
                    ax * col_dim : (ax + 1) * col_dim,
                ]
                dest.set(ay * ct + ax, tile)

    def row_math_call_golden(
        self,
        call,
        srcs,
        dest,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Apply the unit's per-tile math to each tile of a block row (ROW granularity).

        Pops one src pair per tile and runs the unit's own _batch_golden on a single
        tile, writing dest[call.dest + k] for each of the row's block_tiles_x tiles.
        Custom (loop_spec) ops run the golden per tile, so one tile per call.
        """
        count = _call_count(node, dest.block_tiles_x)
        single = tile_operation(operation)
        dimensions = single.max_output_dimensions
        for k in range(count):
            tensor_a, tensor_b = srcs.pop()
            tensor_a, tensor_b = _ensure_srcs(tensor_a, tensor_b, dimensions)
            _, _, result = self._batch_golden(
                tensor_a, tensor_b, dest.get(call.dest + k), single, config, node
            )
            dest.set(call.dest + k, result.reshape(dimensions))

    def tilize_unpack_call_golden(
        self,
        call,
        inputs,
        srcs,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Tilize one src_a tile into the source registers.

        The tilize unpack path reads row-major L1 and tilizes into src. Each tile is
        tilized on its own (tilize is per-tile independent) and pushed; downstream math
        copies it through and the output is collected by TilizedOutputTiles, matching
        the batch tilize_golden then reshape.
        """
        src = node.src_a
        tile_shape = src.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        count = (
            inputs.block_tiles_x
            if self.granularity == InvocationGranularity.ROW
            and not getattr(node, "custom", False)
            else 1
        )
        for k in range(count):
            tilized = tilize_block(
                inputs.tile_a(call.in0 + k),
                tile_dims,
                src.data_format,
                tile_shape.total_num_faces(),
                tile_dimensions=tile_dims,
                face_r_dim=tile_shape.face_r_dim,
            ).reshape(tile_dims)
            srcs.push(tilized, None)

    def sub_bcast_col_unpack_golden(
        self,
        call,
        inputs,
        srcs,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Broadcast src_b's block tile and push it with every src_a tile of the row.

        Column-broadcast sub reuses one broadcast src_b tile across the whole block
        row. call.in1 is the row's first src_b tile; broadcast it once and pair it
        with each of the row's block_tiles_x src_a tiles.
        """
        b_broadcast = self.broadcast_tile_golden(
            inputs.tile_b(call.in1), operation, node, node.src_b
        )
        count = _call_count(node, inputs.block_tiles_x)
        for k in range(count):
            srcs.push(inputs.tile_a(call.in0 + k), b_broadcast)

    def sub_bcast_col_math_golden(
        self,
        call,
        srcs,
        dest,
        node: "FpuNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Subtract the broadcast src_b from each src_a tile of the block row."""
        count = _call_count(node, dest.block_tiles_x)
        single = tile_operation(operation)
        dimensions = single.max_output_dimensions
        for k in range(count):
            tensor_a, tensor_b = srcs.pop()
            _, _, result = self.eltwise_golden(
                tensor_a, tensor_b, torch.zeros(dimensions), config, single, node
            )
            dest.set(call.dest + k, result.reshape(dimensions))

    def untilize_pack_call_golden(
        self,
        call,
        dest,
        output,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Write each tile of a block row (relu'd, raw) to the output buffer.

        Block-row-granular: call.dest/call.out are the row's first dest slot and output
        tile. The actual untilize happens once over the whole output in
        UntilizePackOutput.finish (matching the batch untilize_golden), so here we only
        apply relu and hand the raw tile to the collector.
        """
        single = tile_operation(operation)
        for k in range(dest.block_tiles_x):
            tile = dest.get(call.dest + k)
            if pack_node.pack_relu != PackerReluType.NoRelu:
                tile = self.relu_golden(tile, config, single, pack_node)
            output.write(call.out + k, tile)

    def matmul_pack_call_golden(
        self,
        call,
        dest,
        output,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> None:
        """Pack every tile of a matmul block from dest to the output buffer.

        A block-granular matmul packer fires once per block: call.out is the block's
        top-left output tile id. Each of the block's rt*ct dest tiles is packed (with
        relu/l1-acc via _batch_golden) to its global output tile position.
        """
        out_tiles_x = pack_node.output.tile_count_x
        out_tiles_y = pack_node.output.tile_count_y
        row0 = call.out // out_tiles_x
        col0 = call.out % out_tiles_x
        rt = min(operation.block_tiles_y, out_tiles_y - row0)
        ct = min(operation.block_tiles_x, out_tiles_x - col0)
        single = tile_operation(operation)

        for ty in range(rt):
            for tx in range(ct):
                tile = dest.get(ty * ct + tx)
                if pack_node.pack_relu != PackerReluType.NoRelu:
                    tile = self.relu_golden(tile, config, single, pack_node)
                output.write((row0 + ty) * out_tiles_x + (col0 + tx), tile)

    def reduce_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        block_max: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        tile_shape = operation.tile_shape
        dimensions = operation.max_output_dimensions
        num_faces = tile_shape.total_num_faces()
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        grid_y, grid_x = dimensions[0] // tile_dims[0], dimensions[1] // tile_dims[1]
        reduce_dim = ReduceDimension.Row if block_max else node.fpu.reduce_dim
        pool_type = ReducePool.Max if block_max else node.fpu.reduce_pool
        pool = torch.amax if pool_type == ReducePool.Max else torch.sum
        generate_golden = get_golden_generator(ReduceGolden)

        def reduce(tensor: torch.Tensor, fold_blocks: bool) -> torch.Tensor:
            reduced = tilize_block(
                tensor, dimensions, output_format, num_faces, tile_dimensions=tile_dims
            ).flatten()
            reduced = generate_golden(
                reduced,
                reduce_dim,
                pool_type,
                output_format,
                tile_cnt=grid_y * grid_x,
                tile_shape=tile_shape,
            ).flatten()

            if fold_blocks:
                tiles = reduced.view(grid_y, grid_x, -1)
                for by in range(0, grid_y, operation.block_tiles_y):
                    for bx in range(0, grid_x, operation.block_tiles_x):
                        block = tiles[
                            by : by + operation.block_tiles_y,
                            bx : bx + operation.block_tiles_x,
                        ]
                        folded = pool(block, dim=1)
                        if not block_max:
                            folded = pool(folded, dim=0, keepdim=True)
                        block[:] = 0
                        block[: len(folded), 0] = folded

            return untilize_block(
                reduced,
                output_format,
                dimensions,
                tile_dimensions=tile_dims,
                num_faces=num_faces,
            ).flatten()

        src_reduced = reduce(tensor_a, block_max or node.reduce_to_tile)
        dest_reduced = reduce(tensor_dst, block_max)

        if pool_type == ReducePool.Average:
            span = tile_dims[1] if reduce_dim == ReduceDimension.Row else tile_dims[0]
            scaler = tensor_b.flatten()[0].item()
            golden_tensor = (src_reduced * span + dest_reduced) * scaler
        else:
            golden_tensor = pool(torch.stack((src_reduced, dest_reduced)), dim=0)

        return (tensor_a, tensor_b, golden_tensor.to(src_reduced.dtype))

    def unary_sfpu_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "SfpuNode",
        batch_dims: tuple,
    ) -> torch.Tensor:
        sfpu = node.sfpu
        format_input = config.sentinel.golden_math_format
        format_output = config.sentinel.golden_math_format
        dest_acc = config.dest_acc

        generate_sfpu_golden = get_golden_generator(UnarySFPUGolden)

        return generate_sfpu_golden(
            sfpu.operation,
            tensor,
            format_output,
            dest_acc,
            format_input,
            batch_dims,
            sfpu.iterations,
            sfpu.dest_idx,
            sfpu.fill_const_value,
            skip_tilize=True,
        )

    def binary_sfpu_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "SfpuNode",
        batch_dims: tuple,
    ) -> torch.Tensor:
        sfpu = node.sfpu
        math_format = config.sentinel.golden_math_format

        generate_binary_golden = get_golden_generator(BinarySFPUGolden)

        return generate_binary_golden(
            sfpu.operation,
            tensor,
            sfpu.dst_index_in0,
            sfpu.dst_index_in1,
            sfpu.dst_index_out,
            sfpu.iterations,
            batch_dims,
            math_format,
            skip_tilize=True,
        )

    def untilize_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "PackNode",
    ) -> torch.Tensor:
        untilize = get_golden_generator(UntilizeGolden)
        tile_shape = node.output.tile_shape
        return untilize(
            tensor,
            node.output.data_format,
            dimensions=node.output.dimensions,
            tile_dimensions=(tile_shape.total_row_dim(), tile_shape.total_col_dim()),
        )

    def relu_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "PackNode",
    ) -> torch.Tensor:
        intermediate_format = config.sentinel.golden_pack_src
        relu_config = PackGolden.generate_relu_config(
            node.pack_relu, node.relu_threshold, intermediate_format
        )
        return PackGolden.apply_relu(tensor, relu_config, intermediate_format)
