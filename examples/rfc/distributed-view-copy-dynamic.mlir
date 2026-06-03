// Dynamic-shape exercise for ktdp.construct_distributed_memory_view.
//
// Composed for end-to-end test coverage of the symbolic-BoxSet path; layout
// pattern derived from examples/rfc/distributed-view-copy.mlir (concrete
// distributed view) and examples/triton-ktir/vector_add_dynamic_ktir.mlir
// (?-shape view bound to a sizes: operand).  Not transcribed from the RFC.
//
// Layout (logical 64x26 row-major):
//   A0 = cols [0, s0),     stored on HBM
//   A1 = cols [s0, 2*s0),  stored on HBM
//   B  = full 64x26,       stored on HBM (output)
//
// s0 is fixed at 13 because construct_access_tile requires a concrete
// access_tile_set; the access tile shape 64x26 covers exactly the global
// domain when 2*s0 = 26.  The global distributed view and B are concrete
// (memref<64x26xf16>) because the regex parser does not accept ? in
// construct_distributed_memory_view's result type.

#A0_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>
#A1_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 - s0 >= 0, -d1 + 2*s0 - 1 >= 0)>
#full   = affine_set<(d0, d1)     : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 25 >= 0)>
#order  = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @distributed_view_copy_dynamic(
      %a0_ptr: index,
      %a1_ptr: index,
      %b_ptr: index,
      %s0_in: i32
  ) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %s0 = arith.index_cast %s0_in : i32 to index

    // (1) Per-partition memory views with symbolic trailing dim.
    %A0_view = ktdp.construct_memory_view %a0_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A0_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    %A1_view = ktdp.construct_memory_view %a1_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A1_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    // (1) Compose the two partitions into a single logical 64x26 distributed
    // view.  Result type is concrete (the regex parser does not accept `?`
    // in the result memref of construct_distributed_memory_view).
    %A_view = ktdp.construct_distributed_memory_view
        (%A0_view, %A1_view : memref<64x?xf16>, memref<64x?xf16>)
        : memref<64x26xf16>

    // (1) Output view B (concrete).
    %B_view = ktdp.construct_memory_view %b_ptr, sizes: [64, 26], strides: [26, 1] {
      coordinate_set = #full,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x26xf16>

    // (2) Direct access tile over the full 64x26 global domain.
    %A_tile = ktdp.construct_access_tile %A_view[%c0, %c0] {
      access_tile_set = #full, access_tile_order = #order
    } : memref<64x26xf16> -> !ktdp.access_tile<64x26xindex>

    %B_tile = ktdp.construct_access_tile %B_view[%c0, %c0] {
      access_tile_set = #full, access_tile_order = #order
    } : memref<64x26xf16> -> !ktdp.access_tile<64x26xindex>

    // (3) Load distributed -> store contiguous.
    %data = ktdp.load %A_tile : !ktdp.access_tile<64x26xindex> -> tensor<64x26xf16>
    ktdp.store %data, %B_tile : tensor<64x26xf16>, !ktdp.access_tile<64x26xindex>

    return
  }
}
