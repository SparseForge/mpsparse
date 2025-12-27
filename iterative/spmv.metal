#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

constant int SIMD_WIDTH = 32;
constant int THREADGROUP_SIZE = 256;

kernel void spmv_op(
    device const int* A_rows [[buffer(0)]],
    device const int* A_cols [[buffer(1)]],
    device const float* A_vals [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* b [[buffer(4)]],
    constant uint& num_rows,
    constant uint& num_cols,
    uint gid [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_simdgroup ]],
    uint sid [[ simdgroup_index_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    uint row = (bid*(THREADGROUP_SIZE/SIMD_WIDTH) + sid);

    if (row > num_rows) {
        return;
    }

    int row_start = A_rows[row];
    int row_end = A_rows[row + 1];

    float p_sum = 0.0;

    for (int i = row_start + tid; i < row_end; i+= SIMD_WIDTH) {
        int col_index = A_cols[i];
        float val = A_vals[i];
        float other_val = x[col_index];

        p_sum += val*other_val;
    
    }

    float complete_sum = simd_sum(p_sum);

    if (tid == 0) {
        b[row] = complete_sum;
    }

}