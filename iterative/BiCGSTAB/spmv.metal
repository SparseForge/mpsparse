//kernels for spmv, spvmv, and inner product

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
    constant uint& num_rows [[buffer(5)]],
    constant uint& num_cols [[buffer(6)]],
    uint gid [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_simdgroup ]],
    uint sid [[ simdgroup_index_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    uint row = (bid*(THREADGROUP_SIZE/SIMD_WIDTH) + sid);

    float p_sum = 0.0;

    if (row < num_rows) {

        int row_start = A_rows[row];
        int row_end = A_rows[row + 1];

        for (int i = row_start + tid; i < row_end; i+= SIMD_WIDTH) {
            int col_index = A_cols[i];
            float val = A_vals[i];
            float other_val = x[col_index];

            p_sum += val*other_val;
        
        }
    }

    float complete_sum = simd_sum(p_sum);

    if (tid == 0 && row < num_rows) {
        b[row] = complete_sum;
    }

}

kernel void inner_product(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device atomic_float* ret [[buffer(2)]],
    constant uint& size [[buffer(3)]], 
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]
) {
    //need to optimize via reduction
    if (gid < size) {
        float prod = a[gid] * b[gid];
        float local_sum = simd_sum(prod);

        if (tid == 0) {
            atomic_fetch_add_explicit(ret, local_sum, memory_order_relaxed);
        }
    }
}

kernel void fused_spvmv(
    device const int* A_rows [[buffer(0)]],
    device const int* A_cols [[buffer(1)]],
    device const float* A_vals [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* y [[buffer(4)]],
    device float* b [[buffer(5)]],
    device atomic_float* ret [[buffer(6)]],
    constant uint& num_rows [[buffer(7)]],
    constant uint& num_cols [[buffer(8)]],
    uint gid [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_simdgroup ]],
    uint sid [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    uint row = (bid*(THREADGROUP_SIZE/SIMD_WIDTH) + sid);
    threadgroup float temp[THREADGROUP_SIZE/SIMD_WIDTH];

    if (lid < (THREADGROUP_SIZE / SIMD_WIDTH)) {
        temp[lid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float p_sum = 0.0;

    if (row < num_rows) {
        int row_start = A_rows[row];
        int row_end = A_rows[row + 1];

        for (int i = row_start + tid; i < row_end; i+= SIMD_WIDTH) {
            int col_index = A_cols[i];
            float val = A_vals[i];
            float other_val = x[col_index];

            p_sum += val*other_val;
        
        }

    }

    float complete_sum = simd_sum(p_sum);

    if (tid == 0 & row < num_rows) {
        b[row] = complete_sum;
        temp[sid] = complete_sum * y[row];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0 && row < num_rows) {
        float thread_group_sum = 0.0;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_sum += temp[i];
        }

        if (thread_group_sum != 0.0 && !isnan(thread_group_sum)) {
            atomic_fetch_add_explicit(ret, thread_group_sum, memory_order_relaxed);
        }
    }
}
