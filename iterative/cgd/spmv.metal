#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

constant int SIMD_WIDTH = 32;
constant int THREADGROUP_SIZE = 256;

kernel void divides(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        c[0] = a[0] / b[0];
    }
}

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

kernel void weighted_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant float& weight_a [[buffer(2)]],
    constant float& weight_b [[buffer(3)]],
    constant uint& num_elements [[buffer(4)]],
    device float* c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_elements) {
        return;
    }
    c[gid] = weight_a * a[gid] + weight_b * b[gid];
}

kernel void inner_product(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device atomic_float* ret [[buffer(2)]],
    constant uint& size [[buffer(3)]], 
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]
) {
    if (gid < size) {
        float prod = a[gid] * b[gid];
        float local_sum = simd_sum(prod);

        if (tid == 0) {
            atomic_fetch_add_explicit(ret, local_sum, memory_order_relaxed);
        }
    }
}

kernel void zero_out(
    device float* in [[buffer(0)]]
) {
    in[0] = 0.0;
}


kernel void weighted_add_buffer(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* weight_a [[buffer(2)]],
    device const float* weight_b [[buffer(3)]],
    constant uint& num_elements [[buffer(4)]],
    device float* c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_elements) {
        return;
    }
    c[gid] = weight_a[0] * a[gid] + weight_b[0] * b[gid];
}

kernel void iter_update_buffer(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const float* weight_b [[buffer(3)]],
    constant uint& num_elements [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_elements) {
        return;
    }
    if (mode == 0) {
        c[gid] = a[gid] + weight_b[0] * b[gid];
    } else {
        c[gid] = a[gid] - weight_b[0] * b[gid];
    }
}



