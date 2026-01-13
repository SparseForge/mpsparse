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

    if (row >= num_rows) {
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

kernel void update_xr(
    device float* x [[buffer(0)]],
    device float* r [[buffer(1)]],
    device const float* p [[buffer(2)]],
    device const float* Ap [[buffer(3)]],     
    device const float* r_dot_r [[buffer(4)]], 
    device const float* p_Ap [[buffer(5)]],  
    device atomic_float* r_norm_new [[buffer(6)]],
    constant uint& size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]
) {

    float micro_r_sum = 0.0;

    if (gid < size) {
        float alpha = 0.0;

        if (abs(p_Ap[0]) > 1e-9) {
            alpha = r_dot_r[0] / p_Ap[0];
        }

        x[gid] = x[gid] + alpha * p[gid];
        r[gid] = r[gid] - alpha * Ap[gid];
        micro_r_sum = r[gid] * r[gid];
    }

    float local_r_sum = simd_sum(micro_r_sum);

    if (tid == 0) {
        atomic_fetch_add_explicit(r_norm_new, local_r_sum, memory_order_relaxed);
    }
    
}

kernel void update_p(
    device float* p [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const float* r_norm_old [[buffer(2)]],
    device const float* r_norm_new [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) {
        return;
    }
    p[gid] = r[gid] + (r_norm_new[0]/r_norm_old[0]) * p[gid];
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
    device float* in1 [[buffer(0)]],
    device float* in2 [[buffer(1)]]
) {
    in1[0] = 0.0;
    in2[0] = 0.0;
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




kernel void fused_spvmv(
    device const int* A_rows [[buffer(0)]],
    device const int* A_cols [[buffer(1)]],
    device const float* A_vals [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* y [[buffer(4)]],
    device float* b [[buffer(5)]],
    device atomic_float* ret [[buffer(6)]],
    constant uint& num_rows,
    constant uint& num_cols,
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

    if (row >= num_rows) {
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
        temp[sid] = complete_sum * y[row];
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (lid == 0) {
        float thread_group_sum = 0.0;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_sum += temp[i];
        }

        if (thread_group_sum != 0.0 && !isnan(thread_group_sum)) {
            atomic_fetch_add_explicit(ret, thread_group_sum, memory_order_relaxed);
        }

    }



}
