//BiCGSTAB-specific kernels

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

constant int SIMD_WIDTH = 32;
constant int THREADGROUP_SIZE = 256;
constant float STAB_TOLERANCE = 1e-15f; 

kernel void upp(
    device float* p [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* scalars [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    //In the future, we should probably figure out a way to fuse the spmv (v = Ap) step with this step to reduce overhead
    //However, not implemented right now :sob:

    threadgroup float rho_new;
    threadgroup float rho_old;
    threadgroup float alpha;
    threadgroup float omega;

    if (lid == 0) {
        if (mode > 0) {
            rho_new = scalars[(mode - 1) % 2];
            rho_old = scalars[mode % 2];
            alpha = scalars[2];
            omega = scalars[3];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < size) {
        if (mode == 0) {
            p[gid] = r[gid];
        } 
        else {
            float beta = 0.0f;
            
            bool stable_denominator = (abs(rho_old) > STAB_TOLERANCE) && (abs(omega) > STAB_TOLERANCE);

            if (stable_denominator) {
                beta = (rho_new / rho_old) * (alpha / omega);
                
                if (isinf(beta) || isnan(beta)) {
                    beta = 0.0f;
                }
            }
            
            p[gid] = r[gid] + beta * (p[gid] - omega * v[gid]);
        }
    }

    if (gid == 0) {
        scalars[4] = 0.0f; 
        scalars[5] = 0.0f; 
        scalars[6] = 0.0f; 
        scalars[7] = 0.0f; 
    }
}

kernel void f_upalpha_ups(
    device float* s [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* scalars [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    //In the future, we should probably figure out a way to fuse the spmv (t = As) step with this step to reduce overhead
    //However, not implemented right now :sob:

    threadgroup float alpha;
    if (lid == 0) {
        alpha = scalars[(mode+1)%2] / scalars[4];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < size) {
        s[gid] = r[gid] - alpha * v[gid];
    }

    if (gid == 0) {
        scalars[2] = alpha;

        //also zero out rho_prev
        scalars[mode%2] = 0.0f;
    }

}
kernel void f_dip(
    device const float* s [[buffer(0)]],
    device const float* t [[buffer(1)]],
    device float* scalars [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    //given t, s compute t * s and t * t

    threadgroup float temp_st[THREADGROUP_SIZE/SIMD_WIDTH];
    threadgroup float temp_tt[THREADGROUP_SIZE/SIMD_WIDTH];
    float prod_st = 0.0;
    float prod_tt = 0.0;
    
    if (lid < (THREADGROUP_SIZE / SIMD_WIDTH)) {
        temp_st[lid] = 0.0f;
        temp_tt[lid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < size) {
        prod_st = s[gid] * t[gid];
        prod_tt = t[gid] * t[gid];
    }

    float local_sum_st = simd_sum(prod_st);
    float local_sum_tt = simd_sum(prod_tt);

    if (tid == 0) {
        temp_st[sid] = local_sum_st;
        temp_tt[sid] = local_sum_tt;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    //reduction

    if (lid == 0) {
        float thread_group_sum_st = 0.0f;
        float thread_group_sum_tt = 0.0f;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_sum_st += temp_st[i];
            thread_group_sum_tt += temp_tt[i];
        }

        if (thread_group_sum_st != 0.0 && !isnan(thread_group_sum_st)) {
            atomic_fetch_add_explicit((device atomic_float*) &scalars[5], thread_group_sum_st, memory_order_relaxed);
        }

        if (thread_group_sum_tt != 0.0 && !isnan(thread_group_sum_tt)) {
            atomic_fetch_add_explicit((device atomic_float*) &scalars[6], thread_group_sum_tt, memory_order_relaxed);
        }
    }   
}

kernel void f_dspvmv(
    device const int* A_rows [[buffer(0)]],
    device const int* A_cols [[buffer(1)]],
    device const float* A_vals [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* y [[buffer(4)]],
    device const float* z [[buffer(5)]],
    device float* b [[buffer(6)]],
    device atomic_float* ret_y [[buffer(7)]],
    device atomic_float* ret_z [[buffer(8)]],
    constant uint& num_rows [[buffer(9)]],
    constant uint& num_cols [[buffer(10)]],
    uint gid [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_simdgroup ]],
    uint sid [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    //computes y A x and z A x, putting intermediate into b and final into ret_y, ret_z
    //don't use yet, with alterations might be useful later
    uint row = (bid*(THREADGROUP_SIZE/SIMD_WIDTH) + sid);
    threadgroup float temp_y[THREADGROUP_SIZE/SIMD_WIDTH];
    threadgroup float temp_z[THREADGROUP_SIZE/SIMD_WIDTH];

    if (lid < (THREADGROUP_SIZE / SIMD_WIDTH)) {
        temp_y[lid] = 0.0f;
        temp_z[lid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= num_rows) {
        return;
    }

    int row_start = A_rows[row];
    int row_end = A_rows[row + 1];

    float p_sum = 0.0f;

    for (int i = row_start + tid; i < row_end; i+= SIMD_WIDTH) {
        int col_index = A_cols[i];
        float val = A_vals[i];
        float other_val = x[col_index];

        p_sum += val*other_val;
    
    }

    float complete_sum = simd_sum(p_sum);

    if (tid == 0) {
        b[row] = complete_sum;
        temp_y[sid] = complete_sum * y[row];
        temp_z[sid] = complete_sum * z[row];
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);


    //reduction

    if (lid == 0) {
        float thread_group_sum_y = 0.0;
        float thread_group_sum_z = 0.0;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_sum_y += temp_y[i];
            thread_group_sum_z += temp_z[i];
        }

        if (thread_group_sum_y != 0.0 && !isnan(thread_group_sum_y)) {
            atomic_fetch_add_explicit(ret_y, thread_group_sum_y, memory_order_relaxed);
        }
        if (thread_group_sum_z != 0.0 && !isnan(thread_group_sum_z)) {
            atomic_fetch_add_explicit(ret_z, thread_group_sum_z, memory_order_relaxed);
        }
    }
}

kernel void f_upomega_upxr_laa(
    device float* x [[buffer(0)]],
    device float* r [[buffer(1)]],
    device const float* p [[buffer(2)]],
    device const float* r_0 [[buffer(3)]],
    device const float* s [[buffer(4)]],
    device const float* t [[buffer(5)]],
    device float* scalars [[buffer(6)]],
    constant uint& size [[buffer(7)]],
    constant uint& mode [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float omega; 
    threadgroup float alpha; 

    if (lid == 0) {
        float phi2 = scalars[6];
        if (abs(phi2) < 1e-20) {
            omega = 0.0f; 
        } else {
            omega = scalars[5] / phi2;
        }
        alpha = scalars[2];
    }

    float micro_rho_sum = 0.0;
    float micro_norm_sum = 0.0;

    threadgroup float temp_norm[THREADGROUP_SIZE/SIMD_WIDTH];
    threadgroup float temp_rho[THREADGROUP_SIZE/SIMD_WIDTH];
    
    if (lid < (THREADGROUP_SIZE / SIMD_WIDTH)) {
        temp_rho[lid] = 0.0f;
        temp_norm[lid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < size) {
        x[gid] = x[gid] + alpha * p[gid] + omega * s[gid];
        r[gid] = s[gid] - omega * t[gid];

        micro_rho_sum = r_0[gid] * r[gid];
        micro_norm_sum = r[gid] * r[gid];
    }

    float local_rho_sum = simd_sum(micro_rho_sum);
    float local_norm_sum = simd_sum(micro_norm_sum);

    if (tid == 0) {
        temp_norm[sid] = local_norm_sum;
        temp_rho[sid] = local_rho_sum;
    }
    if (gid == 0) {
        scalars[3] = omega;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    //reduction

     if (lid == 0) {
        float thread_group_rho_sum = 0.0f;
        float thread_group_norm_sum = 0.0f;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_rho_sum += temp_rho[i];
            thread_group_norm_sum += temp_norm[i];
        }

        if (thread_group_rho_sum != 0.0 && !isnan(thread_group_rho_sum)) {
            atomic_fetch_add_explicit((device atomic_float*) &scalars[mode%2], thread_group_rho_sum, memory_order_relaxed);
        }

        if (thread_group_norm_sum != 0.0 && !isnan(thread_group_norm_sum)) {
            atomic_fetch_add_explicit((device atomic_float*) &scalars[7], thread_group_norm_sum, memory_order_relaxed);
        }
    }      
}

